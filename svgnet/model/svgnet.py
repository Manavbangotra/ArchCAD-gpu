"""
Source: https://github.com/nicehuster/SymPointV2/blob/master/svgnet/model/svgnet.py

Contains substantial modifications to the original code.

NOTICE: Original repository does not specify a license.
This derivative work is for research purposes only.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ..util import cuda_cast
from .pointtransformer import Model as PointT
#from .pointnet2 import Model as PointT
from .decoder import Decoder
from ..vision.vision_embed import ImgEmbed #
from .label_space import target_rows, usable


import numpy as np

class SVGNet(nn.Module):
    def __init__(
        self,cfg,criterion=None):
        super().__init__()
        self.criterion = criterion

        self.image_embed = ImgEmbed(cfg) #


        # NOTE backbone
        self.backbone = PointT(cfg)
        self.decoder = Decoder(cfg,self.backbone.planes)
        self.num_classes = cfg.semantic_classes
        self.test_object_score = 0.1
        # Head columns each target id covers (coarse ids -> their members).
        self.register_buffer("target_rows", target_rows(self.num_classes), persistent=False)
        
    def train(self, mode=True):
        return super().train(mode)   # nn.Module returns self; .eval() chains
        
    def forward(self, batch,return_loss=True):
        # 9 fields from hand-built batches (tools/inference.py); 10 when the
        # loader adds per-batch source metadata ahead of the file names.
        coords,feats,semantic_labels,offsets,lengths,layerIds, imgs, center, *rest = batch
        json_file = rest[-1]
        meta = rest[0] if len(rest) == 2 else None
        return self._forward(coords,feats,offsets,semantic_labels,lengths,layerIds,imgs,center,json_file,
                             return_loss=return_loss, meta=meta)

     
    def prepare_targets(self,semantic_labels,bg_ind=-1,bg_sem=None,meta=None):
        # Background id follows the class count, so this works for both the
        # 30-class ArchCAD setup and the 35-class FloorPlanCAD setup.
        if bg_sem is None:
            bg_sem = self.num_classes

        instance_ids = semantic_labels[:,1].cpu().numpy()
        semantic_ids = semantic_labels[:,0].cpu().numpy()
        svg_len = semantic_ids.shape[0]

        # One target per distinct (semantic, instance) pair, in order of first
        # appearance. This was a Python list-membership scan plus a set
        # intersection per key -- O(N x K) interpreter work, seconds per step on
        # a 6,000-primitive US tile with hundreds of instances.
        pairs = np.stack([semantic_ids, instance_ids], axis=1)
        uniq, first, inverse = np.unique(pairs, axis=0, return_index=True, return_inverse=True)
        inverse = inverse.reshape(-1)
        order = np.argsort(first, kind="stable")

        keep = [k for k in order if usable(self.target_rows, int(uniq[k, 0]), bg_sem)]
        if keep:
            col = {k: j for j, k in enumerate(keep)}
            lut = np.full(len(uniq), -1, dtype=np.int64)
            for k, j in col.items():
                lut[k] = j
            which = torch.from_numpy(lut[inverse])
            hit = which >= 0
            mask_targets = torch.zeros(svg_len, len(keep))
            mask_targets[torch.nonzero(hit).squeeze(1), which[hit]] = 1
            cls_targets = torch.tensor([int(uniq[k, 0]) for k in keep])
        else:
            cls_targets = torch.tensor([bg_sem])
            mask_targets = torch.zeros(svg_len, 1)

        target = {
            "labels": cls_targets.to(semantic_labels.device),
            "masks": mask_targets.to(semantic_labels.device),
            "label_rows": self.target_rows.to(semantic_labels.device)[cls_targets.to(semantic_labels.device)],
        }
        if meta is not None and meta.get("annotated") is not None:
            target["annotated"] = meta["annotated"].to(semantic_labels.device)
        return [target]

    def _prepare_targets_reference(self,semantic_labels,bg_ind=-1,bg_sem=None):
        """The original per-key loop, kept only so the test can pin the
        vectorised version to it."""
        if bg_sem is None:
            bg_sem = self.num_classes

        instance_ids = semantic_labels[:,1].cpu().numpy()
        semantic_ids = semantic_labels[:,0].cpu().numpy()

        keys = []
        for sem_id,ins_id in zip(semantic_ids,
                             instance_ids):
            if (sem_id,ins_id) not in keys:
                keys.append((sem_id,ins_id))

        cls_targets,mask_targets = [], []
        svg_len = semantic_ids.shape[0]

        for (sem_id,ins_id) in keys:
            # Every id at or above bg_sem is out, regardless of instance.
            # The coarse band-C ids (door-any, furniture-any, ...) sit above it
            # and the classifier head has no column for them, so a target
            # carrying one indexes past the head -- a device-side assert in the
            # matcher, not a recoverable error. They do now carry instance ids,
            # because the annotation UI needs whole objects to select, which is
            # why this can no longer also test ins_id == -1. The loader's
            # coarse_policy normally converts them long before here; this is the
            # backstop.
            if sem_id>=bg_sem: continue


            tensor_mask = torch.zeros(svg_len)
            ind1 = np.where(semantic_ids==sem_id)[0]
            ind2 = np.where(instance_ids==ins_id)[0]
            ind = list(set(ind1).intersection(ind2))
            tensor_mask[ind] = 1
            cls_targets.append(sem_id)
            mask_targets.append(tensor_mask.unsqueeze(1))

        cls_targets = torch.tensor(cls_targets) if cls_targets else torch.tensor([bg_sem])   #
        mask_targets = torch.cat(mask_targets,dim=1) if mask_targets else torch.zeros(svg_len,1)
        
        
        return [{
            "labels": cls_targets.to(semantic_labels.device),
            "masks": mask_targets.to(semantic_labels.device),

        }]

    @cuda_cast
    def _forward(
        self,
        coords,   # [x, 3]
        feats,    # [x, 7]
        offsets,    # [2]
        semantic_labels,
        lengths,
        layerIds,
        imgs,
        centers,
        json_file,
        return_loss=True,
        meta=None,
    ):

        img_embed = self.image_embed(imgs, centers) #

        stage_list={'inputs': {'p_out':coords,"f_out":feats,"offset":offsets},"semantic_labels":semantic_labels[:,0]}
        targets = self.prepare_targets(semantic_labels, meta=meta)
        stage_list.update({"tgt":targets})
        
        stage_list = self.backbone(stage_list)

        outputs = self.decoder(stage_list,layerIds,img_embed,json_file)

        model_outputs = {}
        if not self.training:
            semantic_scores=self.semantic_inference(outputs["pred_logits"],outputs["pred_masks"])
            instances = self.instance_inference(outputs["pred_logits"],outputs["pred_masks"])
            model_outputs.update(
                dict(
                semantic_scores=semantic_scores,
                ), 
            )
       
            model_outputs.update(
                dict(
                semantic_labels=semantic_labels[:,0],
                    ), 
             )
            model_outputs.update(
                dict(
                instances=instances,
                ),
            )

            model_outputs.update(
                dict(
                targets=targets[0],
                ),
            )
            model_outputs.update(
                dict(
                lengths=lengths,
                ),
            )
         
        
        if not return_loss:
            return model_outputs
        # NOTE cal loss
        
        losses = self.criterion(outputs,targets)
        loss_value,loss_dicts = self.parse_losses(losses)
        
        
        return model_outputs,loss_value,loss_dicts

    
    def semantic_inference(self, mask_cls, mask_pred):
        
        mask_cls = F.softmax(mask_cls, dim=-1)[...,:-1] # Q,C
        mask_pred = mask_pred.sigmoid() # Q,G
        semseg = torch.einsum("bqc,bqg->bgc", mask_cls, mask_pred)
        return semseg[0]

    def instance_inference(self,mask_cls,mask_pred,overlap_threshold=0.8):
        
        mask_cls,mask_pred = mask_cls[0],mask_pred[0]
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.num_classes) & (scores >= self.test_object_score)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep][:, :-1]

        cur_prob_masks = cur_scores[..., None] * cur_masks
        current_segment_id = 0
        nline = cur_masks.shape[-1]

        results = []
        # take argmax
        try:
            cur_mask_ids = cur_prob_masks.argmax(0)
        except: 
            return results
        
        for k in range(cur_classes.shape[0]):

            pred_class = cur_classes[k].item()
            pred_score = cur_scores[k].item()
            mask_area = (cur_mask_ids == k).sum().item()
            original_area = (cur_masks[k] >= 0.5).sum().item()
            mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)
            if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                if mask_area / original_area < overlap_threshold:
                    continue
                current_segment_id += 1
                #print(pred_class, pred_score)
                results.append({
                    "masks": mask.cpu().numpy(),
                    "labels": pred_class,
                    "scores": pred_score
                })

        return results
    
    
    def instance_inference2(self, mask_cls, mask_pred):
    
        mask_cls,mask_pred = mask_cls[0],mask_pred[0]
        scores, labels  = F.softmax(mask_cls, dim=-1)[...,:-1].max(-1)
        keep = labels.ne(self.num_classes) & (scores > self.test_object_score)
        mask_pred = mask_pred[keep]
        labels_per_query = labels[keep]
        scores_per_query = scores[keep]

        result_pred_mask = (mask_pred > 0).float()
        heatmap = mask_pred.float().sigmoid()
        mask_scores_per_image = (heatmap * result_pred_mask).sum(1) / (result_pred_mask.sum(1) + 1e-6)
        scores_per_mask = scores_per_query * mask_scores_per_image
        labels_per_mask = labels_per_query

        results = []
        for score, label, mask in zip(scores_per_mask, labels_per_mask, result_pred_mask):
            results.append({
                    "masks": mask.cpu().numpy(),
                    "labels": label.item(),
                    "scores": score.item(),

            })
        return results


    
    def parse_losses(self, losses):
        loss = sum(v for v in losses.values())
        losses["loss"] = loss
        for loss_name, loss_value in losses.items():
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            losses[loss_name] = loss_value.item()
        return loss, losses



