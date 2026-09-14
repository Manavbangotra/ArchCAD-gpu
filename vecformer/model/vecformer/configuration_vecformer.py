"""
VecFormer configuration
"""
import copy
import inspect

from transformers import PretrainedConfig


def _merged(name, value):
    """A fresh copy of the signature default for dict argument `name`, updated with
    `value`. Fresh, because __init__ fills class counts into these dicts: mutating
    the shared defaults made every later VecFormerConfig() (the trainer builds one)
    rewrite the counts inside an existing model's config. Merged, so a YAML may
    override single keys (e.g. backbone enc_channels) without restating the rest."""
    default = inspect.signature(VecFormerConfig.__init__).parameters[name].default
    out = copy.deepcopy(default)
    out.update(copy.deepcopy(value) if value else {})
    return out


class VecFormerConfig(PretrainedConfig):
    model_type='vecformer'

    def __init__(
        self,
        num_instance_classes: int = 35, # number of instance classes of dataset
        num_semantic_classes: int = 35, # number of semantic classes of dataset
        thing_class_idxs: list[int] = [i for i in range(30)], # thing class idxs
        stuff_class_idxs: list[int] = [30,31,32,33,34], # stuff class idxs
        use_layer_fusion: bool = True, # (`bool`): whether to use layer fusion enhancement
        # Joint-training label space (criterion/label_space.py): None = upstream losses;
        # "arch43" = Arch-43 with coarse ids (door-any...) scored as marginals and
        # per-source annotated classes. `sources` fixes the meaning of a batch's
        # source_ids (dataset_args source_id); empty = taxonomy.ANNOTATED order.
        label_space: str = None,
        sources: list = [],
        query_thr = 0.5, # (`float`): query threshold, used only in training
        max_num_queries = -1, # (`int`): max number of queries, used only in training
        # VecFormer Backbone
        sample_mode = "line", # (`str`): sample mode, "line" or "point"
        backbone_config: dict = dict(
            in_channels=7, # point mode: 4, line mode: 7
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(32, 64, 128, 256, 512),
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=(1024, 1024, 1024, 1024, 1024),
            dec_depths=(2, 2, 2, 2),
            dec_channels=(64, 64, 128, 256),
            dec_num_head=(4, 4, 8, 16),
            dec_patch_size=(1024, 1024, 1024, 1024),
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.3,
            pre_norm=True,
            shuffle_orders=True,
            enable_rpe=False,
            enable_flash=True,
            upcast_attention=False,
            upcast_softmax=False,
            cls_mode=False,
            pdnorm_bn=False,
            pdnorm_ln=False,
            pdnorm_decouple=True,
            pdnorm_adaptive=False,
            pdnorm_affine=True,
        ),
        # CAD Decoder
        cad_decoder_config: dict = dict(
            num_instance_classes = None, # (`int`): number of instance classes
            num_semantic_classes = None, # (`int`): number of semantic classes
            input_dim = 64, # (`int`): input dimension of CAD decoder
            embed_dim = 256, # (`int`): embedding dimension of CAD decoder
            activation = "GELU", # (`str`): activation function
            dropout = 0.1, # (`float`): dropout rate
            n_heads = 8, # (`int`): number of attention heads
            n_blocks = 6, # (`int`): number of blocks in CAD decoder
            attn_drop = 0.1, # (`float`): attention drop rate
            objectiveness_flag = False, # (`bool`): flag to indicate if CAD decoder should predict objectiveness
            iter_pred = True, # (`bool`): whether to use every cad decoder block to predict
            only_last_block_sem = True, # (`bool`): only use the last block to predict semantic
            use_attn_mask = False, # (`bool`): whether to use attention mask in CAD decoder
            # use_layer_fusion = True, # (`bool`): whether to use layer fusion enhancement in CAD decoder
        ),
        # Criterion
        instance_criterion_config: dict = {
            "num_instance_classes": None,
            "class_loss_weight": 2.5,
            "ce_non_object_weight": 0.01,
            "bce_loss_weight": 5.0,
            "dice_loss_weight": 5.0,
            "score_loss_weight": 0.5,
            "topk_matches": 1,
            "iter_matcher": True,
            "label_smoothing": 0.1,
            "use_mean_batch_loss": True,
        }, # instance loss config
        semantic_criterion_config: dict = {
            "num_semantic_classes": None,
            "ce_loss_weight": 5.0,
            "ce_unlabeled_weight": 0.1,
            "label_smoothing": 0.1,
            "use_mean_batch_loss": True,
        }, # semantic loss config
        # Text (TextCAD TACE + MSF, model/vecformer/text). Off by default, so the
        # released VecFormer configs and checkpoints are unchanged.
        text_config: dict = dict(
            enabled = False, # (`bool`): fuse text annotations (needs dataset_args use_text: true)
            num_types = 46, # (`int`): len(dataset/text_types.py TYPE_NAMES)
            num_grades = 7, # (`int`): len(dataset/text_types.py GRADES)
            dim = 32, # (`int`): TACE width D
            heads = 4, # (`int`): TACE and MSF heads H
            levels = ("enc0", "enc1", "enc2", "enc3", "enc4"), # backbone stages with MSF (TextCAD: 5 levels)
            knn = 16, # (`int`): nearest annotations per line in MSF cross-attention, 0 = all
            l0_weight = 1e-4, # (`float`): lambda_c on the expected number of open gates
        ),
        # Layer-name embedding (data/floorplancad/layer_names.py): hashed words of each
        # CAD layer's name, averaged, projected and added to the line features after the
        # backbone embedding. Zero-initialised, so it starts as the model without it.
        layer_name_config: dict = dict(
            enabled = False, # (`bool`): needs dataset_args use_layer_names: true
            vocab = 4096, # (`int`): hashed vocabulary size, token 0 = padding
            dim = 32, # (`int`): embedding width
        ),
        num_topk_preds: int = 600, # number of topk predictions
        use_obj_normalization: bool = True, # whether to use object normalization
        obj_normalization_thr: float = 0.01, # object normalization threshold
        use_vector_nms: bool = True, # whether to use vector nms
        vector_nms_kernel: str = "linear", # vector nms kernel
        pred_score_thr: float = 0.5, # predicted score threshold, used in pred_score > pred_score_thr to filter out low-confidence predictions
        mask_logit_thr: float = 0.3, # mask logit threshold, used in pred_masks_sigmoid > mask_logit_thr to get 0/1 mask
        n_primitives_thr: int = 1, # number of primitives threshold
        whether_output_instance: bool = False, # whether to output instance predictions
        # Evaluator
        evaluator_config: dict = {
            "num_classes": None,
            "iou_threshold": 0.5,
            "ignore_label": None,
            "output_dir": "instance_preds/"
        }, # evaluator config
        # MetricsComputer
        metrics_computer_config: dict = {
            "num_classes": None,
            "thing_class_idxs": None,
            "stuff_class_idxs": None
        }, # metrics computer config
        **kwargs
    ):
        super().__init__(**kwargs)
        backbone_config = _merged("backbone_config", backbone_config)
        cad_decoder_config = _merged("cad_decoder_config", cad_decoder_config)
        instance_criterion_config = _merged("instance_criterion_config", instance_criterion_config)
        semantic_criterion_config = _merged("semantic_criterion_config", semantic_criterion_config)
        text_config = _merged("text_config", text_config)
        layer_name_config = _merged("layer_name_config", layer_name_config)
        evaluator_config = _merged("evaluator_config", evaluator_config)
        metrics_computer_config = _merged("metrics_computer_config", metrics_computer_config)

        self.num_instance_classes: int = num_instance_classes
        self.num_semantic_classes: int = num_semantic_classes
        self.thing_class_idxs: list[int] = thing_class_idxs
        self.stuff_class_idxs: list[int] = stuff_class_idxs
        self.use_layer_fusion: bool = use_layer_fusion
        self.label_space = label_space
        self.sources = list(sources)
        self.query_thr: float = query_thr
        self.max_num_queries: int = max_num_queries
        self.sample_mode: str = sample_mode
        # VecFormer Backbone
        self.backbone_config: dict = backbone_config
        # CAD Decoder
        cad_decoder_config["num_instance_classes"] = num_instance_classes
        cad_decoder_config["num_semantic_classes"] = num_semantic_classes
        self.cad_decoder_config: dict = cad_decoder_config
        # Criterion
        instance_criterion_config["num_instance_classes"] = num_instance_classes
        self.instance_criterion_config: dict = instance_criterion_config
        semantic_criterion_config["num_semantic_classes"] = num_semantic_classes
        self.semantic_criterion_config: dict = semantic_criterion_config
        self.text_config: dict = text_config
        self.layer_name_config: dict = layer_name_config
        # Predict
        self.num_topk_preds: int = num_topk_preds
        self.use_obj_normalization: bool = use_obj_normalization
        self.obj_normalization_thr: float = obj_normalization_thr
        self.use_vector_nms: bool = use_vector_nms
        self.vector_nms_kernel: str = vector_nms_kernel
        self.pred_score_thr: float = pred_score_thr
        self.mask_logit_thr: float = mask_logit_thr
        self.n_primitives_thr: int = n_primitives_thr
        self.whether_output_instance = whether_output_instance
        # Evaluator
        evaluator_config["num_classes"] = num_semantic_classes
        evaluator_config["ignore_label"] = num_semantic_classes
        self.evaluator_config: dict = evaluator_config
        # MetricsComputer
        metrics_computer_config["num_classes"] = num_semantic_classes
        metrics_computer_config["thing_class_idxs"] = list(thing_class_idxs)
        metrics_computer_config["stuff_class_idxs"] = list(stuff_class_idxs)
        self.metrics_computer_config: dict = metrics_computer_config