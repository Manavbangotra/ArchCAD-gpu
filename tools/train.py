import numpy as np
import torch
import torch.nn as nn
import yaml
from munch import Munch
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import argparse
import datetime
import os
import os.path as osp
import shutil
import time
# import pdb

from svgnet.data import build_dataloader, build_dataset
from svgnet.model.svgnet import SVGNet as svgnet
from svgnet.model.criterion import SetCriterion
from svgnet.model.matcher import HungarianMatcher
from svgnet.evaluation import PointWiseEval,InstanceEval
from svgnet.util import (
    get_device,
    AverageMeter,
    SummaryWriter,
    build_optimizer,
    checkpoint_save,
    cosine_lr_after_step,
    fixed_lr,
    custimized_lr,
    get_dist_info,
    get_max_memory,
    get_root_logger,
    init_dist,
    is_main_process,
    is_multiple,
    is_power2,
    load_checkpoint,
    set_seed,
    get_scheduler,
    build_new_optimizer
)



def get_args():
    parser = argparse.ArgumentParser("svgnet")
    parser.add_argument("config", type=str, help="path to config file")
    parser.add_argument("--dist", action="store_true", help="run with distributed parallel")
    parser.add_argument("--sync_bn", action="store_true", help="run with sync_bn")
    parser.add_argument("--resume", type=str, help="path to resume from")
    parser.add_argument("--work_dir", type=str, help="working directory")
    parser.add_argument("--skip_validate", action="store_true", help="skip validation")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--exp_name", type=str, default="default")
    
    args = parser.parse_args()
    return args


def train(epoch, model, optimizer, scheduler, scaler, train_loader, cfg, logger, writer):
    global best_metric
    skipped = {"n": 0}

    model.train()
    iter_time = AverageMeter(True)
    data_time = AverageMeter(True)
    meter_dict = {}
    end = time.time()
    # 
    lr = optimizer.param_groups[0]["lr"]  # 确保lr有初值

    if train_loader.sampler is not None and cfg.dist:
        train_loader.sampler.set_epoch(epoch)

    accum = max(1, int(getattr(cfg, "accumulate_steps", 1)))
    save_interval_iters = int(getattr(cfg, "save_interval_iters", 0) or 0)
    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(train_loader, start=1):
        data_time.update(time.time() - end)

        if scheduler is None:
            cosine_lr_after_step(optimizer, cfg.optimizer.lr, epoch - 1, cfg.step_epoch, cfg.epochs)
            # custimized_lr(optimizer, cfg.optimizer.lr, epoch - 1, cfg.step_epoch, cfg.epochs)

        with torch.cuda.amp.autocast(enabled=cfg.fp16):
            try:
                _, loss, log_vars = model(batch)

            except Exception as e:
                # Skipping a bad batch is fine; skipping most of them silently is
                # not. Count them and say so, so a systematic data fault shows up
                # as a number instead of scrolling past as noise.
                skipped["n"] += 1
                if skipped["n"] <= 5 or skipped["n"] % 100 == 0:
                    logger.warning(f"batch {i} skipped ({type(e).__name__}: {e}) "
                                   f"file={batch[-1]} -- {skipped['n']} so far this epoch")
                continue

            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            
        for k, v in log_vars.items():
            if k not in meter_dict.keys() and k != "placeholder":
                meter_dict[k] = AverageMeter()
            meter_dict[k].update(v)
        
        
        # backward
        #
        # Gradient accumulation: the paper trains at an effective batch of ~64
        # (8 GPUs x batch 8). On one GPU the per-step batch has to be 1-2, which
        # makes the Hungarian matching and the loss very noisy. Accumulating
        # over `accumulate_steps` micro-batches restores the effective batch
        # without needing the memory for it.
        if loss > 0:
            # Scale so accumulated gradients average rather than sum.
            scaler.scale(loss / accum).backward()

            if i % accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        if i == 1 and epoch == 1:
            # Report parameters that never receive a gradient — useful once,
            # but it would flood the log if printed every iteration.
            missing = [n for n, p in model.named_parameters() if p.grad is None]
            if missing:
                logger.info(f"{len(missing)} parameters received no gradient, "
                            f"e.g. {missing[:3]}")

        # Mid-epoch checkpoint. On a machine with scheduled power cuts, saving
        # only at epoch end can throw away an hour or more of work. This writes
        # latest.pth (atomically) every save_interval_iters steps. Resuming from
        # it replays the current epoch from its start, but keeps the weights and
        # optimizer state — far cheaper than losing the epoch entirely.
        if save_interval_iters and is_multiple(i, save_interval_iters) and is_main_process():
            checkpoint_save(epoch, model, optimizer, cfg.work_dir, cfg.save_freq, rolling=True,
                            best_metric=best_metric)
            logger.info(f"Rolling checkpoint saved at epoch {epoch} iter {i}")


        # time and print
        remain_iter = len(train_loader) * (cfg.epochs - epoch + 1) - i
        iter_time.update(time.time() - end)
        end = time.time()
        remain_time = remain_iter * iter_time.avg
        remain_time = str(datetime.timedelta(seconds=int(remain_time)))
        lr = optimizer.param_groups[0]["lr"]

        if is_multiple(i, 50):
            log_str = f"Epoch [{epoch}/{cfg.epochs}][{i}/{len(train_loader)}]  "
            log_str += (
                f"lr: {lr:.2g}, eta: {remain_time}, mem: {get_max_memory()}, "
                f"data_time: {data_time.val:.2f}, iter_time: {iter_time.val:.2f}"
            )
            for k, v in meter_dict.items():
                log_str += f", {k}: {v.val:.4f}"
            logger.info(log_str)
    # Flush a partial accumulation window left over at the end of the epoch.
    if len(train_loader) % accum != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    writer.add_scalar("train/learning_rate", lr, epoch)
    for k, v in meter_dict.items():
        writer.add_scalar(f"train/{k}", v.avg, epoch)
    if skipped["n"]:
        logger.warning(f"epoch {epoch}: {skipped['n']} of {len(train_loader)} batches skipped")
    checkpoint_save(epoch, model, optimizer, cfg.work_dir, cfg.save_freq,
                    best_metric=best_metric)


# train 改完不要忘记val

def validate(epoch, model, optimizer, val_loader, cfg, logger, writer):
    val_skipped = {"n": 0}
    logger.info("Validation")
    # dist.get_world_size() raises when training on a single GPU without
    # torchrun; get_dist_info() reports world_size 1 in that case.
    # ignore_label follows the background id, matching tools/inference.py
    # (the previous hardcoded 49 never matched, so background was scored).
    _, world_size = get_dist_info()
    sem_point_eval = PointWiseEval(num_classes=cfg.model.semantic_classes,
                                   ignore_label=cfg.model.semantic_classes,
                                   gpu_num=world_size)
    instance_eval = InstanceEval(num_classes=cfg.model.semantic_classes,
                                 ignore_label=cfg.model.semantic_classes,
                                 gpu_num=world_size)
    meter_dict = {}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(val_loader):

            try:
                with torch.cuda.amp.autocast(enabled=cfg.fp16):
                    res,loss, log_vars = model(batch)
                sem_preds = torch.argmax(res["semantic_scores"],dim=1).cpu().numpy()
                sem_gts = res["semantic_labels"].cpu().numpy()
                sem_point_eval.update(sem_preds, sem_gts)
                instance_eval.update(
                    res["instances"],
                    res["targets"],
                    res["lengths"],
                )
                # meter_dict
                for k, v in log_vars.items():
                    if k not in meter_dict.keys() and k != "placeholder":
                        meter_dict[k] = AverageMeter()
                    meter_dict[k].update(v)
            except Exception as e:
                val_skipped["n"] += 1
                if val_skipped["n"] <= 5:
                    logger.warning(f"validation batch skipped ({type(e).__name__}: {e}) "
                                   f"file={batch[-1]}")

    global best_metric

    if val_skipped["n"]:
        logger.warning(f"validation: {val_skipped['n']} of {len(val_loader)} batches skipped")
    logger.info("Evaluate semantic segmentation")
    miou,acc = sem_point_eval.get_eval(logger)
    logger.info("Evaluate panoptic segmentation")
    sPQ, sRQ, sSQ = instance_eval.get_eval(logger)
    for k, v in meter_dict.items():
        writer.add_scalar(f"val/{k}", v.avg, epoch)
        
    writer.add_scalar("val/mIoU", miou, epoch)
    writer.add_scalar("val/Acc", acc, epoch)
    writer.add_scalar("val/sPQ", sPQ, epoch)
    writer.add_scalar("val/sRQ", sRQ, epoch)
    writer.add_scalar("val/sSQ", sSQ, epoch)


    # Which number decides "best" and when to stop early. sPQ was hardcoded, but
    # panoptic quality can sit at exactly 0 for many epochs on an imbalanced
    # dataset, and `best_metric < sPQ` is then never true — so best.pth was never
    # written at all. mIoU moves from the first epoch, so it is the default.
    which = str(getattr(cfg, "early_stop", {}).get("metric", "miou")).lower()
    score = sPQ if which == "pq" else miou

    if score > best_metric:
        best_metric = score
        checkpoint_save(epoch, model, optimizer, cfg.work_dir, cfg.save_freq, best=True,
                        best_metric=best_metric)
        logger.info(f"New best {which} {best_metric:.3f} at epoch {epoch}")

    return score




def main():
    args = get_args()

    cfg_txt = open(args.config, "r").read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))

    # Seed unconditionally. This used to sit inside the `if args.dist` branch, so
    # single-GPU runs were unseeded and started from different random weights
    # every time — two identical runs differed by 35 IoU points here. That makes
    # any A/B comparison (e.g. with vs without pretraining) unreadable.
    rank = 0
    if args.dist:
        rank = init_dist()
    set_seed(args.seed + rank)
    cfg.dist = args.dist
    
    # work_dir & logger
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        dataset_name = cfg.data.train.type
        cfg.work_dir = osp.join("./work_dirs", dataset_name, osp.splitext(osp.basename(args.config))[0], args.exp_name)

    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)

    logger.info(f"Config:\n{cfg_txt}")
    logger.info(f"Distributed: {args.dist}")
    logger.info(f"Mix precision training: {cfg.fp16}")
    shutil.copy(args.config, osp.join(cfg.work_dir, osp.basename(args.config)))
    writer = SummaryWriter(cfg.work_dir)

    logger.info(f"Save at: {cfg.work_dir}")

    # criterion
    matcher = HungarianMatcher(**cfg.matcher)
    weight_dict = {
            "loss_ce": cfg.matcher.cost_class, 
            "loss_mask": cfg.matcher.cost_mask, 
            "loss_dice": cfg.matcher.cost_dice,
            }
    criterion = SetCriterion(matcher,weight_dict,cfg.criterion).to(get_device())
    
    model = svgnet(cfg.model, criterion=criterion).to(get_device())

    if args.sync_bn:
            nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #logger.info(model)
    
    total_params = 0
    trainable_params = 0
    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
    
    logger.info('Total Number of Parameters: {} M'.format(str(float(total_params) / 1e6)[:5]))
    logger.info('Total Trainable Number of Parameters: {} M'.format(str(float(trainable_params) / 1e6)[:5]))
        

    if args.dist:
        model = DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()], find_unused_parameters=(trainable_params < total_params)
        )
    
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)

    # data
    train_set = build_dataset(cfg.data.train, logger)
    val_set = build_dataset(cfg.data.test, logger)

    train_loader = build_dataloader(args,train_set, training=True, dist=args.dist, **cfg.dataloader.train)
    val_loader = build_dataloader(args,val_set, training=False, dist=args.dist, **cfg.dataloader.test)#args.dist

    # optim
    default_lr = cfg.optimizer.lr  # default for batch 16
    _, world_size = get_dist_info()
    # Gradient accumulation raises the *effective* batch, so it belongs in this
    # scaling too. Without it, accumulating to recover the paper's batch size
    # would still divide the learning rate as though the batch were tiny.
    accum_steps = max(1, int(getattr(cfg, "accumulate_steps", 1)))
    total_batch_size = cfg.dataloader.train.batch_size * world_size * accum_steps
    scaled_lr = default_lr * (total_batch_size / 16)
    cfg.optimizer.lr = scaled_lr
    logger.info(
        f"Scale LR from {default_lr} (batch size 16) to {scaled_lr} "
        f"(effective batch {total_batch_size} = {cfg.dataloader.train.batch_size} "
        f"x {world_size} gpu x {accum_steps} accum)"
    )
    optimizer = build_new_optimizer(model, cfg.optimizer)
    # scheduler
    #scheduler = get_scheduler(cfg.scheduler,optimizer)
    scheduler = None
    
    # pretrain, resume
    start_epoch = 1
    if args.resume:
        logger.info(f"Resume from {args.resume}")
        start_epoch = load_checkpoint(args.resume, logger, model, optimizer=optimizer)
    elif cfg.pretrain:
        logger.info(f"Load pretrain from {cfg.pretrain}")
        load_checkpoint(cfg.pretrain, logger, model)

    global best_metric
    best_metric = 0
    if args.resume:
        # Without this, a resumed run compares against 0 and the first validated
        # epoch overwrites best.pth even when it is worse than what came before.
        try:
            _ck = torch.load(args.resume, map_location="cpu", weights_only=False)
            best_metric = float(_ck.get("best_metric") or 0)
            logger.info(f"Restored best_metric {best_metric:.3f} from {args.resume}")
        except Exception as exc:
            logger.warning(f"Could not read best_metric from {args.resume}: {exc}")

    # if is_main_process():
    #     validate(0, model, optimizer, val_loader, cfg, logger, writer)

    # train and val
    logger.info("Training")
    # Early stopping: end the run once the monitored score has stopped improving,
    # rather than always spending every configured epoch.
    es = getattr(cfg, "early_stop", None) or {}
    patience = int(es.get("patience", 0) or 0)
    min_delta = float(es.get("min_delta", 0.0) or 0.0)
    metric_name = str(es.get("metric", "miou")).lower()
    stale = 0
    prev_best = best_metric

    for epoch in range(start_epoch, cfg.epochs + 1):
        train(epoch, model, optimizer, scheduler, scaler, train_loader, cfg, logger, writer)
        if scheduler is not None:scheduler.step()
        score = validate(epoch, model, optimizer, val_loader, cfg, logger, writer)
        writer.flush()

        if patience > 0 and score is not None:
            if score > prev_best + min_delta:
                prev_best = score
                stale = 0
            else:
                stale += 1
                logger.info(
                    f"No {metric_name} gain over {prev_best:.3f} for {stale}/{patience} "
                    f"epoch(s) (this epoch {score:.3f})"
                )
                if stale >= patience:
                    logger.info(
                        f"Early stop at epoch {epoch}: {metric_name} has not improved "
                        f"by >{min_delta} for {patience} epochs. Best {prev_best:.3f}."
                    )
                    break

    logger.info(f"Finish!!! Model at: {cfg.work_dir}")


if __name__ == "__main__":
    main()
 

 