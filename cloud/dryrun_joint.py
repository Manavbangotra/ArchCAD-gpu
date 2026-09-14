#!/usr/bin/env python3
"""Dry run of the product configuration on real data, before paying for a joint run.

Builds the multisource data (configs/data/product_joint.yaml) and the Arch-43 text
model (configs/model/product_arch43.yaml, init_checkpoint ignored), takes a few real
samples from every source whose directory exists, and runs a training step and an
evaluation pass per source. Reports label ranges, text counts and segment counts,
and fails on non-finite losses or evaluation errors.

    cd vecformer && PYTHONPATH=. python ../cloud/dryrun_joint.py            # GPU, real kernels
    cd vecformer && PYTHONPATH=. python ../cloud/dryrun_joint.py --cpu      # laptop: small model, stand-ins
"""
import argparse
import os
import os.path as osp
import sys

import torch
import yaml

HERE = osp.dirname(osp.abspath(__file__))
VEC = osp.join(HERE, "..", "vecformer")
sys.path.insert(0, VEC)
sys.path.insert(0, osp.join(VEC, "checks"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true", help="pure-torch kernels and a small model")
    ap.add_argument("--data", default=osp.join(VEC, "configs/data/product_joint.yaml"))
    ap.add_argument("--model", default=osp.join(VEC, "configs/model/product_arch43.yaml"))
    ap.add_argument("--per_source", type=int, default=2)
    ap.add_argument("--max_segments", type=int, default=20000, help="skip larger samples in --cpu mode")
    a = ap.parse_args()

    if a.cpu:
        import cpu_kernels
        cpu_kernels.install()
    os.chdir(VEC)                       # config paths are relative to vecformer/
    from data.multisource import build_sources
    from data.floorplancad.floorplancad import FloorPlanCAD
    from model.vecformer.configuration_vecformer import VecFormerConfig
    from model.vecformer.modeling_vecformer import VecFormer

    data_args = yaml.safe_load(open(a.data))["dataset_args"]
    data_args["sources"] = [s for s in data_args["sources"] if osp.isdir(osp.join(s["root_dir"], "train"))]
    print("sources present:", [s["name"] for s in data_args["sources"]])
    if not data_args["sources"]:
        print("no source directories found")
        return 1
    train, val, _ = build_sources(data_args)

    model_args = dict(yaml.safe_load(open(a.model))["model_args"])
    model_args.pop("init_checkpoint", None)
    if a.cpu:
        from test_model_text_cpu import small_config
        small = small_config(True)
        model_args.update(backbone_config=small.backbone_config, cad_decoder_config=small.cad_decoder_config)
    device = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    model = VecFormer(VecFormerConfig(**model_args)).to(device)
    C = model.num_semantic_classes

    failures = 0
    for name, ds in zip(train.names, train.datasets):
        items = []
        for i in range(len(ds)):
            item = ds[i]
            if a.cpu and item.coords.shape[0] > a.max_segments:
                continue
            items.append(item)
            if len(items) == a.per_source:
                break
        if not items:
            print(f"[{name}] no sample under {a.max_segments} segments")
            continue
        sem = torch.cat([it.sem_ids for it in items])
        n_text = [0 if it.text is None else len(it.text["text_types"]) for it in items]
        print(f"[{name}] segments {[it.coords.shape[0] for it in items]} primitives {[len(it.sem_ids) for it in items]} "
              f"texts {n_text} labels {sorted(set(sem.tolist()))}")
        bad = sorted(set(l for l in sem.tolist() if not (0 <= l <= C or model.label_space.is_instance_target(l)
                                                         or l == 55)))
        if bad:
            print(f"[{name}] FAIL labels outside Arch-43: {bad}")
            failures += 1
        batch = FloorPlanCAD.collate_fn(items)
        batch.pop("data_paths")
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        model.train()
        out = model(**batch)
        ok = bool(torch.isfinite(out.loss))
        out.loss.backward()
        model.zero_grad()
        print(f"[{name}] train loss {out.loss.item():.3f} "
              f"sem {out.dict_sublosses.get('semantic_loss', torch.tensor(0)).item():.3f} "
              f"inst {out.dict_sublosses.get('instance_loss', torch.tensor(0)).item():.3f} "
              f"text_l0 {out.dict_sublosses.get('text_l0', torch.tensor(0)).item():.2f}")
        failures += not ok
        v = val[name]
        vb = FloorPlanCAD.collate_fn([v[i] for i in range(min(len(v), 1))])
        vb.pop("data_paths")
        vb = {k: t.to(device) if torch.is_tensor(t) else t for k, t in vb.items()}
        model.eval()
        with torch.no_grad():
            ev = model(**vb)
        print(f"[{name}] eval ok, strict tp+fp+fn {int(sum(ev.metric_states[k].sum() for k in ('strict_tp_per_class', 'strict_fp_per_class', 'strict_fn_per_class')))}")
    print("FAILED" if failures else "ok - joint dry run")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
