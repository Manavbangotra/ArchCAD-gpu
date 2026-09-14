#!/usr/bin/env python3
"""VecFormerConfig: class counts stay with their config, partial dicts merge, round trip.

    python vecformer/checks/test_config.py
"""
import os.path as osp
import sys

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, osp.join(HERE, ".."))
import cpu_kernels  # noqa: E402

cpu_kernels.install()
from model.vecformer.configuration_vecformer import VecFormerConfig  # noqa: E402

FAILURES = []


def check(ok, what):
    if not ok:
        FAILURES.append(what)


def main():
    a = VecFormerConfig(num_semantic_classes=43, num_instance_classes=43, backbone_config=dict(drop_path=0.1),
                        text_config=dict(enabled=True))
    b = VecFormerConfig()                       # what the trainer used to build after the model
    for key, sub in (("cad_decoder_config", "num_semantic_classes"), ("instance_criterion_config", "num_instance_classes"),
                     ("semantic_criterion_config", "num_semantic_classes"), ("evaluator_config", "num_classes"),
                     ("metrics_computer_config", "num_classes")):
        check(getattr(a, key)[sub] == 43, f"{key}.{sub} kept at 43 after another config is built: {getattr(a, key)[sub]}")
        check(getattr(b, key)[sub] == 35, f"default {key}.{sub} is 35")
    check(a.backbone_config["drop_path"] == 0.1 and a.backbone_config["enc_depths"] == (2, 2, 2, 6, 2),
          "partial backbone_config merges with defaults")
    check(b.backbone_config["drop_path"] == 0.3, "defaults untouched by overrides")
    check(a.text_config["enabled"] and a.text_config["l0_weight"] == 1e-4, "partial text_config merges")
    c = VecFormerConfig.from_dict(a.to_dict())
    check(c.cad_decoder_config == a.cad_decoder_config and c.evaluator_config == a.evaluator_config
          and c.text_config["enabled"], "to_dict / from_dict round trip")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  " + f)
        return 1
    print("ok - config: counts stay put, partial dicts merge, round trip")
    return 0


if __name__ == "__main__":
    sys.exit(main())
