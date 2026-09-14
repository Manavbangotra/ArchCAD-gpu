# Vendored: VecFormer

This directory is the official code of

> Xingguang Wei, Haomin Wang, et al. *Point or Line? Using Line-based Representation for
> Panoptic Symbol Spotting in CAD Drawings.* NeurIPS 2025. arXiv:2505.23395.
> https://github.com/WesKwong/VecFormer

imported unmodified from upstream commit `cdb4795` (2025-10-24), Apache License 2.0 (see
`LICENSE`). The project figure (`assets/`) is not included.

It is the base of this repository's line-based panoptic symbol spotter (plan:
reproduce VecFormer, then add TextCAD-style text fusion and the takeoff product's
label space). Every local change is a separate commit on top of this import, so
`git log -- vecformer/` is the complete list of differences from upstream.

Imports inside are rooted at this directory (`from model.vecformer ...`), so run its
entry points with `PYTHONPATH=vecformer`.
