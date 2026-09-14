from data import register_dataset, DatasetSplits
from .floorplancad import FloorPlanCAD

@register_dataset("floorplancad")
def build(dataset_args: dict):
    train_dataset = FloorPlanCAD(**dataset_args, split="train")
    val_dataset = FloorPlanCAD(**dataset_args, split="val")
    test_dataset = FloorPlanCAD(**dataset_args, split="test")
    data_collator = FloorPlanCAD.collate_fn
    return DatasetSplits(train=train_dataset,
                         val=val_dataset,
                         test=test_dataset), data_collator
