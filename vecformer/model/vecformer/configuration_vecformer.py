"""
VecFormer configuration
"""
from transformers import PretrainedConfig


class VecFormerConfig(PretrainedConfig):
    model_type='vecformer'

    def __init__(
        self,
        num_instance_classes: int = 35, # number of instance classes of dataset
        num_semantic_classes: int = 35, # number of semantic classes of dataset
        thing_class_idxs: list[int] = [i for i in range(30)], # thing class idxs
        stuff_class_idxs: list[int] = [30,31,32,33,34], # stuff class idxs
        use_layer_fusion: bool = True, # (`bool`): whether to use layer fusion enhancement
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

        self.num_instance_classes: int = num_instance_classes
        self.num_semantic_classes: int = num_semantic_classes
        self.thing_class_idxs: list[int] = thing_class_idxs
        self.stuff_class_idxs: list[int] = stuff_class_idxs
        self.use_layer_fusion: bool = use_layer_fusion
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
        metrics_computer_config["thing_class_idxs"] = thing_class_idxs
        metrics_computer_config["stuff_class_idxs"] = stuff_class_idxs
        self.metrics_computer_config: dict = metrics_computer_config