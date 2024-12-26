from dataclasses import dataclass
from typing import Optional, Any, List
from ...util.hparams import HyperParams
import yaml


@dataclass

class GraceMultimodalHparams(HyperParams):
    # Experiments
    
    edit_lr: float
    n_iter: int
    # Method
    eps: float
    dist_fn: str
    val_init: str
    val_train: str
    val_reg: str
    reg: str
    replacement: str
    eps_expand: str
    num_pert: str
    dropout: float

    # Module templates
    name: str
    model_name: str
    model_class: str
    tokenizer_class: str
    tokenizer_name: str
    inner_params: List[str]
    device: int
    alg_name: str
    train_base: bool
    val_batch_size: int
    accumulate_bs: int
    eval_only: bool
    opt: str
    lr: float
    debug: bool
    save: bool
    model_save_pt: int
    silent: bool
    
    cedit: int
    iedit: int
    cloc: int
    cbase: int
    grad_clip: float
    
    log_interval: int
    eval_log_interval: int
    final_eval: bool
    val_interval: int #5000
    early_stop_patience: int
    early_stop_key: str
    
    archive: Any
    no_grad_layers: Any
    results_dir: str
    qformer_checkpoint: str
    qformer_name_or_path: str
    state_dict_file: str
    gpu_used_id: List[int] #
    gpu_split: List[str] #
    
    
    # Method
    alg: str
    seed: int
    
    # Image_dir
    coco_image: str
    rephrase_image: str

    # Defaults
    batch_size: int = 128
    max_length: int = 30
    model_parallel: bool = False
    
    exact_match: bool = False
    freeze_qformer: bool = True
    pretrained_ckpt: Optional[str] = None
    max_epochs: Optional[int] = None
    max_iters: Optional[int] = None
    task: Optional[str] = None # 
    num_tasks: Optional[int] = -1

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'GRACE') or print(
            f'GraceMultimodalHparams can not load from {hparams_name_or_path}, '
            f'alg_name is {config["alg_name"]} ')
        return cls(**config)