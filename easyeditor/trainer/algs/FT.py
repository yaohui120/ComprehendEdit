import copy
import logging
from collections import defaultdict

import higher
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from collections import deque
from higher.patch import (
    _MonkeyPatchBase,
    _torch,
    _typing,
    _utils,
    buffer_sync,
    make_functional,
)
from .patch import monkeypatch as _make_functional

from . import local_nn
from .editable_model import EditableModel
from .hooks import hook_model
from ..utils import _inner_params, _logits
import time
import math
import numpy as np
from copy import deepcopy
from ..utils import safe_backward
import pdb

LOG = logging.getLogger(__name__)

def _KD_loss(pred, soft, T=1):
    pred = torch.log_softmax(pred/T, dim=1)
    soft = torch.softmax(soft/T, dim=1)
    # return torch.tensor(0)
    return -1 * torch.mul(soft, pred).sum()/pred.shape[0]

def parent_module(model, pname):
    components = pname.split('.')
    parent = model

    for component in components[:-1]:
        if hasattr(parent, component):
            parent = getattr(parent, component)
        elif component.isdigit():
            parent = parent[int(component)]
        else:
            raise RuntimeError(f"Couldn't find child module {component}")

    if not hasattr(parent, components[-1]):
        raise RuntimeError(f"Couldn't find child module {components[-1]}")

    return parent

def brackets_to_periods(name):
    return name.replace("[", ".").replace("]", "")

def calc_dists(v1, v2):
    v1 = v1 / torch.norm(v1, p=2, dim=-1).unsqueeze(-1)
    v2 = v2 / torch.norm(v2, p=2, dim=-1).unsqueeze(-1)
    return torch.norm(v2-v1, p=2, dim=-1)


from tqdm import tqdm, trange
class FT(EditableModel):
    def __init__(self, model, config, model_constructor):
        super().__init__(model, config, model_constructor)

        if not str(self.config.device).startswith('cuda'):
            self.config.device = f'cuda:{self.config.device}'
        self.model = self.model.to(torch.float32)
        self.opt = None
        self.tokenizer = self.model.opt_tokenizer if self.config.model_name == 'blip2' else self.model.llama_tokenizer
        
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(
            prefix=prefix, keep_vars=keep_vars
        )  # Get default state dict
        state_dict["model_config"] = self.model.config  # Include model config
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        config = state_dict["model_config"]
        del state_dict["model_config"]
        if config != self.model.config:
            LOG.info("Loaded model config doesn't match current model config.")
            LOG.info(f"Loaded: {config}")
            LOG.info(f"Current: {self.model.config}")

        res = super().load_state_dict(state_dict, True)
        assert len(res.unexpected_keys) == 0, "Shouldn't have any unexpected keys"
        return res

    def forward(self, *inputs, **kwargs):
        if 'minigpt4' in self.config.model_name.lower() or 'blip' in self.config.model_name.lower() or 'llava' in self.config.model_name.lower():
            outputs = self.model(*inputs, **kwargs)
        else:
            raise not NotImplementedError("Model not supported")
        return outputs
    
    def outer_parameters(self):
        # return None
        all = []
        for nn, params in _inner_params(self.model.named_parameters(), self.config.inner_params):
            all.append(params)
        return all

    def recover_ori_model(self):
        for n, p in self.model.named_parameters():
            if n in self.save_weight.keys():
                p.data = self.save_weight[n].clone()

    def edit(self, batch, condition=None, detach_history=False, return_factors=False):
        self.model.train()
        if self.config.inner_params[0] in ['Qformer', 'mm_projector']:

            weights = {
                n: p
                for n, p in self.model.named_parameters()
                if n.find(self.config.inner_params[0]) != -1
            }
        else:
            names = set([n for n, p in self.model.named_parameters()])
            pset = set(self.config.inner_params)
            # for p in pset:
            #     assert p in names, f"inner param {p} not in model"

            weights = {
                n: p
                for n, p in self.model.named_parameters()
                if n in pset
            }
        
        # Save old weights for future restoration
        self.save_weight = {k: v.detach().clone() for k, v in weights.items()}

        self.opt = torch.optim.AdamW(
            [v for _, v in weights.items()],
            lr=self.config.edit_lr
        )
        
        for name, w in self.model.named_parameters():
            w.requires_grad = name in pset

        if 'minigpt4' in self.config.model_name.lower() or 'blip' in self.config.model_name.lower() or 'llava' in self.config.model_name.lower():
            for it in range(self.config.num_steps):
                ori_outputs = self.model(batch)
                outputs = ori_outputs.logits
                labels = ori_outputs.labels # ori not have
                loss_infos = self.edit_loss_fn(self.config, outputs, labels, multimodal=True)
                loss = loss_infos['nll']

                if loss_infos['acc'] == 1:
                    break
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        else:
            raise not NotImplementedError("Model not supported")

        edited_model = self.model
        return (
            FT(
                edited_model,
                self.config,
                self.model_constructor,
            ),
            {}
        )
