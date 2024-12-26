from .BaseTrainer import *
import json
import logging
import os
import shutil
import tempfile
import time

import torch
from .losses import kl_loc_loss
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from .utils import (
    EarlyStopper,
    RunningStatAverager,
    _logits,
    formatted_timestamp,
    safe_backward,
    time_delta_seconds,
)

LOG = logging.getLogger(__name__)


class MultimodalTrainer(BaseTrainer):
    def __init__(self, config, train_set: Dataset, val_set: Dataset):
        super().__init__(config, train_set, val_set)

        if hasattr(self.model, "edit_lrs") and not self.config.eval_only:
            self.lr_opt = self.OptimizerClass([self.model.edit_lrs], config.lr_lr)
            if self.archive is not None:
                self.lr_opt.load_state_dict(self.archive["lr_opt"])
        else:
            self.lr_opt = None

        if hasattr(self.config, "ft"):
            if getattr(self.config.ft, "use_locality", False):
                batch = next(self.edit_gen)
                self.model.loc_ids = batch["loc"]["input_ids"]
                self.model.loc_masks = batch["loc"]["attention_mask"]

    def edit_step_ori(self, batch, training: bool):
        self.model.train(training)
        self.original_model.train(training)

        with torch.no_grad():
            base_outputs = self.model(batch["loc"])
            if not isinstance(base_outputs, torch.Tensor):
                base_logits = base_outputs.logits
            else:  
                base_logits = base_outputs
                
            base_image_outputs = self.model(batch["loc_image"])
            if not isinstance(base_image_outputs, torch.Tensor):
                base_image_logits = base_image_outputs.logits
            else:
                base_image_logits = base_image_outputs
        
        # Do the edit

        start = time.time()
        edited_model, model_info = self.model.edit(batch["edit_inner"], batch["cond"])
        edit_time = time.time() - start

        with torch.set_grad_enabled(training):
            # Editing loss
            post_edit_outputs = edited_model(batch["edit_outer"])
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
                post_batch_labels = post_edit_outputs.labels
            else:
                post_edit_logits = post_edit_outputs
                post_batch_labels = batch["edit_outer"]["labels"]

            # rephrase image
            post_image_edit_outputs = edited_model(batch["edit_outer_image"])
            
            if not isinstance(post_image_edit_outputs, torch.Tensor):
                post_image_edit_logits = post_image_edit_outputs.logits
                post_image_batch_labels = post_image_edit_outputs.labels
            else:
                post_image_edit_logits = post_image_edit_outputs
                post_image_batch_labels = batch["edit_outer_image"]["labels"]
                
            inner_edit_outputs = edited_model(batch["edit_inner"])
            
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
                inner_batch_labels = inner_edit_outputs.labels
            else:
                inner_edit_logits = inner_edit_outputs
                inner_batch_labels = batch["edit_inner"]["labels"]

            l_edit = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels, multimodal=True)["nll"]
            l_image_edit = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels, multimodal=True)["nll"]          
            
            # Collect some useful metrics
            with torch.no_grad():
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels, multimodal=True)
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels, multimodal=True)
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels, multimodal=True)
            
            post_base_outputs = edited_model(batch["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
                kl_mask = post_base_outputs.attention_mask
            else:
                post_base_logits = post_base_outputs
                kl_mask = torch.ones(post_base_logits.shape[0], post_base_logits.shape[1]).to(post_base_logits.device)

            post_image_base_outputs = edited_model(batch["loc_image"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
                kl_image_mask = post_image_base_outputs.attention_mask
            else:
                post_image_base_logits = post_image_base_outputs
                kl_image_mask = torch.ones(post_image_base_logits.shape[0], post_image_base_logits.shape[1]).to(base_image_logits.device)

            l_loc = kl_loc_loss(base_logits.detach(), post_base_logits, mask=kl_mask)
            l_image_loc = kl_loc_loss(base_image_logits.detach(), post_image_base_logits, mask=kl_image_mask)

        # if l_edit.isnan():
        #     print("l_edit is nan")
        #     print("input: ", batch["edit_outer"]['text_input'])
        # elif l_image_edit.isnan():
        #     print("l_image_edit is nan")
        #     print("input: ", batch["edit_outer_image"]['text_input'])
        # elif l_loc.isnan():
        #     print("l_loc is nan")
        #     print("input: ", batch["loc"]['text_input'])
        # elif l_image_loc.isnan():
        #     print("l_image_loc is nan")
        #     print("input: ", batch["loc_image"]['text_input'])

        if self.config.alg == "SERAC_MULTI":
            l_total_edit = self.config.cedit * l_edit + self.config.cloc * l_loc + self.config.iedit * l_image_edit
        else:
            l_total_edit = self.config.cedit * l_edit + self.config.cloc * (l_loc + l_image_loc) + self.config.iedit * l_image_edit
        

        if training and self.config.alg != 'ft':
            safe_backward(l_total_edit, self.model.outer_parameters(), self.config.accumulate_bs, allow_unused=True)

        # Text locality
        post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
        base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=1, dim=-1).indices

        # Image locality
        post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
        base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits, dim=-1), k=10, dim=-1).indices

        info_dict = {}
        info_dict['loss/edit'] = l_edit.item()
        info_dict['loss/image_edit'] = l_image_edit.item()
        info_dict['loss/loc'] = l_loc.item()
        info_dict['edit/acc'] = post_edit_dict["acc"].item()
        info_dict['edit/log_prob'] = post_edit_dict["log_prob"].item()
        info_dict['edit/prob'] = post_edit_dict["prob"].item()
        info_dict['inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        info_dict["time/edit"] = edit_time
        info_dict["loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        info_dict["image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
        l_base = torch.tensor(0.0)
        l_total = l_total_edit + self.config.cbase * l_base

        info_dict["loss/total"] = l_total.item()
        info_dict["loss/total_edit"] = l_total_edit.item()
        info_dict["memory/alloc_max"] = torch.cuda.max_memory_allocated()
        info_dict["memory/res_max"] = torch.cuda.max_memory_reserved()
        info_dict = {**info_dict, **model_info}

        return l_total, l_edit, l_loc, l_base, info_dict

    def edit_step(self, batch, training: bool):
        self.model.train(training)
        self.original_model.train(training)

        with torch.no_grad():
            base_outputs = self.model(batch["loc"])
            if not isinstance(base_outputs, torch.Tensor):
                base_logits = base_outputs.logits
            else:  
                base_logits = base_outputs
                
            base_image_outputs = self.model(batch["loc_image"])
            if not isinstance(base_image_outputs, torch.Tensor):
                base_image_logits = base_image_outputs.logits
            else:
                base_image_logits = base_image_outputs
        
        # Do the edit
        start = time.time()
        edited_model, model_info = self.model.edit(batch["edit_inner"], batch["cond"])
        # edited_model, model_info = self.model, {}
        edit_time = time.time() - start

        with torch.set_grad_enabled(training):
            # Editing loss
            post_edit_outputs = edited_model(batch["edit_outer"])
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
                post_batch_labels = post_edit_outputs.labels
            else:
                post_edit_logits = post_edit_outputs
                post_batch_labels = batch["edit_outer"]["labels"]

            inner_edit_outputs = edited_model(batch["edit_inner"])
            
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
                inner_batch_labels = inner_edit_outputs.labels
            else:
                inner_edit_logits = inner_edit_outputs
                inner_batch_labels = batch["edit_inner"]["labels"]
            
            if True: # self.config.alg.lower() in ['hice', 'ft', 'ke']:
                post_image_edit_logits = inner_edit_logits
                post_image_batch_labels = inner_batch_labels
            else:
                # rephrase image
                post_image_edit_outputs = edited_model(batch["edit_outer_image"])
                if not isinstance(post_image_edit_outputs, torch.Tensor):
                    post_image_edit_logits = post_image_edit_outputs.logits
                    post_image_batch_labels = post_image_edit_outputs.labels
                else:
                    post_image_edit_logits = post_image_edit_outputs
                    post_image_batch_labels = batch["edit_outer_image"]["labels"]
            
            l_edit = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels, multimodal=True)["nll"]
            l_image_edit = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels, multimodal=True)["nll"]          
            
            # Collect some useful metrics
            with torch.no_grad():
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels, multimodal=True)
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels, multimodal=True)
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels, multimodal=True)
            
            post_base_outputs = edited_model(batch["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
                kl_mask = post_base_outputs.attention_mask
            else:
                post_base_logits = post_base_outputs
                kl_mask = torch.ones(post_base_logits.shape[0], post_base_logits.shape[1]).to(post_base_logits.device)

            post_image_base_outputs = edited_model(batch["loc_image"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
                kl_image_mask = post_image_base_outputs.attention_mask
            else:
                post_image_base_logits = post_image_base_outputs
                kl_image_mask = torch.ones(post_image_base_logits.shape[0], post_image_base_logits.shape[1]).to(base_image_logits.device)

            l_loc = kl_loc_loss(base_logits.detach(), post_base_logits, mask=kl_mask)
            l_image_loc = kl_loc_loss(base_image_logits.detach(), post_image_base_logits, mask=kl_image_mask)

        if l_edit.isnan():
            print("l_edit is nan")
            print("input: ", batch["edit_outer"]['text_input'])
        elif l_image_edit.isnan():
            print("l_image_edit is nan")
            print("input: ", batch["edit_outer_image"]['text_input'])
        elif l_loc.isnan():
            print("l_loc is nan")
            print("input: ", batch["loc"]['text_input'])
        elif l_image_loc.isnan():
            print("l_image_loc is nan")
            print("input: ", batch["loc_image"]['text_input'])

        if self.config.alg == "SERAC_MULTI":
            l_total_edit = self.config.cedit * l_edit + self.config.cloc * l_loc + self.config.iedit * l_image_edit
        else:
            l_total_edit = self.config.cedit * l_edit + self.config.cloc * (l_loc + l_image_loc) + self.config.iedit * l_image_edit
        
        if training and self.config.alg.lower() not in ['ft', 'ke']:
            safe_backward(l_total_edit, self.model.outer_parameters(), self.config.accumulate_bs, allow_unused=True)
        
        # Text locality
        post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
        base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=1, dim=-1).indices

        # Image locality
        post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
        base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits, dim=-1), k=10, dim=-1).indices

        info_dict = {}
        info_dict['loss/edit'] = l_edit.item()
        info_dict['loss/image_edit'] = l_image_edit.item()
        info_dict['loss/loc'] = l_loc.item()
        info_dict['edit/acc'] = post_edit_dict["acc"].item() # text rephrase acc
        info_dict['edit/log_prob'] = post_edit_dict["log_prob"].item()
        info_dict['edit/prob'] = post_edit_dict["prob"].item()
        info_dict['inner/acc'] = inner_edit_dict["acc"].item() # edit acc
        info_dict['image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        info_dict["time/edit"] = edit_time
        info_dict["loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        info_dict["image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
        l_base = torch.tensor(0.0)
        l_total = l_total_edit + self.config.cbase * l_base

        info_dict["loss/total"] = l_total.item()
        info_dict["loss/total_edit"] = l_total_edit.item()
        info_dict["memory/alloc_max"] = torch.cuda.max_memory_allocated()
        info_dict["memory/res_max"] = torch.cuda.max_memory_reserved()
        
        if 'img_topk' in batch["edit_inner"].keys() or 'ori_rt_img_topk' in batch["edit_inner"].keys():
            # minigpt4 or blip2
            tokenizer = self.model.model.llama_tokenizer if 'vicuna' in self.config.name.lower() else self.model.model.opt_tokenizer
            keys = ['img_topk', 'txt_topk', 'img_last_topk', 'txt_last_topk']
            keys += ['ori_rt_img_topk', 'ori_rt_txt_topk', 'ori_rt_img_last_topk', 'ori_rt_txt_last_topk']
            res_acc, res_pred, targ = {}, {}, {}
            for key in keys:
                t_acc, t_pred, t_targ = [], [], []
                for i in range(len(batch["edit_inner"][key])):
                    sample = batch["edit_inner"][key][i]
                    device = batch["edit_outer"]['image'].device
                    sample['image'] = sample['image'].to(device)
                    with torch.no_grad():
                        key_post_edit_output = edited_model(sample)
                    if not isinstance(key_post_edit_output, torch.Tensor):
                        key_post_edit_logits = key_post_edit_output.logits.cpu()
                        key_post_batch_labels = key_post_edit_output.labels.cpu()
                    else:
                        key_post_edit_logits = key_post_edit_output
                        key_post_batch_labels = sample['labels'].to(device)
                    
                    with torch.no_grad():
                        key_output = self.model.edit_loss_fn(self.config, key_post_edit_logits, key_post_batch_labels, multimodal=True)
                    
                    # t_pred.append([key_output['pred'][key_output['pred']!=0]])
                    # t_targ.append([key_output['targ'][key_output['targ']!=0]])
                    # t_acc.append(key_output['acc'].cpu())
                    if 'ori' in key: # KPI
                        t_targ.append([sample[f'ori_pred_{self.config.model_name.lower()}']])
                    else: # KGI
                        t_targ.append([key_output['targ'][key_output['targ']!=0]])
                    t_pred.append([key_output['pred'][key_output['pred']!=0]])
                    t_acc.append(sum(t_targ[-1][0]==t_pred[-1][0])/len(t_pred[-1][0]))
                    
                    sample['image'] = sample['image'].cpu()
                    torch.cuda.empty_cache()
                res_acc[key] = t_acc
                res_pred[key] = t_pred #[tokenizer.batch_decode(txt, skip_special_tokens=True) for txt in t_pred]
                targ[key] = t_targ #[tokenizer.batch_decode(txt, skip_special_tokens=True) for txt in t_targ]
                info_dict[key+'/acc'] = sum(res_acc[key])/len(res_acc[key])
            
            info_dict["general_dict_acc"] = res_acc
            info_dict["general_dict_pred"] = res_pred
            info_dict["general_dict_targ"] = targ
        
        info_dict = {**info_dict, **model_info}
        
        if self.config.alg.lower() in ['ft']:
            self.model.recover_ori_model()
        return l_total, l_edit, l_loc, l_base, info_dict

    def train_step(self, batch):
        l_total, l_edit, l_loc, l_base, info_dict = self.edit_step(
            batch, training=True
        )

        if self.global_iter > 0 and self.global_iter % self.config.accumulate_bs == 0:
            grad = torch.nn.utils.clip_grad_norm_(
                self.model.outer_parameters(),
                self.config.grad_clip,
                error_if_nonfinite=True,
            )
            info_dict['grad'] = grad.item()

            self.opt.step()
            self.opt.zero_grad()

            if self.lr_opt is not None:
                self.lr_opt.step()
                self.lr_opt.zero_grad()

                for lr_idx, lr in enumerate(self.model.edit_lrs):
                    info_dict[f'lr/lr{lr_idx}'] = lr.item()

        return info_dict

    def _inline_validation_log(self, step, stats, start_time, steps):
        elapsed = (time.time() - start_time) / (step + 1)
        prog = f"{step+1}/{steps}".ljust(20)
        inner_acc = f"{stats['inner/acc_val']:<12.5f}"
        outer_acc = f"{stats['edit/acc_val']:<12.5f}"
        image_acc = f"{stats['image_rephrase/acc_val']:<12.5f}"
        loc_acc = f"{stats['loc/acc_val']:<12.5f}"
        loc_image_acc = f"{stats['image_loc/acc_val']:<12.5f}"

        LOG.info(
          f"Step {prog} outer_acc: {outer_acc} image_acc: {image_acc} inner_acc: {inner_acc} it_time: {elapsed:.4f} loc_acc: {loc_acc}, image_loc: {loc_image_acc}"
        )

    def validate_ori(self, steps=None, log: bool = False):
        if steps is None or steps > len(self.val_set):
            steps = len(self.val_set)

        if log:
            LOG.info(f"Beginning evaluation for {steps} steps...")
        averager = RunningStatAverager("val")

        start_time = time.time()
        for val_step, batch in enumerate(self.val_loader):
            if val_step >= steps:
                break
            _, _, _, _, info_dict = self.edit_step(batch, training=False)
            averager.add(info_dict)

            if (
                log
                and (val_step + 1) % self.config.log_interval == 0
            ):
                self._inline_validation_log(
                    val_step, averager.average(), start_time, steps
                )

        if log:
            self._inline_validation_log(val_step, averager.average(), start_time, steps)
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        return stats

    def validate(self, steps=None, log: bool = False):
        if steps is None or steps > len(self.val_set):
            steps = len(self.val_set)

        if log:
            LOG.info(f"Beginning evaluation for {steps} steps...")
        averager = RunningStatAverager("val")

        start_time = time.time()
        keys = ['img_topk', 'txt_topk', 'img_last_topk', 'txt_last_topk']
        keys += ['ori_rt_img_topk', 'ori_rt_txt_topk', 'ori_rt_img_last_topk', 'ori_rt_txt_last_topk']
        res = {'acc':{}, 'pred':{}, 'targ':{}}
        for key in keys:
            for key1 in res.keys():
                res[key1][key] = []
        # file_path_avg_checkpoint = '{}_{}_{}_general_all_checkpoint.pth'.format(self.config.alg, self.config.model_name, self.config.task)
        # file_path_avg            = '{}_{}_{}_general_final.pth'.format(self.config.alg, self.config.model_name, self.config.task)
        file_path_acc_checkpoint = 'in_domain_{}_{}_{}_checkpoint.pth'.format(self.config.alg, self.config.model_name, self.config.task)
        file_path_acc            = 'in_domain_{}_{}_{}_final.pth'.format(self.config.alg, self.config.model_name, self.config.task)
        for val_step, batch in enumerate(self.val_loader):
            if val_step > 0 and val_step % 50 == 0:
                LOG.info('val_step={}'.format(val_step))
            _, _, _, _, info_dict = self.edit_step(batch, training=False)
            if 'general_dict_acc' in info_dict.keys():
                for ind in ['acc', 'pred', 'targ']:
                    for key in keys:
                        res[ind][key].append(info_dict['general_dict_'+ind][key])
                    del info_dict['general_dict_'+ind]
            averager.add(info_dict)

            if (
                log
                and (val_step + 1) % self.config.log_interval == 0
            ):
                self._inline_validation_log(
                    val_step, averager.average(), start_time, steps
                )
            
            if val_step > 0 and val_step % 50 == 0 and res['acc']['img_topk'] != []:
                # torch.save(averager.average(), file_path_avg_checkpoint)
                torch.save(res, file_path_acc_checkpoint)
        if res['acc']['img_topk'] != []:
            # torch.save(averager.average(), file_path_avg)
            # os.remove(file_path_avg_checkpoint)
            torch.save(res, file_path_acc)
            os.remove(file_path_acc_checkpoint)

        if log:
            self._inline_validation_log(val_step, averager.average(), start_time, steps)
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        return stats

    def test_on_ori_right(self,):
        self.config.device = 0
        self.tokenizer = self.model.model.opt_tokenizer if self.config.model_name == 'blip2' else self.model.model.llama_tokenizer
        preds, targs, acc_old = [], [], []
        # LOG.info(f"{len(self.val_set.ori_right)} samples in total")
        # for step, batch in enumerate(self.val_set.ori_right):
        LOG.info(f"{len(self.val_set.all_edit_inner)} samples in total")
        for step, batch in enumerate(self.val_set.all_edit_inner):
            if (step+1)%100 == 0:
                LOG.info(f"Dealed {step+1} samples")
            batch['image'] = batch['image'].to(self.config.device)
            # _, _, post_edit_dict = self.get_outputs(self.model, batch, multimodal=True)
            pred, targ, post_edit_dict = self.get_predict(self.model, batch, multimodal=True)
            batch['image'] = batch['image'].cpu()
            torch.cuda.empty_cache()
            # preds.append(pred)
            preds.append(post_edit_dict['pred'][post_edit_dict['pred']!=0])
            targs.append(targ)
            acc_old.append(post_edit_dict['acc'])
            # import pdb
            # pdb.set_trace()
        torch.save({'preds':preds, 'targs':targs, 'acc_old':acc_old}, 'comprehendedit_all_inner_test_{}.pth'.format(self.config.model_name.lower()))
        import pdb
        pdb.set_trace()

    def get_outputs(self, model, data, multimodal=True, grad=False, hidden_states=False):
        with torch.set_grad_enabled(grad):
            post_edit_outputs = model(data)
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
                post_batch_labels = post_edit_outputs.labels
            else:
                post_edit_logits = post_edit_outputs
                post_batch_labels = data["labels"]
            with torch.no_grad():
                post_edit_dict = model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels, multimodal=multimodal)
        if not hidden_states:
            return post_edit_logits, post_batch_labels, post_edit_dict
        else:
            return post_edit_logits, post_batch_labels, post_edit_dict, post_edit_outputs['hidden_states']
    
    def get_predict(self, model, data, multimodal=True):
        _, _, post_edit_dict = self.get_outputs(model, data, multimodal=multimodal)
        pred = self.tokenizer.batch_decode([post_edit_dict['pred'][post_edit_dict['pred']!=0]], skip_special_tokens=True)[0]
        targ = self.tokenizer.batch_decode([post_edit_dict['targ'][post_edit_dict['targ']!=0]], skip_special_tokens=True)[0]
        return pred, targ, post_edit_dict
