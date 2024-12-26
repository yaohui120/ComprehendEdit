import torch
import copy
import transformers
import logging

from ..utils import scr, set_dropout, _logits, add_padding, add_sep
from .editable_model import EditableModel

LOG = logging.getLogger(__name__)

from collections import OrderedDict
from transformers.activations import ACT2FN
# import pyvene as pv
# import sys
# sys.path.append("/home/myh/pyvene")
# from pyvene import (
#     ConstantSourceIntervention,
#     SourcelessIntervention,
#     TrainableIntervention,
#     DistributedRepresentationIntervention,
# )
# from pyvene.models.layers import LowRankRotateLayer

# sys.path.append("/home/myh/pyreft")
# import pyreft

import pdb

import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import util


# sys.path.append("/mnt/Data/myh/CLIP")
# from clip import clip
# from PIL import Image
from transformers.utils import ModelOutput
from dataclasses import dataclass
from typing import Optional

import numpy as np
from typing import List
@dataclass
class OurOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    labels: Optional[torch.IntTensor] = None
    fea: Optional[torch.FloatTensor] = None
    embeddings: Optional[torch.FloatTensor] = None
    attention_mask: torch.IntTensor = None
    fc_labels: torch.IntTensor = None
    fc_preds: torch.FloatTensor = None

class HICE(EditableModel):
    # 2个memory，1个存训练集中的样本，1个存测试时当前的编辑样本
    def __init__(self, model, config, model_constructor, sentence_model=None, classifier=None,
                 cache_sentences=None, cache_sentences_embed=None, icl_examples=None, edit_sample=None,
                 Wo=None, W_rand=None, count=None):
        super().__init__(model, config, model_constructor)
        self.embed_dim = 384
        self.class_num = 2
        
        from sentence_transformers import SentenceTransformer
        self.sentences = cache_sentences if cache_sentences is not None else []
        self.sentences_embed = cache_sentences_embed if cache_sentences_embed is not None else []
        
        # 用于提取文本embedding的模型
        self.sentence_model = sentence_model if sentence_model is not None else SentenceTransformer(config.sentence_model_name).to(f'cuda:{config.device}')
        self.classifier = classifier if classifier is not None else nn.Linear(self.embed_dim, self.class_num) # clip 32:512
        
        self.A = torch.zeros(self.class_num, self.embed_dim, self.embed_dim).to(self.config.device)
        self.b = torch.zeros(self.class_num, self.embed_dim, 1).to(self.config.device)
        self.count = torch.tensor([0 for i in range(self.class_num)]).to(self.config.device)
        
        self.fea, self.labels = [], []
        M = self.config.M
        self.W_rand = W_rand if W_rand is not None else torch.randn(self.embed_dim, M).to(self.config.device)
        self.Wo = Wo if Wo is not None else None
        
        # 编辑目标的近邻样本
        self.icl_examples = icl_examples if icl_examples is not None else []
        self.tok = self.model.llama_tokenizer if 'vicuna' in self.config.name.lower() else self.model.opt_tokenizer
        self.edit_sample = edit_sample if edit_sample is not None else []
        
        self.classifier.to(self.config.device)
        for name, params in self.classifier.named_parameters():
            params.requires_grad = True
        self.label_map = {'gqa':1, 'tallyqa':2, 'vsr':3, 'textvqa':4, 'mathvista':5}
        self.G, self.Q = 0, 0
        
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(prefix=prefix, keep_vars=keep_vars)  # Get default state dict
        model_keys = self.model.state_dict(prefix=prefix, keep_vars=keep_vars).keys()  # Remove model params
        for k in model_keys:
            del state_dict[f"model.{k}"]
        if self.config.freeze_cntr:
            cntr_keys = self.replacement.state_dict().keys()
            for k in cntr_keys:
                del state_dict[f"replacement.{k}"]
        state_dict["model_config"] = self.model.config  # Include model config
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        config = state_dict["model_config"]
        del state_dict["model_config"]
        if config != self.model.config:
            LOG.info("Loaded model config doesn't match current model config.")
            LOG.info(f"Loaded: {config}")
            LOG.info(f"Current: {self.model.config}")

        if self.config.freeze_cntr:
            rep_keys = list(state_dict.keys())
            for k in rep_keys:
                if k.startswith("replacement"):
                    del state_dict[k]
            res = super().load_state_dict(state_dict, False)
        else:
            res = super().load_state_dict(state_dict, False)

        # We should only have missing keys for the model, and no unexpected keys
        def ok_to_miss(k):
            return k.startswith("model.") or (self.config.freeze_cntr and k.startswith("replacement."))
        missing_keys = [k for k in res.missing_keys if not ok_to_miss(k)]
        assert len(missing_keys) == 0, f"Should only have missing keys for model: {missing_keys}."
        assert len(res.unexpected_keys) == 0, "Shouldn't have any unexpected keys"
        return res

    def outer_parameters(self):
        model_params = list(self.classifier.parameters())
        return model_params

    def recover_ori_model(self):
        self.sentences = []
        self.sentences_embed = []
        self.icl_examples = []
        self.edit_sample = []
        self.fea, self.labels = [], []
        self.G, self.Q = 0, 0

    def get_txt_embedding(self, input):
        res = torch.tensor(self.sentence_model.encode(input, show_progress_bar=False, device=self.config.device))
        return res

    def edit(self, batch, condition=None, detach_history=False):
        train_data = batch
        new_fact = train_data['prompt'][0] + ' ' + train_data['target'][0]
        target_new = train_data['target'][0]
        paraphrases = train_data['rephrase_prompt'][0]
        # neighbors = train_data['locality_prompt'][0]
        # neighbors_ans = train_data['locality_ground_truth'][0]
        self.sentences.append(f"New Fact: {new_fact}\nPrompt: {new_fact}\n\n")
        self.sentences.append(f"New Fact: {new_fact}\nPrompt: {paraphrases} {target_new}\n\n")
        # self.sentences.append(f"New Fact: {new_fact}\nPrompt: {neighbors} {neighbors_ans}\n\n")

        # self.sentences_embed = self.get_txt_embedding(self.sentences)
        # self.icl_examples = self.find_icl_examples(batch)
        self.sentences_embed = [1]
        self.icl_examples = [1]
        
        new_model = HICE(self.model, self.config, self.model_constructor, self.sentence_model, self.classifier,
                        self.sentences, self.sentences_embed, self.icl_examples, train_data, self.Wo, self.W_rand, self.count)
        new_model.train(self.training)
        return new_model, {}

    def find_icl_examples(self, request):
        stored_embeddings = self.sentences_embed.to(self.config.device)
        stored_embeddings = stored_embeddings / torch.norm(stored_embeddings, p=2, dim=-1).unsqueeze(-1)

        new_fact = request['prompt'][0] + ' ' + request['target'][0]
        query_sentence = f"New Fact: {new_fact}\nPrompt: {new_fact}\n\n"
        query_embedding = self.get_txt_embedding(query_sentence).unsqueeze(0).to(self.config.device)
        query_embedding = query_embedding / torch.norm(query_embedding, p=2, dim=-1)

        hits = util.semantic_search(query_embedding, stored_embeddings, score_function=util.dot_score, top_k=self.config.k)
        assert len(hits) == 1
        hit = hits[0]
        icl_examples = [self.sentences[hit[k]["corpus_id"]] for k in range(len(hit))]
        icl_examples.append(f'New Fact: {new_fact}\nPrompt: {new_fact}\n\n')
        return icl_examples

    def run_classifier(self, inputs):
        new_fact = inputs['prompt'][0] + ' ' + inputs['target'][0]
        query_sentence = f"New Fact: {new_fact}\nPrompt: {new_fact}\n\n"
        query_embedding = self.get_txt_embedding(query_sentence).unsqueeze(0).to(self.config.device)
        query_fea = query_embedding / torch.norm(query_embedding, p=2, dim=-1)

        # stored_embeddings = util.normalize_embeddings(self.sentences_embed).to(self.config.device)
        # sim = query_embedding @ stored_embeddings.T
        
        sim = self.classifier(query_fea)
        # logits = sim / sim.sum()
        logits = sim
        preds = torch.max(logits, dim=1)[1]
        return logits, preds, query_fea
    
    def optimise_ridge_parameter(self, Features, Y):
        # ridges = 10.0**np.arange(-8, 9) # ori
        ridges = 10.0**np.arange(-4, 5)
        num_val_samples = int(Features.shape[0]*0.8)
        losses=[]
        Q_val = Features[0:num_val_samples,:].T @ Y[0:num_val_samples,:]
        G_val = Features[0:num_val_samples,:].T @ Features[0:num_val_samples,:]
        for ridge in ridges:
            Wo = torch.linalg.solve(G_val + ridge*torch.eye(G_val.size(dim=0)), Q_val).T #better nmerical stability than .inv
            Y_train_pred = Features[num_val_samples::,:] @ Wo.T
            losses.append(F.mse_loss(Y_train_pred,Y[num_val_samples::,:]))
        ridge = ridges[np.argmin(np.array(losses))]
        logging.info("Optimal lambda: "+str(ridge))
        return ridge
    
    def target2onehot(self, targets, n_classes):
        onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
        onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
        return onehot
    
    def get_Wo(self):
        fea = torch.cat(self.fea, dim=0).cpu()
        label_list = torch.cat(self.labels, dim=0).cpu()
        Y = self.target2onehot(label_list, self.class_num) # [N, 2]
        # Features_h = fea
        Features_h = torch.nn.functional.relu(fea @ self.W_rand.cpu()) # [N, dim] X [dim, M] = [N, M]
        self.Q = Features_h.T @ Y # [M, 2]
        self.G = Features_h.T @ Features_h # [M, M]
        ridge = self.optimise_ridge_parameter(Features_h, Y)
        self.Wo = torch.linalg.solve(self.G + ridge*torch.eye(self.G.size(dim=0)), self.Q).T # better nmerical stability than .inv
        self.Wo = self.Wo.to(self.config.device)
        return self.Wo
     
    def reset_Wo(self, Wo):
        self.Wo = Wo.clone()
    
    def run_classifier2(self, inputs, mloc_memory=None):
        if mloc_memory is not None:
            new_fact = inputs['prompt'][0] + ' ' + '' #inputs['target'][0]
            query_sentence = f"Prompt: {new_fact}\n\n"
            query_embedding1 = self.get_txt_embedding(query_sentence).unsqueeze(0).to(self.config.device)
            query_fea1 = query_embedding1 / torch.norm(query_embedding1, p=2, dim=-1)

            memory_fea = torch.cat(mloc_memory['embeddings'], dim=0)
            memory_fea = memory_fea / torch.norm(memory_fea, p=2, dim=-1).unsqueeze(-1)
            memory_fea = memory_fea.to(query_fea1.device)
            sim = query_fea1 @ memory_fea.T
            if sim.max() > 0.81:
                return None, torch.tensor([0]).to(self.config.device), query_fea1, query_embedding1
        
        # RanPAC
        new_fact = inputs['prompt'][0] + ' ' + inputs['target'][0]
        query_sentence = f"New Fact: {new_fact}\nPrompt: {new_fact}\n\n"
        query_embedding = self.get_txt_embedding(query_sentence).unsqueeze(0).to(self.config.device)
        query_fea = query_embedding / torch.norm(query_embedding, p=2, dim=-1)
        
        # query_fea_h = query_fea
        query_fea_h = torch.nn.functional.relu(query_fea @ self.W_rand)
        logits = (query_fea_h @ self.Wo.T)
        preds = torch.max(logits, dim=-1)[1]
        
        return logits, preds, query_fea, query_embedding

    def compute_multimodal_edit_quality(self, model, batch, exach_match=False):
        with torch.no_grad():
            outputs = model(batch)
            if isinstance(outputs, torch.Tensor):
                logits = outputs.detach()
                targ = batch["labels"]
                attn_mask = outputs.attention_mask
            else:
                logits = outputs.logits.detach()
                targ = outputs.labels.detach()
                attn_mask = outputs.attention_mask
        logits_ = logits.clone()
        # if logits.dim() == 3:
        #     logits = logits[:, :-1]
        #     targ = targ[:, 1:]
        #     # logits = logits[:, -targ.shape[1]:]
        # mask = targ != -100
        # targ[~mask] = 0
        # exach_match=True
        # pred_ids = logits.argmax(-1).masked_fill(~mask, 0)
        # correct = pred_ids == targ
        # if logits.dim() == 3:
        #     correct = (pred_ids == targ).all(-1)  # We aim for an exact match across the entire sequence
        # acc = correct.float().mean()

        return logits_, targ, attn_mask

    def forward2(self, inputs, train=False, **kwargs):
        from easyeditor.evaluate.multimodal_evaluate import prepare_multimodal_edit
        
        # [ori_class, edit_class]
        if self.class_num == 2:
            classifer_label = [0] if 'locality' in inputs['cat'] else [1]
        else:
            classifer_label = [self.label_map[inputs['source'][0].lower()]] if 'source' in inputs.keys() else [0]
        classifer_label = torch.tensor(classifer_label).to(self.config.device)
        
        if train:
            # construct demonstrations like ike
            new_fact = inputs['prompt'][0] + ' ' + inputs['target'][0]
            query_sentence = f"New Fact: {new_fact}\nPrompt: {new_fact}\n\n"
            query_embedding = self.get_txt_embedding(query_sentence).unsqueeze(0).to(self.config.device)
            fea = query_embedding / torch.norm(query_embedding, p=2, dim=-1)
            self.fea.append(fea.cpu())
            self.labels.append(classifer_label.cpu())
            return None
        
        mloc_memory = None
        if 'mloc_memory' in kwargs.keys():
            mloc_memory = kwargs['mloc_memory']
        logits, preds, fea, query_embedding = self.run_classifier2(inputs, mloc_memory=mloc_memory)
        # loss = F.cross_entropy(logits, classifer_label)
        loss = 0.
        
        # 原始模型跑loc, loc_img; if is loc sample. 
        if len(self.edit_sample) == 0 or preds == 0:
            # 未编辑时直接用原模型的输出
            # outputs = self.forward_ori(inputs, **kwargs)
            return OurOutput(
                # logits=outputs.logits,
                # labels=outputs.labels,
                # attention_mask=outputs.attention_mask,
                fea=fea,
                embeddings=query_embedding,
                labels=classifer_label,
                loss=loss,
                fc_labels=classifer_label,
                fc_preds=preds,
            )
 
        # 如果在域内
        # attn_mask = None
        # prompt = self.edit_sample['prompt'][0]
        # target = self.edit_sample['target'][0]
        
        # image = inputs["image"] if "image" in inputs.keys() else None
        # qs = inputs["prompt"][0]
        # targets = inputs["target"][0]
        
        # assert len(self.icl_examples) != 0
        # new_fact = f'New Fact: {prompt} {target}\nPrompt: {qs}'
        # samples = prepare_multimodal_edit(self.config, self.tok, targets, [''.join(self.icl_examples) + f'{new_fact}'], image)
        # logits, labels, attn_mask = self.compute_multimodal_edit_quality(self.model, samples, self.config.exact_match)
        
        # from ..blip2_models.mini_gpt4 import MiniGPTOutput
        return OurOutput(
                # logits=logits,
                # labels=labels,
                # attention_mask=attn_mask,
                fea=fea,
                embeddings=query_embedding,
                labels=classifer_label,
                loss=loss,
                fc_labels=classifer_label,
                fc_preds=preds,
            )
    
    def forward_ori(self, inputs, **kwargs):
        if 'minigpt4' in self.config.model_name.lower() or 'blip' in self.config.model_name.lower():
            outputs = self.model(inputs, **kwargs)
        elif 'gpt' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask']))
            # outputs = outputs[:, -kwargs['labels'].shape[-1]:, :]
        elif 'llama' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask']))
            # outputs = outputs[:, -kwargs['labels'].shape[-1]:, :]
        elif 'chatglm2' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask']))
            # outputs = outputs[:, -kwargs['labels'].shape[-1]:, :]
        elif 'internlm' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask']))
            # outputs = outputs[:, -kwargs['labels'].shape[-1]:, :]
        elif 'qwen' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask']))
            # outputs = outputs[:, -kwargs['labels'].shape[-1]:, :]
        elif 'mistral' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask']))
            # outputs = outputs[:, -kwargs['labels'].shape[-1]:, :]
        else:
            outputs = _logits(self.model(**kwargs))
        return outputs

    def exact_train_features(self, loader):
        import torch
        from clip import clip
        from PIL import Image
        from scipy.spatial.distance import cdist
        import numpy as np
        import os

        if os.path.exists(f'{self.config.task.lower()}_train_img_txt_fea.pth'):
            results = torch.load(f'{self.config.task.lower()}_train_img_txt_fea.pth')
            return results
        device = "cuda:{}".format(self.config.device)
        model, preprocess = clip.load("ViT-B/32", device=device)

        img_fea, txt_fea = [], []
        results = []
        pid = []
        for i, batch in enumerate(loader):
            if i > 0 and i % 200 == 0:
                checkpoint = {'img_fea':img_fea, 'txt_fea':txt_fea, 'pid':pid, 'step':i}
                torch.save(checkpoint, f'{self.config.task.lower()}_train_img_txt_fea_checkpoint.pth')
                logging.info('{}th sample'.format(i))
            img = batch['edit_inner']['image_path'][0]
            image = preprocess(Image.open(img)).unsqueeze(0).to(device)
            batch['edit_inner']['ori_qs'] = [batch['edit_inner']['ori_qs'][0][:70]]
            text = clip.tokenize(batch['edit_inner']['ori_qs']).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)

            img_fea.append(image_features)
            txt_fea.append(text_features)
            pid.append(batch['edit_inner']['pid'][0])
            
        results = {'img_feas':torch.cat(img_fea, dim=0).cpu(),
                   'txt_feas':torch.cat(txt_fea, dim=0).cpu(),
                   'pid':pid}
        
        EPSILON = 1e-8
        def calc_dist(vectors):
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            dists = cdist(vectors, vectors, 'sqeuclidean')  # [nb_classes, N]
            return dists.T
        
        results['img_fea_sim'] = calc_dist(torch.cat(img_fea, dim=0).cpu())
        results['txt_fea_sim'] = calc_dist(torch.cat(txt_fea, dim=0).cpu())
        results['pid'] = pid
        
        torch.save(results, f'{self.config.task.lower()}_train_img_txt_fea.pth')
        os.remove(f'{self.config.task.lower()}_train_img_txt_fea_checkpoint.pth')
        return results

    def train_classifier(self, train_loader, test_loader):
        import logging, os
        weight_file = 'ranpac_{}_{}.pth'.format(self.config.task.lower(), self.config.sentence_model_name.split('/')[-1])

        if os.path.exists(weight_file):
            params = torch.load(weight_file)
            params['Wo'], params['W_rand'] = params['Wo'].to(self.config.device), params['W_rand'].to(self.config.device)
            self.fea, self.labels, self.Wo = params['fea'], params['labels'], params['Wo']
        else:
            keys = ["edit_inner", "edit_outer", "loc", "loc_image"]
            fea, labels = [], []
            for i, batch in enumerate(train_loader):
                self.recover_ori_model()
                edited_model, _ = self.edit(batch["edit_inner"], batch["cond"])
                for key in keys:
                    _ = edited_model.forward2(batch[key], train=True)
                    fea += edited_model.fea
                    labels += edited_model.labels
                    
                if i > 0 and i % 1000 == 0:
                    logging.info('Training Done {}th sample'.format(i))
                    
            self.fea, self.labels = fea, labels
            Wo = self.get_Wo()
            params = {'Wo':Wo.cpu(), 'W_rand':self.W_rand.cpu().clone(),'fea':fea, 'labels':labels}
            torch.save(params, weight_file)
        
        memory_file = 'memory_{}_{}.pth'.format(self.config.task.lower(), self.config.sentence_model_name.split('/')[-1])
        if os.path.exists(memory_file):
            memory = torch.load(memory_file)
        else:
            memory = self.create_mloc_memory(self.Wo, train_loader)
            torch.save(memory, memory_file)

        self.recover_ori_model()
        self.test_classifier_acc(params, test_loader, mloc_memory=memory)

    def test_classifier_acc(self, params, test_loader, mloc_memory=None):
        self.fea, self.labels, self.Wo, self.W_rand = params['fea'], params['labels'], params['Wo'].to(self.config.device), params['W_rand'].to(self.config.device)
        keys = ["edit_inner", "edit_outer", "loc", "loc_image"]
        accs = []
        for key in keys:
            accs.append([])
        for j, batch in enumerate(test_loader):
            self.recover_ori_model()
            if j % 500 == 0:
                logging.info('Testing Classifier Done {}th sample'.format(j))
            edited_model, _ = self.edit(batch["edit_inner"], batch["cond"])
            # RanPAC
            edited_model.W_rand = self.W_rand.clone()
            edited_model.reset_Wo(self.Wo)
            
            for k in range(len(keys)):
                key = keys[k]
                out = edited_model.forward2(batch[key], train=False, mloc_memory=mloc_memory)
                accs[k].append(out.fc_labels==out.fc_preds)
            torch.cuda.empty_cache()
        for k in range(len(keys)):
            tt = sum(accs[k]) / len(accs[k])
            print('{} test acc={}'.format(keys[k], np.round(tt.cpu().numpy(), decimals=4)))
        acc = sum([sum(tt) for tt in accs]) / (len(accs)*len(accs[0]))
        print('test acc={}'.format(np.round(acc.cpu().numpy(), decimals=4)))

    def create_mloc_memory(self, Wo, train_loader):
        # save hard mloc sample
        accs = []
        keys = ["loc_image"]
        memory = {'embeddings':[], 'sentences':[], 'qs':[], 'ans':[]}
        for key in keys:
            accs.append([])
        for j, batch in enumerate(train_loader):
            self.recover_ori_model()
            if j % 500 == 0 and j > 0:
                logging.info('Constructing Memory Done {}th sample'.format(j))
            edited_model, _ = self.edit(batch["edit_inner"], batch["cond"])
            # RanPAC
            edited_model.W_rand = self.W_rand.clone()
            edited_model.reset_Wo(Wo)
            
            for k in range(len(keys)):
                key = keys[k]
                out = edited_model.forward2(batch[key], train=False)
                accs[k].append(out.fc_labels==out.fc_preds)
                if out.fc_labels != out.fc_preds:
                    inputs = batch[key]
                    new_fact = inputs['prompt'][0] + ' ' + inputs['target'][0]
                    query_sentence = f"Prompt: {new_fact}\n\n"
                    if query_sentence not in memory['sentences']:
                        memory['embeddings'].append(out.embeddings)
                        memory['sentences'].append(query_sentence)
                        memory['qs'].append(inputs['prompt'][0])
                        memory['ans'].append(inputs['target'][0])
        torch.cuda.empty_cache()
        for k in range(len(keys)):
            tt = sum(accs[k]) / len(accs[k])
            logging.info('{} test acc={}'.format(keys[k], np.round(tt.cpu().numpy(), decimals=4)))
        print('memory size=',len(memory['qs']))
        memory['embeddings'] = [tt.cpu() for tt in memory['embeddings']]
        import pdb
        # pdb.set_trace()
        return memory


if __name__ == '__main__':
    import types

    model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")

    config = types.SimpleNamespace()
    config.inner_params = [
        "transformer.h.9.mlp.c_fc.weight",
        "transformer.h.9.mlp.c_proj.weight",
        "transformer.h.10.mlp.c_fc.weight",
        "transformer.h.10.mlp.c_proj.weight",
        "transformer.h.11.mlp.c_fc.weight",
        "transformer.h.11.mlp.c_proj.weight",
    ]
    config.edit_lr = 0.0001

    config.gtn = types.SimpleNamespace()
    config.gtn.n_hidden = 1
    config.gtn = config.gtn.__dict__

    gtn = our(model, config, lambda: copy.deepcopy(model)).cuda()
    # torch.save(gtn.state_dict(), "test_state.pt")
    import pdb; pdb.set_trace()
    gtn.load_state_dict(torch.load("test_state.pt"))
    x = torch.arange(20).view(1, 20).cuda() + 1000
    orig_logits = gtn(x)
    edited = gtn.edit(x, masks=torch.ones_like(x), labels=x)
    post_logits = gtn(x)

    assert torch.allclose(orig_logits, post_logits)

    orig_param = [p for (n, p) in gtn.model.named_parameters() if n == config.inner_params[-1]][0]
    edited_param = [p for (n, p) in edited.model.named_parameters() if n == config.inner_params[-1]][0]

    LOG.info((orig_param - edited_param).abs().max())
    edited.eval()
    LOG.info(gtn(x, labels=x).loss, edited(x, labels=x).loss, edited.edit_loss_fn(edited(x).logits, x)["nll"])
    edited2 = edited.edit(x, masks=torch.ones_like(x), labels=x)
    LOG.info(gtn(x, labels=x).loss, edited(x, labels=x).loss, edited2(x, labels=x).loss)
