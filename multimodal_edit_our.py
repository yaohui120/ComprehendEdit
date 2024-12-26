import torch
import types
from statistics import mean

from easyeditor import BaseEditor, MultimodalTrainer, MultimodalEditor
from easyeditor import CaptionDataset, VQADataset, ComprehendEdit, selectDataset
from easyeditor import MENDMultimodalTrainingHparams, SERACMultimodalTrainingHparams, IKEMultimodalHyperParams, MENDMultimodalHparams \
    , SERACMultimodalHparams
from easyeditor import encode_ike_facts_multimodal
from sentence_transformers import SentenceTransformer
import logging
import os
import json
import numpy as np
import pdb

DATASET_DICT = {
    'vqa': VQADataset,
    'comprehendedit': ComprehendEdit,
}
data_dir = {
    'vqa': '/home/myh/Datasets/MMEdit',
    'comprehendedit': '/home/myh/EasyEdit-main/published_version/ComprehendEdit',
}


def print_result(metrics):
    if 'rewrite_acc' in metrics[0]['post'].keys():
        rewrite_acc = mean([m['post']['rewrite_acc'].item() for m in metrics])
        print(f'rewrite_acc: {rewrite_acc}')
    # text generality
    if 'rephrase_acc' in metrics[0]['post'].keys():
        rephrase_acc = mean([m['post']['rephrase_acc'].item() for m in metrics])
        print(f'rephrase_acc: {rephrase_acc}')
    # visual generality
    if 'rephrase_image_acc' in metrics[0]['post'].keys():
        rephrase_image_acc = mean([m['post']['rephrase_image_acc'].item() for m in metrics])
        print(f'rephrase_acc: {rephrase_image_acc}')   
    # text locality
    if 'locality_acc' in metrics[0]['post'].keys():
        locality_acc = mean([m['post']['locality_acc'].item() for m in metrics])
        print(f'locality_acc: {locality_acc}')
    # multimodel locality
    if 'multimodal_locality_acc' in metrics[0]['post'].keys():
        locality_image_acc = mean([m['post']['multimodal_locality_acc'].item() for m in metrics])
        print(f'locality_image_acc: {locality_image_acc}')

def print_result_k(res, tok, daxiaoxie=False, topk=4, ori=False):
    assert res['acc'].keys() == res['pred'].keys() and res['acc'].keys() == res['targ'].keys(), 'keys are not equal!'
    keys = list(res['acc'].keys())

    for key in keys:
        assert len(res['acc'][key]) == len(res['pred'][key]) and len(res['acc'][key]) == len(res['targ'][key]), 'number of samples are not equal!'
    samples_num = len(res['acc'][keys[0]])
    
    if not ori:
        for KEY in ['pred', 'targ']:
            for key in keys: # ['img_topk', ...]
                for i in range(samples_num):
                    for j in range(len(res[KEY][key][i])):
                        if isinstance(res[KEY][key][i][j], list):
                            res[KEY][key][i][j] = tok.batch_decode(res[KEY][key][i][j], skip_special_tokens=True)

        # prediction of each sample [[],[],...]->[...]
        for KEY in ['pred', 'targ']:
            for key in keys: # ['img_topk', ...]
                if isinstance(res[KEY][key][0][0], list):
                    tt = []
                    for i in range(samples_num):
                        tt.append([res[KEY][key][i][j][0] for j in range(len(res[KEY][key][i]))])
                    res[KEY][key] = tt
        
        for key in keys:
            for i in range(samples_num):
                length = len(res['pred'][key][i])
                res['acc'][key][i] = [0 for j in range(length)]
                for j in range(length):
                    res['pred'][key][i][j] = res['pred'][key][i][j].strip()
                    res['targ'][key][i][j] = res['targ'][key][i][j].strip()
                    if daxiaoxie:
                        ttt =  res['pred'][key][i][j].lower() == res['targ'][key][i][j].lower()
                    else:
                        ttt = res['pred'][key][i][j] == res['targ'][key][i][j]   
                    if ttt:
                        res['acc'][key][i][j] = 1
                    else:
                        res['acc'][key][i][j] = 0
    
    values = []
    for key in keys:
        if 'last' not in key: 
            SUM = sum([sum(res['acc'][key][i][:topk]) for i in range(samples_num)])
        else:
            SUM = sum([sum(res['acc'][key][i][-topk:]) for i in range(samples_num)])
        if isinstance(SUM, torch.Tensor):
            SUM = SUM.cpu().item()
        values.append(np.round(100*SUM/(samples_num*topk), decimals=2))
    tt = ''
    for i in range(len(values)):
        tt = tt + keys[i] + ' '
    print(tt)
    tt = ''
    for i in range(len(values)):
        tt = tt + str(values[i]) + ','
    print(tt)

def Generate_Embedding_for_IKE(model='blip2'):
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/{}.yaml'.format(model))
    hparams.task = hparams.task_name
    Dataset = DATASET_DICT[hparams.task.lower()]
    print(f'Using datasets: {hparams.task.lower()}')

    train_ds = Dataset(data_dir[hparams.task.lower()], config=hparams, mode='train', topk=-1, diverse=True)
    ## Generate embedding files for IKE
    sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
    encode_ike_facts_multimodal(sentence_model, train_ds, hparams)

def test_IKE(model='blip2'):
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/{}.yaml'.format(model))
    hparams.task = hparams.task_name
    Dataset = DATASET_DICT[hparams.task.lower()]
    print(f'Using datasets: {hparams.task.lower()}')

    editor = MultimodalEditor.from_hparams(hparams)
    eval_ds = Dataset(data_dir[hparams.task.lower()], config=hparams, mode='test', topk=5, size=None)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True
    )
    # torch.save(metrics, 'result_IKE_{}_{}.pth'.format(model, hparams.task.lower()))
    print_result(metrics)

def train_FT(model='blip2', train=True):
    from easyeditor import FTMultimodalTrainingHparams
    hparams = FTMultimodalTrainingHparams.from_hparams('hparams/TRAINING/FT/{}.yaml'.format(model))
    Dataset = DATASET_DICT[hparams.task.lower()]
    print(f'Using datasets: {hparams.task.lower()}')
    
    eval_ds = Dataset(data_dir[hparams.task.lower()], config=hparams, mode='test', topk=5, size=None)
    train_ds = Dataset(data_dir[hparams.task.lower()], config=hparams, mode='train', topk=-1) if train else eval_ds
    
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    if train:
        print('Start Training...')
        trainer.run()
    else:
        print('Start Testing...')
        # make sure hparams.archive is not null
        # assert hparams.archive is not None
        val_steps = len(eval_ds._data)+1
        val_info = trainer.validate(log=True)
        trainer.echo(val_steps, val_info, pretty=True)
        # tokenizer = trainer.model.model.opt_tokenizer if hparams.model_name.lower()=='blip2' else trainer.model.model.llama_tokenizer
        # print_res('in_domain_{}_{}_{}_final.pth'.format(self.config.alg, self.config.model_name, self.config.task), tokenizer)

def train_SERAC(model='blip2', train=True):
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/{}.yaml'.format(model))
    Dataset = DATASET_DICT[hparams.task.lower()]
    print(f'Using datasets: {hparams.task.lower()}')

    eval_ds = Dataset(data_dir[hparams.task.lower()], config=hparams, mode='test', topk=5, size=None)
    train_ds = Dataset(data_dir[hparams.task.lower()], config=hparams, mode='train', topk=-1) if train else eval_ds
    
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    if train:
        print('Start Training...')
        trainer.run()
    else:
        print('Start Testing...')
        # make sure hparams.archive is not null
        # assert hparams.archive is not None
        val_steps = len(eval_ds._data)+1
        val_info = trainer.validate(log=True)
        trainer.echo(val_steps, val_info, pretty=True)
        # tokenizer = trainer.model.model.opt_tokenizer if hparams.model_name.lower()=='blip2' else trainer.model.model.llama_tokenizer
        # print_res('in_domain_{}_{}_{}_final.pth'.format(self.config.alg, self.config.model_name, self.config.task), tokenizer)

def train_MEND(model='blip2', train=True):
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/{}.yaml'.format(model))
    Dataset = DATASET_DICT[hparams.task.lower()]
    print(f'Using datasets: {hparams.task.lower()}')
    
    eval_ds = Dataset(data_dir[hparams.task.lower()], config=hparams, mode='test', topk=5, size=None)
    train_ds = Dataset(data_dir[hparams.task.lower()], config=hparams, mode='train', topk=-1) if train else eval_ds
    
    hparams.alg = "MEND_MULTIGPU"
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    if train:
        print('Start Training...')
        trainer.run()
    else:
        print('Start Testing...')        
        # make sure hparams.archive is not null
        # assert hparams.archive is not None
        val_steps = len(eval_ds._data)+1
        val_info = trainer.validate(log=True)
        trainer.echo(val_steps, val_info, pretty=True)
        # tokenizer = trainer.model.model.opt_tokenizer if hparams.model_name.lower()=='blip2' else trainer.model.model.llama_tokenizer
        # print_res('in_domain_{}_{}_{}_final.pth'.format(self.config.alg, self.config.model_name, self.config.task), tokenizer)

def train_HICE(model='blip2', train=True):
    from easyeditor import hiceMultimodalTrainingHparams
    hparams = hiceMultimodalTrainingHparams.from_hparams('hparams/TRAINING/HICE/{}.yaml'.format(model))
    Dataset = DATASET_DICT[hparams.task.lower()]
    print(f'Using datasets: {hparams.task.lower()}')
    
    eval_ds = Dataset(data_dir[hparams.task.lower()], config=hparams, mode='test', topk=5, size=None)
    train_ds = Dataset(data_dir[hparams.task.lower()], config=hparams, mode='train', topk=-1) if train else eval_ds
    
    if train:
        hparams.M = 10000
        trainer = MultimodalTrainer(
            config=hparams,
            train_set=train_ds,
            val_set=eval_ds
        )
        _ = trainer.model.exact_train_features(trainer.train_loader)

        # save demonstrations constructed from diverse train samples
        from torch.utils.data import DataLoader
        train_ds = Dataset(data_dir[hparams.task.lower()], config=hparams, mode='train', topk=-1, diverse=True)
        diverse_train_data = f'hice_{hparams.task}_{hparams.model_name}_embeddings.pth'
        if not os.path.exists(diverse_train_data):
            sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.gpu_used_id[0]}')
            sentences = []
            for i, train_data in enumerate(train_ds):
                if i > 0 and i % 1000 == 0:
                    print('Done {}th sample'.format(i))
                new_fact = train_data['prompt'] + ' ' + train_data['target']
                target_new = train_data['target']
                paraphrases = train_data['rephrase_prompt']
                sentences.append(f"New Fact: {new_fact}\nPrompt: {new_fact}\n\n")
                sentences.append(f"New Fact: {new_fact}\nPrompt: {paraphrases} {target_new}\n\n")
            embeddings = sentence_model.encode(sentences)
            torch.save({'sentences': sentences, 'embeddings': embeddings}, diverse_train_data)

        trainer.model.train_classifier(trainer.train_loader, trainer.val_loader)
    else:
        # test like ike
        hparams = IKEMultimodalHyperParams.from_hparams('hparams/TRAINING/HICE/{}_test.yaml'.format(model))
        hparams.alg_name = "HICE"
        hparams.model_parallel = True # if model parallel manually
        hparams.gpu_used_id = [] if hparams.model_name == 'blip2' else []
        hparams.gpu_split = [] if hparams.model_name == 'blip2' else []
        hparams.k = 16
        hparams.diverse = 5 # diverse%
        hparams.M = 10000 # dimension of projected feature
        hparams.threshold = 0.81 if hparams.task_name.lower() == 'vqa' else 0.9 # threshold of fc, vqa: 0.81, coomprehendedit:0.9
        editor = MultimodalEditor.from_hparams(hparams)
    
        metrics, edited_model, _ = editor.edit_dataset_hice(
            ds=eval_ds,
            train_ds=eval_ds, # doesn't matter
            keep_original_weight=True
        )

def print_res(file, tok, daxiaoxie = False):
    for i in range(1, 5):
        res = torch.load(file)
        res = print_result_k(res, tok, daxiaoxie, i)
        # res = torch.load(file)
        # res = print_result_k(res, tok, daxiaoxie, i, ori=True)


if __name__ == "__main__":
    # Generate_Embedding_for_IKE(model='blip2') # running the HICE first!, so it can construct 5% dataset
    # test_IKE(model='blip2')
    # train_FT(model='minigpt4', train=False)
    train_HICE(model='blip2', train=True)
    # train_SERAC(model='blip2', train=True)
    # train_MEND(model='blip2', train=True)