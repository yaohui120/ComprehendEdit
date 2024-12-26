"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from .processor.base_dataset import BaseDataset
from .processor.blip_processors import BlipImageEvalProcessor
from ..trainer.utils import dict_to
from PIL import Image
import random
import typing
import torch
import transformers
import json
import pdb

class selectDataset(BaseDataset):
    def __init__(self, dataset: str, size:  typing.Optional[int] = None, config=None, *args, **kwargs):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        
        self.dataset = dataset
        results = []
        self.prompt = "Question: {} Short answer: "
        if dataset.lower() == 'gqa':
            dataset_dir = '/home/myh/Datasets/GQA'
            annotations_path = os.path.join(dataset_dir, 'val_balanced_questions.json')
            image_dir = os.path.join(dataset_dir, 'images')
            annotations = [json.loads(q) for q in open(annotations_path, "r")][0]
            keys = list(annotations.keys())

            for key in keys:
                sample = annotations[key]
                results.append({'image': sample['imageId']+'.jpg', # 2406500.jpg
                    'question': sample['question'],
                    'answer': sample['answer'],
                    'full answer': sample['fullAnswer'],
                    'type': sample['types']})
        elif dataset.lower() == 'tallyqa':
            # TallyQA
            # https://github.com/manoja328/TallyQA_dataset
            self.prompt = "Question: {} Answer with a number. Short answer: "
            dataset_dir = '/home/myh/Datasets/TallyQA' # 38589, all images are from VG_100K
            annotations_path = os.path.join(dataset_dir, 'test.json')
            image_dir = dataset_dir
            annotations = [json.loads(q) for q in open(annotations_path, "r")][0]
            for i in range(len(annotations)):
                sample = annotations[i]
                results.append({'image': sample['image'], # VG_100K_2/1.jpg
                    'question': sample['question'],
                    'answer': str(sample['answer']),
                    'issimple': sample['issimple'],
                    'data_source': sample['data_source']})
        elif dataset.lower() == 'vsr':
            # VSR
            # https://github.com/cambridgeltl/visual-spatial-reasoning
            self.prompt = "Question: Is this description true or false? Description: {} Short answer: "
            dataset_dir = '/home/myh/Datasets/VSR'
            annotations_path = os.path.join(dataset_dir, 'data_files/all_vsr_validated_data.jsonl') # 10972
            image_dir = os.path.join(dataset_dir, 'images')
            annotations = [json.loads(q) for q in open(annotations_path, "r")]
            for i in range(len(annotations)):
                sample = annotations[i]
                results.append({'image': sample['image'], # VG_100K_2/1.jpg
                    'question': sample['caption'], # discriminate this discription true or false
                    'answer': 'true' if sample['label'] else 'false', # 1:true, 0:false
                    'relation': sample['relation']})
        elif dataset.lower() == 'textvqa':
            import numpy as np
            dataset_dir = '/home/myh/LLaVA-main/playground/data/eval/textvqa'
            train_data_dir = 'TextVQA_0.5.1_train.json' 
            val_data_dir = 'TextVQA_0.5.1_val.json' # 5000
            train_annotations_path = os.path.join(dataset_dir, train_data_dir) # 34602
            val_annotations_path = os.path.join(dataset_dir, val_data_dir) # 5000
            image_dir = os.path.join(dataset_dir, 'train_images')
            train_annotations = [json.loads(q) for q in open(train_annotations_path, "r")][0]['data']
            val_annotations = [json.loads(q) for q in open(val_annotations_path, "r")][0]['data']
            annotations = train_annotations+val_annotations
            results = []
            for i in range(len(annotations)):
                sample = annotations[i]

                max_num, true_answer = 0, ''
                for ans in np.unique(sample['answers']):
                    tt = sample['answers'].count(ans)
                    if tt > max_num:
                        max_num = tt
                        true_answer = ans
                sample['all_answer'] = sample['answers']
                sample['answers'] = true_answer
                
                results.append({'image': sample['image_id']+'.jpg',
                    'question': sample['question'],
                    'answer': sample['answers'],
                    'all_answer': sample['all_answer']})
        elif dataset.lower() == 'mathvista':
            import pandas as pd
            from pandas import read_parquet
            dataset_dir = '/home/myh/Datasets/MathVista/'
            val_data_dir = 'data/testmini-00000-of-00001-725687bf7a18d64b.parquet' 
            val_annotations_path = os.path.join(dataset_dir, val_data_dir)
            image_dir = os.path.join(dataset_dir)
            val_annotations = read_parquet(val_annotations_path)
            annotations = val_annotations
            results = []
            for i in range(len(annotations)):
                sample = annotations.loc[i]

                if sample['question_type'] == 'multi_choice':# free_form, multi_choice
                    choices = ''
                    for choice in sample['choices']:
                        choices += (choice+', ')
                    choices = choices[:-2]
                    sample['question'] += ('Choose one answer from following choices:{}.'.format(choices))
                
                results.append({'image': sample['image'],
                    'question': sample['question'],
                    'answer': sample['answer']})
            # pdb.set_trace()
        elif dataset.lower() == 'okvqa':
            dataset_dir = '/home/myh/Datasets/MMEdit/'
            train_data_dir = 'vqa_train.json'
            val_data_dir = 'vqa_eval.json'
            train_annotations_path = os.path.join(dataset_dir, train_data_dir) # 6346
            val_annotations_path = os.path.join(dataset_dir, val_data_dir) # 2093
            image_dir = os.path.join(dataset_dir)
            train_annotations = [json.loads(q) for q in open(train_annotations_path, "r")][0]
            val_annotations = [json.loads(q) for q in open(val_annotations_path, "r")][0]
            annotations = train_annotations+val_annotations
            
            # import numpy as np
            # train = "/home/myh/Datasets/editing-data/multimodal_locality/OK-VQA dataset/okvqa_loc.json"
            # image_dir = '/home/myh/Datasets/MMEdit'
            # annotations = [json.loads(q) for q in open(os.path.expanduser(train), "r")][0]
            # for i in range(len(annotations)):
            #     sample = annotations[i]
                
            #     max_num, true_answer = 0, ''
            #     for ans in np.unique(sample['answer']):
            #         tt = sample['answer'].count(ans)
            #         if tt > max_num:
            #             max_num = tt
            #             true_answer = ans
            #     sample['all_answer'] = sample['answer']
            #     sample['answer'] = true_answer
                
            #     sample['m_loc'] = sample["image"]
            #     sample['m_loc_q'] = sample['question']
            #     sample['m_loc_a'] = sample['answer']

            results = []
            for i in range(len(annotations)):
                sample = annotations[i]
                results.append({'image': sample['m_loc'],
                    'question': sample['m_loc_q'],
                    'answer': sample['m_loc_a']})
        elif dataset.lower() == 'nq':
            dataset_dir = '/home/myh/Datasets/MMEdit/'
            train_data_dir = 'vqa_train.json'
            val_data_dir = 'vqa_eval.json'
            train_annotations_path = os.path.join(dataset_dir, train_data_dir) #
            val_annotations_path = os.path.join(dataset_dir, val_data_dir) #
            image_dir = os.path.join(dataset_dir)
            train_annotations = [json.loads(q) for q in open(train_annotations_path, "r")][0]
            val_annotations = [json.loads(q) for q in open(val_annotations_path, "r")][0]
            annotations = train_annotations+val_annotations
            
            # nq_path = "/home/myh/Datasets/NQ-Dataset/NQ-open.train.jsonl"
            # nq_qa_ori = [json.loads(q) for q in open(os.path.expanduser(nq_path), "r")]
            # annotations1 = [qa for qa in nq_qa_ori if len(qa['answer']) == 1] # 79300, questions with only one answer
            # for i in range(len(annotations)):
            #     sample = annotations[i]
            #     sample['loc'] = sample['question']
            #     sample['loc_ans'] = sample['answer'][0]

            results = []
            for i in range(len(annotations)):
                sample = annotations[i]
                results.append({
                    'question': sample['loc'],
                    'answer': sample['loc_ans']})
        elif dataset.lower() == 'vqa': 
            # 原始vqa数据集去掉can we构造的vqa后的数据集
            dataset_dir = '/home/myh/EasyEdit-main/data_our/vqa_except_can_we_data.json'
            image_dir = '/home/myh/Datasets/MMEdit'
            annotations = [json.loads(q) for q in open(dataset_dir, "r")][0]
            results = []
            for i in range(len(annotations)):
                sample = annotations[i]
                results.append({
                    'question': sample['src'],
                    'image': sample['image'],
                    'answer': sample['alt']})
        
        # get tokenizer and vis_processor
        vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
        if (config is not None and hasattr(config, 'tokenizer_name')):
            tok_name = (
                config.tokenizer_name
                if config.tokenizer_name is not None
                else config.name
            )
            tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
                tok_name, trust_remote_code=True
            )            
            if tokenizer.pad_token == None or tokenizer.pad_token == '':
                tokenizer.pad_token = tokenizer.eos_token  
                
        super().__init__(vis_processor)
        
        for i in range(len(results)):
            self.annotation.append({'src':results[i]['question'],
                            'alt':results[i]['answer']})
            if 'image' in results[i].keys():
                self.annotation[-1]['image'] = os.path.join(image_dir, results[i]['image'])
            
        # pdb.set_trace()
        self.config = config
        self.tok = tokenizer
        self.max_length = 32

        data = []
        if size is not None:
            self.annotation = self.annotation[:size]
        for i, record in enumerate(self.annotation):
            
            if record['alt'] == "":
                continue
                    
            item = {
                'prompt': record['src'],
                'target': record['alt'],
            }
            
            if 'image' in record.keys():
                image_path = os.path.join(record["image"])
                item['image'] = image_path

            data.append(item)
            
        if size is not None:
            data = data[:size]        
        self._data = data

    def __getitem__(self, index):
        return self._data[index]
    
    def __len__(self):
        return len(self._data)

    def collate_fn(self, batch):
        # edit_inner：单条编辑，图像，描述，标签
        self.config.device = 0
        src = [b['prompt'] for b in batch]
        trg = [b['target'] for b in batch]
        
        # edit_inner
        edit_inner = {}
        if 'image' in batch[0].keys():
            image = [b['image'] for b in batch]
            image = [self.vis_processor(Image.open(image_path).convert("RGB")) for image_path in image]
            edit_inner['image'] = torch.stack(image, dim=0)
        else: # for NQ and OK-VQA
            edit_inner['image'] = None
        edit_inner['text_input'] = [self.prompt.format(s) + t for s, t in zip(src, trg)] #[self.prompt.format(s) for s in src]
        edit_inner['labels'] = trg
        edit_inner['text_labels'] = trg
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            edit_inner['prompts_len'] = [len(self.tok.encode(self.prompt.format(s), add_special_tokens=False)) for s in src]
            edit_inner['labels'] = self.tok.encode(trg, add_special_tokens=False, return_tensors="pt",)
        else:
            edit_inner['prompts_len'] = [len(self.tok.encode(self.prompt.format(s))) for s in src]
            edit_inner['labels'] = self.tok.encode(trg, return_tensors="pt",)
        edit_inner['image_path'] = [b['image'] for b in batch]
        edit_inner['ori_qs'] = src
        
        batch = {
            "edit_inner": edit_inner,
        }
        # pdb.set_trace()
        return dict_to(batch, self.config.device)
