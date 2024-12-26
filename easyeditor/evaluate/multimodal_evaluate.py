from ..models.melo.melo import LORA

import typing
from itertools import chain
from typing import List, Optional

import numpy as np
import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from ..util import HyperParams
from .evaluate_utils import (
    test_seq2seq_batch_prediction_acc,
    test_batch_prediction_acc,
    test_prediction_acc,
    test_generation_quality,
    test_concept_gen,
    test_safety_gen,
    test_instance_change,
    PPL,
    kl_loc_loss,
    es,
    es_per_icl,
    per_generation,
    F1
)



def compute_icl_multimodal_edit_quality(
        model,
        model_name,
        hparams: HyperParams,
        tok: AutoTokenizer,
        # vis_tok,
        icl_examples,
        record: typing.Dict,
        device,
        pre_edit: bool = False
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :param snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    vis_root = hparams.coco_image
    rephrase_root = hparams.rephrase_image
    # First, unpack rewrite evaluation record.
    target = record["target"]
    prompt = record["prompt"]
    image = record["image"] if record["image"].is_cuda else record["image"].to(hparams.device)
    rephrase = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    rephrase_image = record["image_rephrase"] if 'image_rephrase' in record.keys() else None
    if rephrase_image is not None:
        rephrase_image = rephrase_image if rephrase_image.is_cuda else rephrase_image.to(hparams.device)

    if "locality_prompt" in record.keys():
        loc_q = record["locality_prompt"]
        loc_a = record["locality_ground_truth"]
    if "multimodal_locality_image" in record.keys():
        m_loc_image = record["multimodal_locality_image"] if record["multimodal_locality_image"].is_cuda else record["multimodal_locality_image"].to(hparams.device)
        m_loc_q = record["multimodal_locality_prompt"]
        m_loc_a = record["multimodal_locality_ground_truth"]

    new_fact = f'New Fact: {prompt} {target}\nPrompt: {prompt}'

    if pre_edit:
        edit_acc, _, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                             target, prompt, image)
    else:
        edit_acc, _, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                             target, new_fact, image)
    ret = {
        f"rewrite_acc": edit_acc
    }
    if rephrase is not None:
        rephrase_acc, _, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                                 target, f'New Fact: {prompt} {target}\nPrompt: {rephrase}', image)
        ret['rephrase_acc'] = rephrase_acc

    if "image_rephrase" in record.keys():
        rephrase_image_acc, _, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                                       target, new_fact, rephrase_image)
        ret['rephrase_image_acc'] = rephrase_image_acc

    if "locality_prompt" in record.keys():
        if pre_edit:
            _, _, locality_output = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                                           loc_a, loc_q, None, is_loc=True)
        else:
            _, _, locality_output = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                                           loc_a, f'New Fact: {prompt} {target}\nPrompt: {loc_q}', None, is_loc=True)
        ret['locality_output'] = locality_output

    if "multimodal_locality_image" in record.keys():
        if pre_edit:
            _, _, locality_image_output = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                                                 m_loc_a, m_loc_q, m_loc_image, is_loc=True)
        else:
            _, _, locality_image_output = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                                                 m_loc_a, f'New Fact: {prompt} {target}\nPrompt: {m_loc_q}', m_loc_image, is_loc=True)
        ret['multimodal_locality_output'] = locality_image_output

    return ret

def icl_multimodal_lm_eval(
        model,
        model_name,
        hparams: HyperParams,
        tokenizer,
        icl_examples,
        target,
        x,
        image,
        is_loc=False,
        neighborhood=False,
        return_targ=False)-> typing.Dict:
    device = torch.device(f'cuda:{hparams.device}')

    samples = prepare_multimodal_edit(hparams, tokenizer, target, [''.join(icl_examples) + f'{x}'], image)

    return compute_multimodal_edit_quality(model, samples,
                                           hparams.exact_match, return_targ=return_targ)
    #if not is_loc else compute_multimodal_edit_quality_demo(model, samples)


def prepare_multimodal_edit(hparams,
                            tok,
                            target,
                            prompts,
                            image):
    if isinstance(target, str):
        target = [target, ]
    if isinstance(prompts, str):
        prompts = [prompts, ]
    if image is not None and len(image.shape) == 3:
        image = image.unsqueeze(0)
    text_input = [prompt_ + ' ' + target_ for prompt_, target_ in zip(prompts, target)]

    if hparams.model_name == 'minigpt4':
        prompts_len = [len(tok.encode(prompt, add_special_tokens=False)) for prompt in prompts]
        target = tok(target, add_special_tokens=False, return_tensors="pt", )["input_ids"]
    else:
        prompts_len = [len(tok.encode(prompt, add_special_tokens=False)) for prompt in prompts]
        target = tok([' ' + target_ if target_[0] != ' ' else target_ for target_ in target], add_special_tokens=False,
                     return_tensors="pt", )["input_ids"]

    ret = {
        'text_input': text_input,
        'image': image,
        'labels': target,
        'prompts_len': prompts_len
    }
    return ret


def compute_multimodal_edit_quality(model, batch, exach_match=False, return_targ=False):
    with torch.no_grad():
        outputs = model(batch)
        if isinstance(outputs, torch.Tensor):
            logits = outputs.detach().cpu()
            targ = batch["labels"].cpu()
        else:
            logits = outputs.logits.detach().cpu()
            targ = outputs.labels.detach().cpu()

    if logits.dim() == 3:
        logits = logits[:, :-1]
        targ = targ[:, 1:]
        # logits = logits[:, -targ.shape[1]:]
    logits_ = logits.clone()
    mask = targ != -100
    targ[~mask] = 0
    if exach_match:
        pred_ids = logits.argmax(-1).masked_fill(~mask, 0)
        correct = pred_ids == targ
        if logits.dim() == 3:
            correct = (pred_ids == targ).all(-1)  # We aim for an exact match across the entire sequence
        acc = correct.float().mean()
    else:
        pred_ids = logits.argmax(-1).masked_fill(~mask, 0).detach().cpu()
        correct = pred_ids == targ
        correct = correct & mask
        num_non_padding = mask.sum().float().item()
        acc = correct.sum() / num_non_padding

    if return_targ:
        return acc, pred_ids.numpy(), targ.numpy()
    else:
        return acc, pred_ids.numpy(), logits_


def compute_multimodal_edit_quality_demo(model, batch):
    with torch.no_grad():
        outputs = model(batch)
        if isinstance(outputs, torch.Tensor):
            logits = outputs.detach().cpu()
        else:
            logits = outputs.logits.detach().cpu()
            # targ = outputs.labels.detach().cpu()
        targ = batch["labels"].cpu()
    logits_ = logits.clone()
    if logits.dim() == 3:
        logits = logits[:, :-1]
        # targ = targ[:, 1:]
        logits = logits[:, -targ.shape[1]:]
    mask = targ != -100
    targ[~mask] = 0
    pred_ids = logits.argmax(-1).masked_fill(~mask, 0).detach().cpu()
    correct = pred_ids == targ
    correct = correct & mask
    num_non_padding = mask.sum().float().item()
    acc = correct.sum() / num_non_padding

    return acc, pred_ids.numpy(), logits_


def compute_multimodal_edit_results(
        model,
        model_name,
        hparams: HyperParams,
        tok: AutoTokenizer,
        record: typing.Dict,
        device
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    ret = {}
    # First, unpack rewrite evaluation record.

    target = record["target"]
    rewrite_prompts = record["prompt"]
    image = record["image"] if record["image"].is_cuda else record["image"].to(hparams.device)

    edit_inner = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, image)
    ret['rewrite_acc'], _ = compute_multimodal_edit_quality(model, edit_inner)

    if "rephrase_prompt" in record.keys():
        rephrase_prompts = record["rephrase_prompt"]
        edit_outer = prepare_multimodal_edit(hparams, tok, target, rephrase_prompts, image)
        ret['rephrase_acc'], _ = compute_multimodal_edit_quality(model, edit_outer)

    if "image_rephrase" in record.keys():
        rephrase_image = record["image_rephrase"]
        rephrase_image = rephrase_image if rephrase_image.is_cuda else rephrase_image.to(hparams.device)
        edit_image_outer = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, rephrase_image)
        ret['image_rephrase_acc'], _ = compute_multimodal_edit_quality(model, edit_image_outer)

    if 'locality_prompt' in record.keys():
        locality_prompt = record["locality_prompt"]
        locality_ground_truth = record["locality_ground_truth"]
        locality = prepare_multimodal_edit(hparams, tok, locality_ground_truth, locality_prompt, None)
        _, ret['locality_output'] = compute_multimodal_edit_quality(model, locality)

    if 'multimodal_locality_prompt' in record.keys():
        m_loc_prompt = record["multimodal_locality_prompt"]
        m_loc_ground_truth = record["multimodal_locality_ground_truth"]
        m_loc_image = record["multimodal_locality_image"]
        m_loc_image = m_loc_image if m_loc_image.is_cuda else m_loc_image.to(hparams.device)
        m_locality = prepare_multimodal_edit(hparams, tok, m_loc_ground_truth, m_loc_prompt, m_loc_image)
        _, ret['multimodal_locality_output'] = compute_multimodal_edit_quality(model, m_locality)
    # Form a list of lists of prefixes to test.

    return ret


def compute_multimodal_edit_results_demo(
        model,
        model_name,
        hparams: HyperParams,
        tok: AutoTokenizer,
        record: typing.Dict,
        device
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    ret = {}
    # First, unpack rewrite evaluation record.

    target = record["target"]
    rewrite_prompts = record["prompt"]
    image = record["image"] if record["image"].is_cuda else record["image"].to(hparams.device)

    edit_inner = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, image)
    ret['rewrite_acc'], _, logits = compute_multimodal_edit_quality_demo(model, edit_inner)

    if "rephrase_prompt" in record.keys():
        rephrase_prompts = record["rephrase_prompt"]
        edit_outer = prepare_multimodal_edit(hparams, tok, target, rephrase_prompts, image)
        ret['rephrase_acc'], _ = compute_multimodal_edit_quality(model, edit_outer)

    if "image_rephrase" in record.keys():
        rephrase_image = record["image_rephrase"]
        rephrase_image = rephrase_image if rephrase_image.is_cuda else rephrase_image.to(hparams.device)
        edit_image_outer = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, rephrase_image)
        ret['image_rephrase_acc'], _ = compute_multimodal_edit_quality(model, edit_image_outer)

    if 'locality_prompt' in record.keys():
        locality_prompt = record["locality_prompt"]
        locality_ground_truth = record["locality_ground_truth"]
        locality = prepare_multimodal_edit(hparams, tok, locality_ground_truth, locality_prompt, None)
        _, ret['locality_output'] = compute_multimodal_edit_quality(model, locality)

    if 'multimodal_locality_prompt' in record.keys():
        m_loc_prompt = record["multimodal_locality_prompt"]
        m_loc_ground_truth = record["multimodal_locality_ground_truth"]
        m_loc_image = record["multimodal_locality_image"]
        m_loc_image = m_loc_image if m_loc_image.is_cuda else m_loc_image.to(hparams.device)
        m_locality = prepare_multimodal_edit(hparams, tok, m_loc_ground_truth, m_loc_prompt, m_loc_image)
        _, ret['multimodal_locality_output'] = compute_multimodal_edit_quality(model, m_locality)
    # Form a list of lists of prefixes to test.

    return ret, logits

    prompt_tok = tok(
        prompt,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    trg_tok = tok(
        target,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    prompt_tok['labels'] = trg_tok['input_ids']
    # prompt_tok['decoder_attention_mask'] = trg_tok['attention_mask']

    with torch.no_grad():
        outputs = model(**prompt_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits

        assert logits.size(1) == trg_tok['input_ids'].size(1)
        ans = torch.argmax(logits, dim=-1)
        if locality:
            return ans.squeeze().detach().cpu().numpy().tolist()

        return \
        torch.mean((trg_tok['input_ids'][:, :-1] == ans[:, :-1]).float(), dim=-1).detach().cpu().numpy().tolist()[0]

def get_txt_embedding(input, sentence_model, device):
    # return tensor, dim: [dim], such as shape=[384]
    with torch.no_grad():
        res = torch.tensor(sentence_model.encode(input, show_progress_bar=False, device=device))
    return res.to(device)

def our_predict_domain(prompt, target, sentence_model, Wo, W_rand, device, mloc_memory=None, threshold=0.81):
    new_fact = prompt + ' ' + target
    input = f"New Fact: {new_fact}\nPrompt: {new_fact}\n\n"
    # query_embedding = torch.tensor(sentence_model.encode(input, show_progress_bar=False, device=device)).unsqueeze(0).to(device)
    query_embedding = get_txt_embedding(input, sentence_model, device)
    query_fea = query_embedding / torch.norm(query_embedding, p=2, dim=-1)
    query_fea_h = query_fea
    query_fea_h = torch.nn.functional.relu(query_fea @ W_rand)
    logits = (query_fea_h @ Wo.T)
    preds = torch.max(logits, dim=-1)[1]
    return logits, preds

from sentence_transformers import util
def select_icl_samples(prompt, target, stored_data, sentence_model, device, hparams, query_embedding=None):
    new_fact = prompt + ' ' + target
    # query_sentence = f"New Fact: {new_fact}\nPrompt: {request['prompt']}\n\n"
    query_sentence = f"New Fact: {new_fact}\nPrompt: {new_fact}\n\n"
    if query_embedding is not None:
        query_embedding = query_embedding.to(device)
    else:
        query_embedding = torch.tensor(sentence_model.encode(query_sentence, show_progress_bar=False, device=device))
        query_embedding = query_embedding.unsqueeze(0).to(device)
        query_embedding = query_embedding / torch.norm(query_embedding, p=2, dim=-1)

    stored_sentences = stored_data['sentences']
    stored_embeddings = stored_data['embeddings']
    stored_embeddings = torch.tensor(stored_embeddings).to(device)
    # stored_embeddings = util.normalize_embeddings(stored_embeddings)
    stored_embeddings = stored_embeddings / torch.norm(stored_embeddings, p=2, dim=-1).unsqueeze(-1)
    
    hits = util.semantic_search(query_embedding, stored_embeddings, score_function=util.dot_score, top_k=hparams.k)
    assert len(hits) == 1
    hit = hits[0]
    icl_examples = [stored_sentences[hit[k]["corpus_id"]] for k in range(len(hit))]
    # print(hit[0]['score'])
    # too long will cause RuntimeError
    if len(''.join(icl_examples)) > 2500:
        while (len(''.join(icl_examples)) > 2500):
            del icl_examples[-1]
    return icl_examples

def compute_hice_multimodal_edit_quality(
        model,
        model_name,
        hparams: HyperParams,
        tok: AutoTokenizer,
        # vis_tok,
        icl_examples,
        record: typing.Dict,
        device,
        pre_edit: bool = False,
        Wo: torch.Tensor = None,
        W_rand: torch.Tensor = None,
        sentence_model = None,
        **kwargs
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :param snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    vis_root = hparams.coco_image
    rephrase_root = hparams.rephrase_image
    # First, unpack rewrite evaluation record.
    target = record["target"]
    prompt = record["prompt"]
    image = record["image"] if record["image"].is_cuda else record["image"].to(hparams.device)
    rephrase = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    rephrase_image = record["image_rephrase"] if 'image_rephrase' in record.keys() else None
    if not pre_edit:
        stored_data = kwargs["stored_data"]
        test_input_embeddings = kwargs["test_input_embeddings"] if 'test_input_embeddings' in kwargs.keys() else None
    
    Wo, W_rand = Wo, W_rand
    sentence_model = sentence_model
    sentence_model1 = kwargs['sentence_model1'] if 'sentence_model1' in kwargs.keys() else sentence_model
    mloc_memory = kwargs['mloc_memory'] if 'mloc_memory' in kwargs.keys() else None
    threshold = hparams.threshold
    
    if rephrase_image is not None:
        rephrase_image = rephrase_image if rephrase_image.is_cuda else rephrase_image.to(hparams.device)

    if "locality_prompt" in record.keys():
        loc_q = record["locality_prompt"]
        loc_a = record["locality_ground_truth"]
    if "multimodal_locality_image" in record.keys():
        m_loc_image = record["multimodal_locality_image"] if record["multimodal_locality_image"].is_cuda else record["multimodal_locality_image"].to(hparams.device)
        m_loc_q = record["multimodal_locality_prompt"]
        m_loc_a = record["multimodal_locality_ground_truth"]

    new_fact = f'New Fact: {prompt} {target}\nPrompt: {prompt}'
    ret = {}
    
    if pre_edit:
        edit_acc, _, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                             target, prompt, image)
    else:
        _, edit_preds = our_predict_domain(prompt, target, sentence_model1, Wo, W_rand, device, mloc_memory, threshold)
        if edit_preds == 0:
            edit_acc, e_preds, e_logits = icl_multimodal_lm_eval(model, model_name, hparams, tok, [''],
                                             target, prompt, image)
        else:
            edit_acc, e_preds, e_logits = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                             target, f'New Fact: {prompt} {target}\nPrompt: {prompt}', image)
        ret['rewrite_fc_acc'] = 0 if edit_preds == 0 else 1
    ret["rewrite_acc"] = edit_acc
    
    if not pre_edit: # 编辑后的模型才测泛化性
        if rephrase is not None:
            _, rephrase_preds = our_predict_domain(rephrase, target, sentence_model1, Wo, W_rand, device, mloc_memory, threshold)
            if rephrase_preds == 0:
                rephrase_acc, r_preds, r_logits = icl_multimodal_lm_eval(model, model_name, hparams, tok, [''],
                                                    target, rephrase, image)
            else:
                icl_examples = select_icl_samples(rephrase, target, stored_data, sentence_model, device, hparams)
                rephrase_acc, r_preds, r_logits = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                                    target, f'New Fact: {prompt} {target}\nPrompt: {rephrase}', image)
            ret['rephrase_fc_acc'] = 0 if rephrase_preds == 0 else 1
            ret['rephrase_acc'] = rephrase_acc

        # if "image_rephrase" in record.keys():
        #     rephrase_image_acc, _, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
        #                                                 target, new_fact, rephrase_image)
        #     ret['rephrase_image_acc'] = rephrase_image_acc

    if "locality_prompt" in record.keys():
        if pre_edit:
            _, l_preds, locality_output = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                                           loc_a, loc_q, None, is_loc=True)
        else:
            _, locality_preds = our_predict_domain(loc_q, loc_a, sentence_model1, Wo, W_rand, device, mloc_memory, threshold)
            if locality_preds == 0:
                _, l_preds, locality_output = icl_multimodal_lm_eval(model, model_name, hparams, tok, [''],
                                                           loc_a, loc_q, None, is_loc=True)
            else:
                icl_examples = select_icl_samples(loc_q, loc_a, stored_data, sentence_model, device, hparams)
                _, l_preds, locality_output = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                                           loc_a, f'New Fact: {prompt} {target}\nPrompt: {loc_q}', None, is_loc=True)
            ret['locality_fc_acc'] = 1 if locality_preds == 0 else 0
        ret['locality_output'] = locality_output        

    if "multimodal_locality_image" in record.keys():
        if pre_edit:
            _, ml_preds, locality_image_output = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                                                 m_loc_a, m_loc_q, m_loc_image, is_loc=True)
        else:
            _, multimodal_locality_preds = our_predict_domain(m_loc_q, m_loc_a, sentence_model1, Wo, W_rand, device, mloc_memory, threshold)
            if multimodal_locality_preds == 0:
                _, _ml_preds, locality_image_output = icl_multimodal_lm_eval(model, model_name, hparams, tok, [''],
                                                                 m_loc_a, m_loc_q, m_loc_image, is_loc=True)
            else:
                icl_examples = select_icl_samples(loc_q, loc_a, stored_data, sentence_model, device, hparams)
                _, _ml_preds, locality_image_output = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                                                 m_loc_a, f'New Fact: {prompt} {target}\nPrompt: {m_loc_q}', m_loc_image, is_loc=True)
            ret['multimodal_locality_fc_acc'] = 1 if multimodal_locality_preds == 0 else 0
        ret['multimodal_locality_output'] = locality_image_output

    return ret
