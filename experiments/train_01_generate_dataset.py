from ast import Num
import os
import sys
import json
import hashlib
from pdb import set_trace
from numpy import isin
from sympy import elliptic_f
from torch import Value
from tqdm import tqdm
from utils import DATA_ROOT, dump_json, dump_jsonl, load_jsonl_iteratively, load_config

NUM_1B = 1e+9

def dict_content_hash(d):
    new_d = {}
    for k, v in d.items():
        assert isinstance(v, dict), f"Value of {k} should be a dict"
        filtered = {_k: _v  for _k, _v in v.items() if _v != 0}
        if len(filtered) > 0:
            new_d[k] = filtered
    dict_str = json.dumps(new_d, sort_keys=True)
    return new_d, hashlib.md5(dict_str.encode()).hexdigest()

def write_text(src, tgt, b_tokens, tokenizer_type, lang=None, record_ids=False, docid_fn=None, filter_docids=None):
    """
    Write text to tgt file from src file, with a limit of b_tokens.
    - `docid_fn`: if record_ids is True, the docid_fn will be used to write the doc ids
    - `filter_docids`: if record_ids is False, the docids will be used to filter the items
    """
    all_cnts = 0
    if record_ids:
        assert filter_docids is None and lang is not None, "docids shouldn't and lang should be provided if record_ids is True"
        docid_fp = open(docid_fn, 'a', encoding="utf8") if record_ids else None
        print(f"Writing doc ids to {docid_fn}")
    
    if filter_docids is not None:
        assert isinstance(filter_docids, set), "docids should be a set"
        assert record_ids is False, "docids shouldn't be provided if record_ids is True"
        

    with open(tgt, 'a', encoding="utf8") as fp:
        num_tokens = b_tokens * NUM_1B
        log_rate = 0.05
        while all_cnts < num_tokens:
            desc = f"DATA_ROOT{src[len(DATA_ROOT):]} → DATA_ROOT{tgt[len(DATA_ROOT):]} ({b_tokens}B tokens)"
            for item in tqdm(load_jsonl_iteratively(src), desc=desc):
                if all_cnts / num_tokens > log_rate:
                    print(f"\n{all_cnts}/{num_tokens} ({all_cnts/num_tokens:.4f}) tokens is written")
                    log_rate += 0.05

                if f'{tokenizer_type}' not in item:
                    raise NotImplementedError(f"Please run `python3 tokenization` first to get tokens for each data item for file {src}, with tokenizer {tokenizer_type}")
                if record_ids:
                    docid_fp.write(f"{json.dumps({'docid': item['docid'], 'lang': lang}, ensure_ascii=False)}\n")
                if filter_docids is not None:
                    if item['docid'] not in filter_docids:
                        continue

                string = json.dumps({'text': item['text']}, ensure_ascii=False)
                fp.write(f"{string}\n")
                
                cnt = item[f'{tokenizer_type}']['num_tokens'] 
                all_cnts += cnt
                if all_cnts >= num_tokens:
                    break    
    print(f"Finished writing {all_cnts} tokens to {tgt} from {src}")

def only_allow_monolingual_kg_05B_training(btoken_kg):
    """
    It is because of the shit-code when excluding AdaXEval docids for creating cross-lingual transfer dataset
    Currently, the cross-lingual transfer needs to decide what docids to include by:
    1. Read the full docids that used in kg dataset (knowledge injection dataset)
    2. Exclude the docids that are not in AdaXEval dataset
    """
    all_tokens = []
    for lang, b_tokens in btoken_kg.items():
        if isinstance(b_tokens, dict):
            if 'num_tokens' in b_tokens:
                all_tokens.append(b_tokens['num_tokens'])
            else:
                raise ValueError(f"btoken_kg should be a dict with num_tokens, but get {b_tokens}")
        elif isinstance(b_tokens, (int, float)):
            all_tokens.append(b_tokens)
        else:
            raise ValueError(f"b_tokens should be a dict with num_tokens or a float, but get {b_tokens}")
        
        assert len(set(all_tokens)) <= 2 and sum(all_tokens) == max(all_tokens), "Only allow training with knowldege dataset with 0.5B tokens"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str)
    parser.add_argument("--kg_only", action="store_true", help="Only generate knowledge dataset")
    parser.add_argument("--ct_only", action="store_true", help="Only generate cross-lingual transfer dataset")
    parser.add_argument("--update_config", action="store_true")
    args = parser.parse_args()

    if not args.kg_only and not args.ct_only:
        collect_kg = True
        collect_ct = True
    elif args.kg_only:
        collect_kg = True
        collect_ct = False
    elif args.ct_only:
        collect_kg = False
        collect_ct = True
    if args.kg_only and args.ct_only:
        raise ValueError("Both kg_only and ct_only cannot be True at the same time")
    
    config = load_config(args.config_name)
    assert args.config_name == config['name'], f"config_name should be the same as the config file name, but got {args.config_name} vs. {config['name']}"
    data_root = config['dataset']['dataset-dir']
    
    exp_root = os.path.join(config['exp-dir'], config['name'])
    save_dir = os.path.join(exp_root, "dataset")
    os.makedirs(save_dir, exist_ok=True)
    dump_json(config, f"{exp_root}/config.json", pretty=True)

    if args.update_config:
        sys.exit(0)

    collections = config['dataset']['collection']
    btoken_kg = collections['knowledge']
    btoken_ct = collections['crosslingual-transfer']['en-ja']
    tokenizer_type = config['dataset']['tokenizer']['type']

    if collect_kg:
        kg_data_dirs = []
        # for lang in ["en", "ja", "zh", "en_jstage"]:
        for data_type in btoken_kg.keys():
            lang = data_type.split('-')[0]
            # data_type = f"{lang}-medical"

            data_suffix = None
            data_filepath = None
            filtered_docid_fn = None
            line_clip = None # for noisy dataset only, to clip lines from the data_filepath
            if isinstance(btoken_kg[data_type], int) or isinstance(btoken_kg[data_type], float):
                repeat = 1
                if btoken_kg[data_type] == 0:
                    continue
                b_tokens = btoken_kg[data_type]
            elif isinstance(btoken_kg[data_type], dict):
                assert 'num_tokens' in btoken_kg[data_type]
                repeat = btoken_kg[data_type]['repeat'] if 'repeat' in btoken_kg[data_type] else 1
                if "doc_ids" in btoken_kg[data_type]:
                    b_tokens = btoken_kg[data_type]['num_tokens']
                    filtered_docid_fn = btoken_kg[data_type]['doc_ids']
                elif "data_filepath" in btoken_kg[data_type]:
                    assert 'data_suffix' in btoken_kg[data_type]
                    b_tokens = btoken_kg[data_type]['num_tokens']
                    data_filepath = btoken_kg[data_type]['data_filepath']
                    data_suffix = btoken_kg[data_type]['data_suffix']
                elif "data_filepaths" in btoken_kg[data_type]:
                    assert data_type == "noisy", "data_filepaths is only supported for noisy dataset"                    
                    assert 'data_suffix' in btoken_kg[data_type]
                    assert isinstance(btoken_kg[data_type]['data_filepaths'], list), "data_filepaths should be a list of filepaths"
                    data_filepaths = btoken_kg[data_type]['data_filepaths']
                    data_suffix = btoken_kg[data_type]['data_suffix']
                    b_tokens = 0.5 * len(data_filepaths)  # b_tokens for each single file for noisy is 0.5B, thus the total b_tokens is 0.5 * len(data_filepaths)
                else:
                    raise ValueError(f"btoken_kg should be a dict with num_tokens and doc_ids or data_filepath, but get {btoken_kg[data_type]}")
                            
            if data_suffix is None:
                kg_data_dir = os.path.join(config['dataset']['kg-dataset-dir'], f'{lang}-{b_tokens}')
            else:
                kg_data_dir = os.path.join(config['dataset']['kg-dataset-dir'], f'{lang}_{data_suffix}-{b_tokens}')
            os.makedirs(kg_data_dir, exist_ok=True)
        
            src_kg_fn = os.path.join(kg_data_dir, "knowledge.jsonl")
            src_docid_fn = os.path.join(kg_data_dir, "doc_ids.jsonl")
            
            if not os.path.exists(src_kg_fn) or not os.path.exists(src_docid_fn):
                print(f"Knowledge dataset not found at {kg_data_dir}, generating...")
                if lang == "en":
                    filename = "native.subset.jsonl"
                elif lang == "ja" or lang == "zh" or lang == "en_jstage":
                    filename = "medical_native.jsonl"
                elif lang == "codeswitch":
                    if b_tokens > 0:
                        assert data_filepath is not None, "data_filepath should be provided for codeswitch dataset"
                elif lang == "noisy": 
                    if "line_clip" in btoken_kg[data_type] and btoken_kg[data_type]["line_clip"] > 0:
                        assert data_filepaths is not None and len(data_filepaths) > 0, \
                            "data_filepaths should be provided for noisy dataset"
                        assert btoken_kg[data_type]["num_tokens"] == 0 and btoken_kg[data_type]["line_clip"] > 0, \
                            "For noisy dataset, only support line_clip based dataset creation"
                        line_clip = btoken_kg[data_type]["line_clip"]
                else:
                    raise NotImplementedError(f"Language {lang} not supported")

                if line_clip is not None:
                    assert data_filepaths is not None and len(data_filepaths) > 0, \
                        "data_filepaths should be provided and exist for noisy dataset"
                    if not os.path.exists(src_kg_fn):
                        print(f"Clipping first {line_clip} lines from {data_filepaths} to {src_kg_fn}")
                        write_items = []
                        for data_filepath in data_filepaths:
                            for item in load_jsonl_iteratively(data_filepath, request_num=line_clip):
                                write_items.append(item)
                        dump_jsonl(write_items, src_kg_fn)
                    else:
                        print(f"Knowledge dataset already exists at {src_kg_fn}")
                elif b_tokens > 0:
                    if data_filepath is None:
                        native_fn = os.path.join(data_root, lang, filename)
                    else:
                        native_fn = data_filepath
                        print(f"Using data_filepath {data_filepath} for {lang}")
                    print("===> Reading instructions from: ", native_fn)
                    
                    if filtered_docid_fn is None:
                        write_text(
                            native_fn, src_kg_fn, b_tokens=b_tokens, 
                            tokenizer_type=tokenizer_type, lang=lang, filter_docids=None,
                            record_ids=True, docid_fn=src_docid_fn)
                    else:
                        filtered_docids = set([item['docid'] for item in load_jsonl_iteratively(filtered_docid_fn)])
                        write_text(
                            native_fn, src_kg_fn, b_tokens=b_tokens, 
                            tokenizer_type=tokenizer_type, filter_docids=filtered_docids)
                        os.symlink(filtered_docid_fn, src_docid_fn)
                else:
                    raise ValueError(f"b_tokens should be greater than 0 or line_clip should be provided, but get b_tokens={b_tokens} and line_clip={line_clip}")
            else:
                print(f"Knowledge dataset already exists at {kg_data_dir}")

            if repeat == 1:
                kg_fn = os.path.join(save_dir, f"{lang}_knowledge.jsonl")
                if os.path.islink(kg_fn):
                    os.unlink(kg_fn)
                os.symlink(src_kg_fn, kg_fn)
                print(f"Linking {src_kg_fn} to {kg_fn}")
            else:
                for i in range(repeat):
                    kg_fn = os.path.join(save_dir, f"{lang}_knowledge_{i}.jsonl")
                    if os.path.islink(kg_fn):
                        os.unlink(kg_fn)
                    os.symlink(src_kg_fn, kg_fn)
                    print(f"Linking {src_kg_fn} to {kg_fn}")
                    

    if collect_ct:
        recipe, hash_id = dict_content_hash(collections['crosslingual-transfer'])
        if len(recipe) == 0:
            print("No cross-lingual transfer dataset to collect")
            sys.exit(0)    
        
        ct_data_dir = os.path.join(config['dataset']['ct-dataset-dir'], hash_id)
        os.makedirs(ct_data_dir, exist_ok=True)

        type_fn = os.path.join(ct_data_dir, "data_type.json")
        src_ct_fn = os.path.join(ct_data_dir, "transfer.jsonl")
        
        ct_fn = os.path.join(save_dir, "transfer.jsonl")
        lang2docid = {}

        ## Locate the available docids
        assert src_docid_fn is not None, "docid_fn should be provided"
        if not os.path.exists(type_fn) or not os.path.exists(src_ct_fn) or not os.path.exists(src_docid_fn):
            print(f"Collecting cross-lingual transfer dataset to {save_dir}")
            dump_json(recipe, type_fn, pretty=True)
                
            for lang_pair in recipe:
                assert lang_pair.startswith("en-")
                tgt_lang = lang_pair[3:]
                assert tgt_lang in ["ja"], \
                    f"Target language {tgt_lang} not supported, only ja is supported. If you add new language, please carefully revise the code below"

                btoken_ct = recipe[lang_pair]
                for data_type in btoken_ct:
                    if btoken_ct[data_type] == 0:
                        continue
                    
                    print(f"Collecting {lang_pair}, {data_type} dataset")
                    filename = os.path.join(data_root, tgt_lang, f"{data_type.replace('-', '_')}.jsonl")
                    if not os.path.exists(filename):
                        raise FileNotFoundError(f"File not found: {filename}")
                    
                    if data_type.startswith("medical"):
                        exclude_docids = set()
                        if isinstance(btoken_ct[data_type], dict):
                            b_tokens = btoken_ct[data_type]['num_tokens']
                            if btoken_ct[data_type]['exclude'] is True:
                                # eval_root = os.path.join(config['dataset']['kg-dataset-dir'], f'{tgt_lang}-{btoken_kg[f"{tgt_lang}-medical"]}', "eval_qa")
                                # assert os.path.exists(eval_root), f"File not found for doc exclusion (Usually the evaluation data): {eval_root}"
                                # for prompt_type in ["bisent-prompt", "monosent-prompt", "trisent-prompt"]:
                                #     prompt_path = os.path.join(eval_root, prompt_type, "prompts.jsonl")
                                #     assert os.path.exists(prompt_path), f"File not found for doc exclusion (Usually the evaluation data): {prompt_path}"                                
                                #     for exclude_item in tqdm(load_jsonl_iteratively(prompt_path), "Collecting docids to exclude from evaluation data"):
                                #         exclude_docids.add(exclude_item['en']['docid'])
                                adaxeval_docids = os.path.join(config['dataset']['kg-dataset-dir'], "ja-0.5/eval_qa", "docids.jsonl")
                                assert os.path.exists(adaxeval_docids), f"File not found for doc exclusion (Usually the evaluation data): {adaxeval_docids}"
                                for docid in tqdm(load_jsonl_iteratively(adaxeval_docids), "Collecting docids to exclude from evaluation data"):
                                    exclude_docids.add(docid)
                            elif btoken_ct[data_type]['exclude'] is not False:
                                raise ValueError(f"exclude should be True or False, but get {btoken_ct[data_type]['exclude']}")
                            print(f"Excluding {len(exclude_docids)} docids from {adaxeval_docids}")
                        elif isinstance(btoken_ct[data_type], float) or isinstance(btoken_ct[data_type], int):
                            b_tokens = btoken_ct[data_type]    
                        else:
                            raise ValueError(f"btoken_ct should be a dict or a float, but get {type(btoken_ct[data_type])}")
                        
                        if lang2docid == {}:
                            only_allow_monolingual_kg_05B_training(btoken_kg)
                            # docid_fn = os.path.join(config['dataset']['kg-dataset-dir'], f'{tgt_lang}-{btoken_kg[f"{tgt_lang}-medical"]}', "doc_ids.jsonl")
                            docid_fn = os.path.join(config['dataset']['kg-dataset-dir'], f'{tgt_lang}-0.5', "doc_ids.jsonl")
                            for item in tqdm(load_jsonl_iteratively(docid_fn), desc=f"Collecting docids for training for cross-lingual transfer type: {data_type}"):
                                if item['lang'] not in lang2docid:
                                    lang2docid[item['lang']] = set()
                                if item['docid'] not in exclude_docids:
                                    lang2docid[item['lang']].add(item['docid'])
                        write_text(filename, src_ct_fn, b_tokens=b_tokens, tokenizer_type=tokenizer_type, filter_docids=lang2docid[tgt_lang])
                    else:
                        assert isinstance(btoken_ct[data_type], float) or isinstance(btoken_ct[data_type], int), f"btoken_ct should be a float or int, but get {type(btoken_ct[data_type])}"
                        write_text(filename, src_ct_fn, b_tokens=btoken_ct[data_type], tokenizer_type=tokenizer_type)
    
        if os.path.islink(ct_fn):
             os.unlink(ct_fn)
        
        os.symlink(src_ct_fn, ct_fn)
        print(f"Linking {src_ct_fn} to {ct_fn}")
        
        


