import os
import sys
import json

from tqdm import tqdm
from utils import DATA_ROOT, dump_json, load_json, load_jsonl_iteratively

def update_meta(meta_data, field, token_counter, num_docs):
    if llmjp_tokens != -1: 
        llmjp_tokens += 1
    if llama3_tokens != -1: 
        llama3_tokens += 1
    if qwen_tokens != -1: 
        qwen_tokens += 1
    meta_data['num_docs'][field] = num_docs
    meta_data['num_tokens']['llmjp'][field] = token_counter['llmjp']
    meta_data['num_tokens']['llama3'][field] = token_counter['llama3']
    meta_data['num_tokens']['qwen'][field] = token_counter['qwen']
    return meta_data

def initialize_token_counter():
    token_counter = {"llmjp": -1, "llama3": -1, "qwen": -1}
    return token_counter

def update_tokens(item, token_counter):
    if "llmjp-tokens" in item:
        llmjp_tokens += item["llmjp-tokens"]
    if "llama3-tokens" in item:
        llama3_tokens += item["llama3-tokens"]
    if "qwen-tokens" in item:
        qwen_tokens += item["qwen-tokens"]
    token_counter['llmjp'] += llmjp_tokens
    token_counter['llama3'] += llama3_tokens
    token_counter['qwen'] += qwen_tokens
    return token_counter
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="ja", choices=['zh', 'ja'])
    parser.add_argument("--use_en", action="store_true")
    parser.add_argument("--en_ratio", type=float, default=1)
    parser.add_argument("--use_roman", action="store_true")
    parser.add_argument("--use_halfroman", action="store_true")
    parser.add_argument("--use_roman2en", action="store_true")
    parser.add_argument("--use_native2en", action="store_true")
    parser.add_argument("--rewrite", action="store_true")

    args = parser.parse_args()

    lang = args.lang

    NUM_JA_DOCS = 61444
    DATASET_ROOT = os.path.join(DATA_ROOT, "datasets", "medical")
    INSTRUCTION_ROOT = os.path.join(DATA_ROOT, "instructions")
    TRAIN_ROOT = os.path.join(DATA_ROOT, "for_train", lang)
    os.makedirs(TRAIN_ROOT, exist_ok=True)

    if args.use_en:
        num_en_docs = NUM_JA_DOCS * args.en_ratio

    outfn = "base"
    if args.use_en:
        outfn += f'.with_{args.en_ratio}en'
    if args.use_roman:
        outfn += '.use_roman'
    if args.use_halfroman:
        outfn += '.use_halfroman'
    if args.use_native2en:
        outfn += '.use_native2en'
    if args.use_roman2en:
        outfn += '.use_roman2en'

    outfn += ".jsonl"

    if os.path.exists(outfn) and args.rewrite is False:
        print(f"The specified dataset is already generated: {outfn}")
        sys.exit(0)


    meta_data_fn = os.path.join(TRAIN_ROOT, "dataset.meta_data.json")
    meta_data = {} if not os.path.exists(meta_data_fn) else load_json(meta_data_fn)
    
    meta_data[meta_data_fn] = {"num_docs": {}, "num_tokens":{}}
    meta_data[meta_data_fn]["num_tokens"] = {
        "llmjp": {}, "llama3": {}, "qwen": {}
    }
    with open(os.path.join(TRAIN_ROOT, outfn), 'w', encoding="utf8") as outfp:
        # print(f"Adding Japanese raw text into datasets")
        # for i, item in tqdm(enumerate(load_jsonl_iteratively(
        #         os.path.join(DATASET_ROOT, lang, "data.jsonl")))):
        #     string = json.dumps({'text': f"{item['title']}。 {item['abstract']}"}, ensure_ascii=False)
        #     outfp.write(f"{string}\n")
        # meta_data[meta_data_fn]['num_docs'][f'plain_{args.lang}'] = i
        
        print(f"Adding Japanese native instructions into datasets")
        token_counter = initialize_token_counter()
        for num_docs, item in tqdm(enumerate(load_jsonl_iteratively(
                os.path.join(INSTRUCTION_ROOT, lang, "native.jsonl")))):
            string = json.dumps({'text': item['text']}, ensure_ascii=False)
            outfp.write(f"{string}\n")
            token_counter = update_tokens(token_counter)
        meta_data[meta_data_fn] = update_meta(
            meta_data=meta_data[meta_data_fn], field='native',
            token_counter=token_counter, prefix="native", num_docs=num_docs)
        
        if args.use_en:
            # print(f"Adding English raw text into datasets, with number of document: {len(num_en_docs)}")
            en_docids = set()
            for item in tqdm(load_jsonl_iteratively(
                    os.path.join(DATASET_ROOT, "en", "data.jsonl"), 
                    request_num=num_en_docs)):
                en_docids.add(item['docid'])
            #     string = json.dumps({'text': f"{item['title']} {item['abstract']}"}, ensure_ascii=False)
            #     outfp.write(f"{string}\n")
            # meta_data[meta_data_fn]['num_docs'][f'plain_en'] = num_en_docs

            token_counter = initialize_token_counter()
            print(f"Adding English native instructions into datasets")
            infn = os.path.join(INSTRUCTION_ROOT, 'en', 'native.jsonl')
            num_docs = 0
            for item in tqdm(load_jsonl_iteratively(infn)):
                if item['docid'] in en_docids:
                    string = json.dumps({'text': item['text']}, ensure_ascii=False)
                    outfp.write(f"{string}\n")
                    num_docs += 1
                    token_counter = update_tokens(token_counter)
            meta_data[meta_data_fn] = update_meta(
                meta_data=meta_data[meta_data_fn], field='en',
                token_counter=token_counter, num_docs=num_docs)
        
        if args.use_roman:
            print(f"Adding romanization instructions into datasets")
            token_counter = initialize_token_counter()
            infn = os.path.join(INSTRUCTION_ROOT, lang, 'roman.jsonl')
            for num_docs, item in tqdm(enumerate(load_jsonl_iteratively(infn))):
                string = json.dumps({'text': item['text']}, ensure_ascii=False)
                outfp.write(f"{string}\n")
                token_counter = update_tokens(token_counter)
            meta_data[meta_data_fn] = update_meta(
                meta_data=meta_data[meta_data_fn], field='roman',
                token_counter=token_counter, num_docs=num_docs)
        
        
        if args.use_halfroman:
            print(f"Adding half-romanization instructions into datasets")
            token_counter = initialize_token_counter()
            infn = os.path.join(INSTRUCTION_ROOT, lang, 'halfroman.jsonl')
            for num_docs, item in tqdm(enumerate(load_jsonl_iteratively(infn))):
                string = json.dumps({'text': item['text']}, ensure_ascii=False)
                outfp.write(f"{string}\n")
                token_counter = update_tokens(token_counter)
            meta_data[meta_data_fn] = update_meta(
                meta_data=meta_data[meta_data_fn], field='halfroman',
                token_counter=token_counter, num_docs=num_docs)
        

        if args.native2en:
            print(f"Adding translation instructions into datasets")
            token_counter = initialize_token_counter()
            infn = os.path.join(INSTRUCTION_ROOT, lang, 'native2en.jsonl')
            for num_docs, item in tqdm(enumerate(load_jsonl_iteratively(infn))):
                string = json.dumps({'text': item['text']}, ensure_ascii=False)
                outfp.write(f"{string}\n")
                token_counter = update_tokens(token_counter)
            meta_data[meta_data_fn] = update_meta(
                meta_data=meta_data[meta_data_fn], field='native2en',
                token_counter=token_counter, num_docs=num_docs)
        

        if args.use_roman2en:
            print(f"Adding cross-lingual-romain instructions into datasets")
            token_counter = initialize_token_counter()
            infn = os.path.join(INSTRUCTION_ROOT, lang, 'roman2en.jsonl')
            for num_docs, item in tqdm(enumerate(load_jsonl_iteratively(infn))):
                string = json.dumps({'text': item['text']}, ensure_ascii=False)
                outfp.write(f"{string}\n")
                token_counter = update_tokens(token_counter)
            meta_data[meta_data_fn] = update_meta(
                meta_data=meta_data[meta_data_fn], field='roman2en',
                token_counter=token_counter, num_docs=num_docs)
        


    dump_json(meta_data, meta_data_fn, pretty=True)

