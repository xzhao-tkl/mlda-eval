# Load model directly
import os
from pdb import set_trace
import sys
import json
from transformers import AutoTokenizer
from tqdm import tqdm
from utils import DATA_ROOT, load_jsonl_iteratively
 

def write_tokenize_file(tokenizer_name, infn, outfn):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer_name == "llm-jp/llm-jp-3-7.2b":
        tokenizer_name = "llm-jp"
    elif tokenizer_name == "Qwen/Qwen2.5-7B":
        tokenizer_name = "qwen2.5"
    elif tokenizer_name == "meta-llama/Meta-Llama-3-8B":
        tokenizer_name = "llama3"
    else:
        raise NotImplementedError
    
    with open(outfn, 'w', encoding="utf8") as fp:
        for i, item in tqdm(enumerate(load_jsonl_iteratively(infn))):
            if tokenizer_name not in item:
                try:
                    token_ids = tokenizer(item['text'])['input_ids']
                    item[tokenizer_name] = {
                        "num_tokens": len(token_ids),
                        "token_ids": token_ids
                    }
                except:
                    item[tokenizer_name] = {
                        "num_tokens": 0,
                        "token_ids": []
                    }
                    print(f"Encounter error for item: {item}")
                    
            string = json.dumps(item, ensure_ascii=False)
            fp.write(f"{string}\n")
            
    os.replace(outfn, infn)

if __name__ == "__main__":
    import argparse
    from multiprocessing import Pool

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang", type=str, default="zh", 
        choices=[
            'zh', 'ja', 'en', 'en_jstage', 
            'denoising/baseline',
            'denoising/syntax.denoising',
            'denoising/umls.monolingual.50',
            'denoising/umls.monolingual.100',
            'denoising/umls.multilingual.ja.50', 
            'denoising/umls.multilingual.ja.100', 
            'denoising/wordnet.monolingual.16',
            'denoising/wordnet.monolingual.32',
            'denoising/wordnet.monolingual.64',
            'denoising/wordnet.multilingual.ja.16',
            'denoising/wordnet.multilingual.ja.32',
            'denoising/wordnet.multilingual.ja.64',
            ])

    parser.add_argument("--file_pattern", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, 
                        choices=[
                            "llm-jp/llm-jp-3-7.2b", 
                            "meta-llama/Meta-Llama-3-8B", 
                            "Qwen/Qwen2.5-7B"])
    args = parser.parse_args()

    
    tasks = []
    for fn in os.listdir(os.path.join(DATA_ROOT, "instructions", args.lang)):
        if os.path.isdir(os.path.join(DATA_ROOT, "instructions", args.lang, fn)):
            continue
        if args.file_pattern is not None and args.file_pattern not in fn:
            continue
        if "token" in fn:
            continue
        if args.lang == 'en' and "subset" not in fn:
            continue

        outfn = fn[:-6] + ".token" + fn[-6:]
        infn = os.path.join(DATA_ROOT, "instructions", args.lang, fn)
        outfn = os.path.join(DATA_ROOT, "instructions", args.lang, outfn)
        print(f"Tokenizing {infn}")

        tasks.append((args.tokenizer, infn, outfn))

    with Pool(processes=16) as pool:
        pool.starmap(write_tokenize_file, tasks)