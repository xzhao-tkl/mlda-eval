from ast import dump
import os
from pdb import set_trace
import re
import json 
import random
from sympy import li
from tqdm import tqdm
from utils import load_jsonl_iteratively 

LANG_UMLS_CODES = {
    "en": "ENG", "ja": "JPN", "es": "SPA", "cs": "CZE", 
    "fr": "FRA", "de": "GER", "pt": "POR", "it": "ITA",
    "sv": "SWE", "ru": "RUS", "pl": "POL", "fi": "FIN",
    "nl": "DUT", "ko": "KOR", "no": "NOR", "lv": "LVA",
    "hu": "HUN", "ar": "ARA", "tr": "TUR", "lt": "LIT", "is": "ISL"
}

POS2CHAR = {'NOUN': 'n', 'VERB': 'v', 'ADJ': 'a', 'ADV': 'r'}

WORDNET_ROOT = "/data/xzhao/dataset/roman-pretrain/datasets/medical/en_jstage/wordnet"
def load_tokenized_docs(start=0, end=None):
    """
    Load NER documents by both bionlp and scibert, from the specified range in iterativel way
    Parameters: 
    - start: The starting index (inclusive)
    - end: The ending index (exclusive)
    Return: A generator that yields tuples of (doc, scibert_doc)
    """
    end = 99999999 if end is None else end
    assert end > start
    filename = f"{WORDNET_ROOT}/tokenization_merged.full.jsonl"
    with open(filename, 'r', encoding="utf8") as fp:
        cnt = 0
        for line1 in tqdm(fp, "Processing tokenized documents to perform code-switching"):
            doc = json.loads(line1.rstrip('\n'))
            if cnt >= start and (end is None or cnt < end):
                yield doc
            cnt += 1
            if cnt > end:
                break

def get_tokens_for_switch(doc):
    """
    Get lemmas from a document based on the specified strategy.
    Parameters:
    - doc: `dict`, A NER document to retrieve lemmas from.
    - strategy: `string`, The strategy to use for retrieving lemmas (e.g., "max_score", "random").
    - exact_match: `bool`, Whether to perform exact matching on the target text (optional).
    - target_text: `str`, The target text to match against (required if exact_match is True).
    - random_seed: `int`, The random seed to use for reproducibility (optional).
    Return: A list of lemmas.
    """
    filtered_tokens = [token for token in doc['tokens'] if token['pos'] in ['NOUN', "ADJ", "VERB", "ADV"]]
    return filtered_tokens
    
def choice_synonym(all_syns, random_seed=None):
    """Choose one synonym from a list of synonyms based on the specified strategy."""
    if random_seed is not None:
        random.seed(random_seed)
    lang = random.choice(list(all_syns.keys()))
    return random.choice(all_syns[lang])

def code_switch(text, switch_recipe):
    offset = 0
    for switch in switch_recipe:
        # print(switch)
        text = text[:switch['start'] + offset] + switch['code'] + text[switch['end'] + offset:]
        offset += len(switch['code']) - (switch['end'] - switch['start'])
    return text

def process_docs_by_code_switching(doc, lemma_syns, switch_ratio=1.0, random_seed=None):
    assert "text" in doc, "Document must contain 'text' field"
    assert isinstance(doc['text'], str), "'text' field must be a string"

    if random_seed is not None:
        random.seed(random_seed)

    tokens = get_tokens_for_switch(doc)
    switch_recipe = []
    for i, token in enumerate(tokens):
        if token['pos'] not in POS2CHAR:
            continue
        token_pos = POS2CHAR[token['pos']]
        if (token['lemma'], token_pos) not in lemma_syns:
            continue

        syn_cands = lemma_syns[(token['lemma'], token_pos)]
        if 'en' in syn_cands:
            syn_cands['en'] = [s for s in syn_cands['en'] if s.lower() != token['lemma'].lower()]
            if syn_cands['en'] == []:
                syn_cands.pop('en')
        if len(syn_cands) == 0:
            continue

        lemma_syn = choice_synonym(all_syns=syn_cands)
        if lemma_syn is None:
            continue
        
        if random.random() > switch_ratio:
            continue
        switch_recipe.append({
            "start": token['start'],
            "end": token['end'],
            "raw": token["text"],
            "code": lemma_syn
        })
        # print(f"'{ent['text']}' -> '{ent_alias}'")
    sorted_recipe = sorted(switch_recipe, key=lambda x: x['start'])
    switched_text = code_switch(text=doc['text'], switch_recipe=sorted_recipe)
    # print("===> Count of CUIS:", len(cuis))
    # print("===> Count of switched entities:", len(switch_recipe))
    # print("===> Original text: ", doc['text'])
    # print("===> Switched text: ", switched_text)
    return switched_text, switch_recipe

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_lang", type=str, default="en", choices=["en"])
    parser.add_argument(
        "--tgt_langs", type=str, nargs='+', 
        default=["ja"], choices=["ja", "en"], help="List of target languages")
    parser.add_argument("--sample_times", type=int, default=1)
    parser.add_argument("--switch_ratio", type=float, default=1.0, help="The ratio of entities to be switched")
    parser.add_argument("--rewrite", action="store_true", help="Whether to rewrite the original text")

    args = parser.parse_args()

    assert args.src_lang in WORDNET_ROOT, "Source language not supported"
    
    # Define the file path to dump the code-switched file
    assert args.src_lang == "en", "Currently only support en as source language"
    if len(args.tgt_langs) == 1 and args.src_lang == args.tgt_langs[0]:
        dump_path = os.path.join(WORDNET_ROOT, f"wordnet.monolingual.{str(int(args.switch_ratio*100))}.jsonl")
    else:
        dump_path = os.path.join(WORDNET_ROOT, f"wordnet.multilingual.{'-'.join(args.tgt_langs)}.{str(int(args.switch_ratio*100))}.jsonl")

    if os.path.exists(dump_path) and not args.rewrite:
        print("====> Dump path already exists. Exiting...")
        sys.exit(0)

    dump_path_filehandler = open(dump_path, "w", encoding="utf-8")
    print("====> Dump path: ", dump_path)

    # Collect the English -> Japanese switching codes from preprocessed files (by 05_03_prepare_dataset.py)
    all_switch_codes = {}
    for lang in args.tgt_langs:
        codeswitch_datapath = f"{WORDNET_ROOT}/wordnet_{lang}.jsonl"
        for item in tqdm(load_jsonl_iteratively(codeswitch_datapath), desc="Loading code-switching data"):
            if (item["lemma"], item["pos"]) not in all_switch_codes:
                all_switch_codes[(item["lemma"], item["pos"])] = {}
            all_switch_codes[(item["lemma"], item["pos"])][lang] = item["synonyms"]

    pair_cnt = 0
    lemma_counter = set()
    for lemma, pos in all_switch_codes.keys():
        pair_cnt += 1
        lemma_counter.add(lemma)
        for lang in all_switch_codes[(lemma, pos)]:
            assert len(all_switch_codes[(lemma, pos)][lang]) > 0, f"No synonyms found for ({lemma}, {pos}) in {lang}"
    print("====> Total unique (lemma, pos) pairs: ", pair_cnt)
    print("====> Total unique lemmas: ", len(lemma_counter))

    # Load the raw data from English J-STAGE data
    for doc in load_tokenized_docs(start=0, end=None):
        new_doc = {
            "docid": doc["docid"], 
            "raw": {"keywords": [], "subjects": [], "sentences": []}, 
            "noisy": [{"keywords": [], "subjects": [], "sentences": []} for _ in range(args.sample_times)],
            "metadata": [{"keywords": [], "subjects": [], "sentences": []} for _ in range(args.sample_times)]}
        doc = doc['token']
        
        # Process title
        new_doc['raw']['abstract'] = doc['abstract']["text"]
        for ind in range(args.sample_times):
            new_doc["noisy"][ind]["abstract"], new_doc["metadata"][ind]["abstract"] = process_docs_by_code_switching(
                doc=doc['abstract'],
                lemma_syns=all_switch_codes, 
                switch_ratio=args.switch_ratio)

        # Process title
        new_doc['raw']['title'] = doc['title']["text"]
        for ind in range(args.sample_times):
            new_doc["noisy"][ind]["title"], new_doc["metadata"][ind]["title"] = process_docs_by_code_switching(
                doc=doc['title'],
                lemma_syns=all_switch_codes,
                switch_ratio=args.switch_ratio)

        # Process keywords
        new_doc['raw']['keywords'] = [keyword["text"] for keyword in doc['keywords']]
        for keyword in doc['keywords']:
            for ind in range(args.sample_times):
                switched_text, recipe = process_docs_by_code_switching(
                        doc=keyword,
                        lemma_syns=all_switch_codes,
                        switch_ratio=args.switch_ratio)
                new_doc["noisy"][ind]["keywords"].append(switched_text)
                new_doc["metadata"][ind]["keywords"].append(recipe)

        # Process subjects
        new_doc['raw']['subjects'] = [subject["text"] for subject in doc['subjects']]
        for keyword in doc['subjects']:
            for ind in range(args.sample_times):
                switched_text, recipe = process_docs_by_code_switching(
                    doc=keyword,
                    lemma_syns=all_switch_codes,
                    switch_ratio=args.switch_ratio)
                new_doc["noisy"][ind]["subjects"].append(switched_text)
                new_doc["metadata"][ind]["subjects"].append(recipe)

        # Process sentences
        new_doc['raw']['sentences'] = [sentence["text"] for sentence in doc['sentences']]
        for keyword in doc['sentences']:
            for ind in range(args.sample_times):
                switched_text, recipe = process_docs_by_code_switching(
                    doc=keyword,
                    lemma_syns=all_switch_codes,
                    switch_ratio=args.switch_ratio)
                new_doc["noisy"][ind]["sentences"].append(switched_text)
                new_doc["metadata"][ind]["sentences"].append(recipe)

        # Process generated question-answer pairs
        if 'qa' in doc:
            new_doc['raw']['qa'] = [[qa_pair["text"] for qa_pair in qa_section] for qa_section in doc['qa']]
            for ind in range(args.sample_times):
                new_doc["noisy"][ind]["qa"] = []
                new_doc["metadata"][ind]["qa"] = []
                for qa_ind in range(len(doc["qa"])):
                    new_doc["noisy"][ind]["qa"].append([])
                    new_doc["metadata"][ind]["qa"].append([])
                    for sent in doc["qa"][qa_ind]:
                        switched_text, recipe = process_docs_by_code_switching(
                                doc=sent, lemma_syns=all_switch_codes, switch_ratio=args.switch_ratio)
                        new_doc["noisy"][ind]["qa"][qa_ind].append(switched_text)
                        new_doc["metadata"][ind]["qa"][qa_ind].append(recipe)

        # Process conclusion-type data
        if "conclu" in doc:
            new_doc["raw"]["conclu"] = [sent["text"] for sent in doc["conclu"]]
            for ind in range(args.sample_times):
                new_doc["noisy"][ind]["conclu"] = []
                new_doc["metadata"][ind]["conclu"] = []
                for sent in doc["conclu"]:
                    switched_text, recipe = process_docs_by_code_switching(
                            doc=sent, lemma_syns=all_switch_codes, switch_ratio=args.switch_ratio)
                    new_doc["noisy"][ind]["conclu"].append(switched_text)
                    new_doc["metadata"][ind]["conclu"].append(recipe)

        # Process gmrc
        if "gmrc" in doc:
            new_doc["raw"]["gmrc"] = {}
            for key in doc["gmrc"]:
                new_doc["raw"]["gmrc"][key] = doc["gmrc"][key]
            for ind in range(args.sample_times):
                new_doc["noisy"][ind]["gmrc"] = {}
                new_doc["metadata"][ind]["gmrc"] = {}
                for key in doc["gmrc"]:
                    new_doc["noisy"][ind]["gmrc"][key], new_doc["metadata"][ind]["gmrc"][key] = process_docs_by_code_switching(
                        doc=doc["gmrc"][key], lemma_syns=all_switch_codes, switch_ratio=args.switch_ratio)

        if "causal" in doc:
            new_doc["raw"]["causal"] = {}
            for ind in range(args.sample_times):
                new_doc["noisy"][ind]["causal"] = {}
                new_doc["metadata"][ind]["causal"] = {}
                for causal, sents_ls in doc["causal"].items():
                    new_doc["raw"]["causal"][causal] = [[sent["text"] for sent in sents] for sents in sents_ls]
                    new_doc["noisy"][ind]["causal"][causal] = []
                    new_doc["metadata"][ind]["causal"][causal] = []
                    for i, sents in enumerate(sents_ls):
                        new_doc["noisy"][ind]["causal"][causal].append([])
                        new_doc["metadata"][ind]["causal"][causal].append([])
                        for j, sent in enumerate(sents):
                            switched_text, recipe = process_docs_by_code_switching(
                                    doc["causal"][causal][i][j], lemma_syns=all_switch_codes, switch_ratio=args.switch_ratio)
                            new_doc["noisy"][ind]["causal"][causal][-1].append(switched_text)
                            new_doc["metadata"][ind]["causal"][causal][-1].append(recipe)

        dump_path_filehandler.write(json.dumps(new_doc, ensure_ascii=False) + "\n")
    dump_path_filehandler.close()

