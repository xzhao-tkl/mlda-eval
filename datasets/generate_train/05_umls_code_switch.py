from ast import dump
import os
from pdb import set_trace
import re
import json 
import random
from tqdm import tqdm
from utils import load_jsonl_iteratively 

LANG_UMLS_CODES = {
    "en": "ENG", "ja": "JPN", "es": "SPA", "cs": "CZE", 
    "fr": "FRA", "de": "GER", "pt": "POR", "it": "ITA",
    "sv": "SWE", "ru": "RUS", "pl": "POL", "fi": "FIN",
    "nl": "DUT", "ko": "KOR", "no": "NOR", "lv": "LVA",
    "hu": "HUN", "ar": "ARA", "tr": "TUR", "lt": "LIT", "is": "ISL"
}
CODE_SWITCH_ROOT = "/data/xzhao/dataset/roman-pretrain/datasets/medical/en_jstage/code-switch"
def load_nered_docs(start=0, end=None):
    """
    Load NER documents by both bionlp and scibert, from the specified range in iterativel way
    Parameters: 
    - start: The starting index (inclusive)
    - end: The ending index (exclusive)
    Return: A generator that yields tuples of (bionlp_doc, scibert_doc)
    """
    end = 99999999 if end is None else end
    assert end > start
    bionlp_filename = f"{CODE_SWITCH_ROOT}/bionlp_merged.full.jsonl"
    scibert_filename = f"{CODE_SWITCH_ROOT}/scibert_merged.full.jsonl"
    with open(bionlp_filename, 'r', encoding="utf8") as f1, open(scibert_filename, 'r', encoding="utf8") as f2:
        cnt = 0
        for line1, line2 in tqdm(zip(f1, f2), "Processing NER documents to perform code-switching"):
            # Remove trailing newlines if needed
            bionlp_doc = json.loads(line1.rstrip('\n'))
            scibert_doc = json.loads(line2.rstrip('\n'))
            assert bionlp_doc["docid"] == scibert_doc["docid"], "IDs do not match"
            if cnt >= start and (end is None or cnt < end):
                yield bionlp_doc, scibert_doc
            cnt += 1
            if cnt > end:
                break

FILTER_PATTERNS = {
    "en": [" (qualifier value)", " (observable entity)"],
    "ja": ["Not Translated["]
}

def filter_alias_by_patterns(target_code, aliases, lang):
    if lang == 'ja':
        aliases = [alias for alias in aliases if not _has_hankaku_katakana(alias)]
    # elif lang == 'en':
    #     aliases = [alias for alias in aliases if not _filter_specifical_chars(alias)]
    
    aliases = [alias for alias in aliases if alias.lower() != target_code.lower()]
    return [alias for alias in aliases if not any(pat in alias for pat in FILTER_PATTERNS[lang])]
    
def _has_hankaku_katakana(text):
    """Check if the text contains Hankaku Katakana characters."""
    return bool(re.search(r'[\uff65-\uff9f]', text))

def _filter_specifical_chars(alias):
    # Allow letters, numbers, and spaces
    return re.fullmatch(r"[A-Za-z0-9 ]+", alias)

def get_aliases_by_cuis(target_code, cuis, cui_codes, langs):
    """Get aliases from cui_codes based on the provided CUIs and filter patterns."""
    aliases = set()
    for cui in cuis:
        if cui not in cui_codes:
            continue
        for lang in langs:
            if lang in cui_codes[cui]:
                filtered_alias = filter_alias_by_patterns(
                    target_code=target_code, 
                    aliases=cui_codes[cui][lang], lang=lang)
                aliases.update(filtered_alias)
    return list(aliases)

def get_umls_cuis(
        ner_docs, strategy, 
        exact_match=False, target_text=None, 
        random_seed=None, **kwargs):
    """
    Get UMLS CUIs from a list of NER documents based on the specified strategy.
    Parameters:
    - ner_docs: `List of dict`, A list of NER documents to retrieve UMLS CUIs from.
    - strategy: `string`, The strategy to use for retrieving UMLS CUIs (e.g., "max_score", "random").
    - exact_match: `bool`, Whether to perform exact matching on the target text (optional).
    - target_text: `str`, The target text to match against (required if exact_match is True).
    - random_seed: `int`, The random seed to use for reproducibility (optional).
    Return: A list of UMLS CUIs.
    """
    cuis = [cui for cui in ner_docs if 'aliases' in cui and len(cui['aliases']) > 0]

    filtered_cuis = []
    if exact_match:
        assert target_text is not None, "Target text must be provided for exact match"
        for cui in cuis:
            lowered_aliases = set(alias.lower() for alias in cui['aliases'])
            if target_text.lower() in lowered_aliases:
                filtered_cuis.append(cui)
    else:
        filtered_cuis = cuis

    if len(filtered_cuis) == 0:
        return []
    
    if strategy == "max_score":
        max_score = -1
        for cui_ent in filtered_cuis:
            if cui_ent['score'] > max_score:
                best_cand = cui_ent
        return [best_cand['cui']]
    elif strategy == "random":
        if random_seed is not None:
            random.seed(random_seed)
        return [random.choice(filtered_cuis)['cui']]
    elif strategy == "all":
        return [cui_ent['cui'] for cui_ent in filtered_cuis]
    elif strategy == "threshold":
        assert "threshold" in kwargs, "Threshold value is required for 'threshold' strategy"
        return [cui_ent['cui'] for cui_ent in filtered_cuis if cui_ent['score'] >= kwargs["threshold"]]
    else:
        raise NotImplementedError("Unknown strategy: {} for sampling CUI".format(strategy))
    
def choice_cui_alias(aliases, strategy, replaced_text, random_seed=None):
    """Choose one alias from a list of aliases based on the specified strategy."""
    if random_seed is not None:
        random.seed(random_seed)
    
    if len(aliases) == 0:
        return None 
    
    if strategy == 'random':
        return random.choice(aliases)
    elif strategy.endswith('_diff'):
        y = set(replaced_text.lower().split())
        diffs = []
        for alias in aliases:
            x = set(alias.lower().split())
            diffs.append(1 - len(x.intersection(y)) / len(x.union(y)))
        if strategy == 'max_diff':
            cands = [alias for alias in aliases if diffs[aliases.index(alias)] == max(diffs)]
        elif strategy == 'min_diff':
            cands = [alias for alias in aliases if diffs[aliases.index(alias)] == min(diffs)]
        else:
            raise NotImplementedError("Unknown strategy: {} for selecting alias from CUI retrieval".format(strategy))
        return random.choice(cands) if cands else None
    else:
        raise NotImplementedError("Unknown strategy: {} for selecting alias from CUI retrieval".format(strategy))

def code_switch(text, switch_recipe):
    offset = 0
    for switch in switch_recipe:
        # print(switch)
        text = text[:switch['start'] + offset] + switch['code'] + text[switch['end'] + offset:]
        offset += len(switch['code']) - (switch['end'] - switch['start'])
    return text

def process_docs_by_code_switching(
        bionlp_doc, scibert_doc, cui_codes, langs,
        switch_ratio=1.0, random_seed=None):
    assert "text" in bionlp_doc and "text" in scibert_doc, "Both documents must contain 'text' field"
    assert isinstance(bionlp_doc['text'], str) and isinstance(scibert_doc['text'], str), "Both 'text' fields must be strings"
    assert bionlp_doc['text'] == scibert_doc['text'], "Text fields do not match"

    if random_seed is not None:
        random.seed(random_seed)
    
    switch_recipe, examined_indexes = [], []
    merged_entities = bionlp_doc['entities'] + scibert_doc['entities']
    for i, ent in enumerate(merged_entities):
        if len(ent['ents']) == 0:
            continue
        
        # Skip if this entity overlaps with any already examined entity
        if next((True for start, end in examined_indexes if not (ent['end'] < start or ent['start'] > end)), False):
            continue

        examined_indexes.append((ent['start'], ent['end']))
        cuis = get_umls_cuis(
            ner_docs=ent['ents'], strategy="all", random_seed=random_seed+i,
            exact_match=True, target_text=ent['text'], threshold=0.8)
        alias_cands = get_aliases_by_cuis(
            target_code=ent["text"], cuis=cuis, cui_codes=cui_codes, langs=langs)
        ent_alias = choice_cui_alias(
            aliases=alias_cands, strategy="random",
            replaced_text=ent["text"], random_seed=random_seed + i)
        # set_trace()
        if ent_alias is None:
            continue
        
        if random.random() > switch_ratio:
            continue
        switch_recipe.append({
            "start": ent['start'],
            "end": ent['end'],
            "raw": ent["text"],
            "code": ent_alias,
            "type": "en->ja"
        })
        # print(f"'{ent['text']}' -> '{ent_alias}'")
    sorted_recipe = sorted(switch_recipe, key=lambda x: x['start'])
    switched_text = code_switch(text=bionlp_doc['text'], switch_recipe=sorted_recipe)
    # print("===> Count of CUIS:", len(cuis))
    # print("===> Count of switched entities:", len(switch_recipe))
    # print("===> Original text: ", bionlp_doc['text'])
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

    assert args.src_lang in CODE_SWITCH_ROOT, "Source language not supported"
    
    # Define the file path to dump the code-switched file
    assert args.src_lang == "en", "Currently only support en as source language"
    if len(args.tgt_langs) == 1 and args.src_lang == args.tgt_langs[0]:
        dump_path = f"umls.monolingual.{str(int(args.switch_ratio*100))}.jsonl"
    else:
        dump_path = f"umls.multilingual.{'-'.join(args.tgt_langs)}.{str(int(args.switch_ratio*100))}.jsonl"

    if os.path.exists(dump_path) and not args.rewrite:
        print("====> Dump path already exists. Exiting...")
        sys.exit(0)

    dump_path_filehandler = open(dump_path, "w", encoding="utf-8")
    print("====> Dump path: ", dump_path)

    # Collect the English -> Japanese switching codes from preprocessed files (by 05_03_prepare_dataset.py)
    all_switch_codes = {}
    for ner_tool in ["scibert", "bionlp"]:
        codeswitch_datapath = f"{CODE_SWITCH_ROOT}/cui_all-{ner_tool}.jsonl"
        for item in tqdm(load_jsonl_iteratively(codeswitch_datapath), desc="Loading code-switching data"):
            if len(item["translations"]) == 0 or item["cui"] in all_switch_codes:
                continue
            all_switch_codes[item["cui"]] = {}
            for lang in args.tgt_langs:
                assert lang in FILTER_PATTERNS, f"Language {lang} not in filter patterns"
                if lang in LANG_UMLS_CODES and LANG_UMLS_CODES[lang] in item["translations"]:
                    all_switch_codes[item["cui"]][lang] = item["translations"][LANG_UMLS_CODES[lang]]
        
    bionlp_cnt, scibert_cnt = 0, 0
    for cui in all_switch_codes:
        if "bionlp" in all_switch_codes[cui]:
            bionlp_cnt += 1
        if "scibert" in all_switch_codes[cui]:
            scibert_cnt += 1
    print(f"bionlp: {bionlp_cnt}, scibert: {scibert_cnt}, all: {len(all_switch_codes)}")

    # Load the raw data from English J-STAGE data
    for bionlp_doc, scibert_doc in load_nered_docs(start=0, end=None):
        assert bionlp_doc["docid"] == scibert_doc["docid"], "Document IDs do not match"
        new_doc = {
            "docid": bionlp_doc["docid"], 
            "raw": {"keywords": [], "subjects": [], "sentences": []}, 
            "noisy": [{"keywords": [], "subjects": [], "sentences": []} for _ in range(args.sample_times)],
            "metadata": [{"keywords": [], "subjects": [], "sentences": []} for _ in range(args.sample_times)]}
        bionlp_doc = bionlp_doc['ner']
        scibert_doc = scibert_doc['ner'] 

        # Process title
        new_doc['raw']['abstract'] = bionlp_doc['abstract']["text"]
        for ind in range(args.sample_times):
            new_doc["noisy"][ind]["abstract"], new_doc["metadata"][ind]["abstract"] = process_docs_by_code_switching(
                bionlp_doc=bionlp_doc['abstract'], scibert_doc=scibert_doc['abstract'], langs=args.tgt_langs,
                cui_codes=all_switch_codes, switch_ratio=args.switch_ratio, random_seed=42 + ind)

        # Process title
        new_doc['raw']['title'] = bionlp_doc['title']["text"]
        for ind in range(args.sample_times):
            new_doc["noisy"][ind]["title"], new_doc["metadata"][ind]["title"] = process_docs_by_code_switching(
                bionlp_doc=bionlp_doc['title'], scibert_doc=scibert_doc['title'], langs=args.tgt_langs,
                cui_codes=all_switch_codes, switch_ratio=args.switch_ratio, random_seed=42 + ind)

        # Process keywords
        new_doc['raw']['keywords'] = [keyword["text"] for keyword in bionlp_doc['keywords']]
        for bionlp_keyword, scibert_keyword in zip(bionlp_doc['keywords'], scibert_doc['keywords']):
            for ind in range(args.sample_times):
                switched_text, recipe = process_docs_by_code_switching(
                        bionlp_doc=bionlp_keyword, scibert_doc=scibert_keyword, langs=args.tgt_langs,
                        cui_codes=all_switch_codes, switch_ratio=args.switch_ratio, random_seed=42)
                new_doc["noisy"][ind]["keywords"].append(switched_text)
                new_doc["metadata"][ind]["keywords"].append(recipe)

        # Process subjects
        new_doc['raw']['subjects'] = [subject["text"] for subject in bionlp_doc['subjects']]
        for bionlp_subject, scibert_subject in zip(bionlp_doc['subjects'], scibert_doc['subjects']):
            for ind in range(args.sample_times):
                switched_text, recipe = process_docs_by_code_switching(
                    bionlp_doc=bionlp_subject, scibert_doc=scibert_subject, langs=args.tgt_langs,
                    cui_codes=all_switch_codes, switch_ratio=args.switch_ratio, random_seed=42 + ind)
                new_doc["noisy"][ind]["subjects"].append(switched_text)
                new_doc["metadata"][ind]["subjects"].append(recipe)

        # Process sentences
        new_doc['raw']['sentences'] = [sentence["text"] for sentence in bionlp_doc['sentences']]
        for bionlp_sentence, scibert_sentence in zip(bionlp_doc['sentences'], scibert_doc['sentences']):
            for ind in range(args.sample_times):
                switched_text, recipe = process_docs_by_code_switching(
                    bionlp_doc=bionlp_sentence, scibert_doc=scibert_sentence, langs=args.tgt_langs,
                    cui_codes=all_switch_codes, switch_ratio=args.switch_ratio, random_seed=42 + ind)
                new_doc["noisy"][ind]["sentences"].append(switched_text)
                new_doc["metadata"][ind]["sentences"].append(recipe)

        # Process generated question-answer pairs
        if 'qa' in bionlp_doc:
            assert 'qa' in scibert_doc, 'QA section is missing in SciBERT document'
            assert len(bionlp_doc["qa"]) == len(scibert_doc["qa"]), "The number of QA sections do not match"
            new_doc['raw']['qa'] = [[qa_pair["text"] for qa_pair in qa_section] for qa_section in bionlp_doc['qa']]
            for ind in range(args.sample_times):
                new_doc["noisy"][ind]["qa"] = []
                new_doc["metadata"][ind]["qa"] = []
                for qa_ind in range(len(bionlp_doc["qa"])):
                    new_doc["noisy"][ind]["qa"].append([])
                    new_doc["metadata"][ind]["qa"].append([])
                    for bionlp_sent, scibert_sent in zip(bionlp_doc["qa"][qa_ind], scibert_doc["qa"][qa_ind]):
                        switched_text, recipe = process_docs_by_code_switching(
                                bionlp_doc=bionlp_sent, scibert_doc=scibert_sent, langs=args.tgt_langs,
                                cui_codes=all_switch_codes, switch_ratio=args.switch_ratio, random_seed=42 + ind)
                        new_doc["noisy"][ind]["qa"][qa_ind].append(switched_text)
                        new_doc["metadata"][ind]["qa"][qa_ind].append(recipe)

        # Process conclusion-type data
        if "conclu" in bionlp_doc:
            new_doc["raw"]["conclu"] = [sent["text"] for sent in bionlp_doc["conclu"]]
            for ind in range(args.sample_times):
                new_doc["noisy"][ind]["conclu"] = []
                new_doc["metadata"][ind]["conclu"] = []
                assert "conclu" in scibert_doc, 'Conclusion section is missing in SciBERT document'
                for bionlp_sent, scibert_sent in zip(bionlp_doc["conclu"], scibert_doc["conclu"]):
                    switched_text, recipe = process_docs_by_code_switching(
                            bionlp_doc=bionlp_sent, scibert_doc=scibert_sent, langs=args.tgt_langs,
                            cui_codes=all_switch_codes, switch_ratio=args.switch_ratio, random_seed=42 + ind)
                    new_doc["noisy"][ind]["conclu"].append(switched_text)
                    new_doc["metadata"][ind]["conclu"].append(recipe)

        # Process gmrc
        if "gmrc" in bionlp_doc:
            assert "gmrc" in scibert_doc, 'GMRC section is missing in SciBERT document'
            new_doc["raw"]["gmrc"] = {}
            for key in bionlp_doc["gmrc"]:
                new_doc["raw"]["gmrc"][key] = bionlp_doc["gmrc"][key]
            for ind in range(args.sample_times):
                new_doc["noisy"][ind]["gmrc"] = {}
                new_doc["metadata"][ind]["gmrc"] = {}
                for key in bionlp_doc["gmrc"]:
                    assert key in scibert_doc["gmrc"], f'Key {key} is missing in SciBERT GMRC section'
                    new_doc["noisy"][ind]["gmrc"][key], new_doc["metadata"][ind]["gmrc"][key] = process_docs_by_code_switching(
                        bionlp_doc=bionlp_doc["gmrc"][key], scibert_doc=scibert_doc["gmrc"][key], langs=args.tgt_langs,
                        cui_codes=all_switch_codes, switch_ratio=args.switch_ratio, random_seed=42 + ind)

        if "causal" in bionlp_doc:
            assert "causal" in scibert_doc, 'causal section is missing in SciBERT document'
            new_doc["raw"]["causal"] = {}
            for ind in range(args.sample_times):
                new_doc["noisy"][ind]["causal"] = {}
                new_doc["metadata"][ind]["causal"] = {}
                for causal, sents_ls in bionlp_doc["causal"].items():
                    assert causal in scibert_doc["causal"], f'Key {causal} is missing in SciBERT causal section'
                    new_doc["raw"]["causal"][causal] = [[sent["text"] for sent in sents] for sents in sents_ls]
                    new_doc["noisy"][ind]["causal"][causal] = []
                    new_doc["metadata"][ind]["causal"][causal] = []
                    for i, sents in enumerate(sents_ls):
                        new_doc["noisy"][ind]["causal"][causal].append([])
                        new_doc["metadata"][ind]["causal"][causal].append([])
                        for j, sent in enumerate(sents):
                            switched_text, recipe = process_docs_by_code_switching(
                                    bionlp_doc["causal"][causal][i][j], scibert_doc["causal"][causal][i][j], langs=args.tgt_langs,
                                    cui_codes=all_switch_codes, switch_ratio=args.switch_ratio, random_seed=42)
                            new_doc["noisy"][ind]["causal"][causal][-1].append(switched_text)
                            new_doc["metadata"][ind]["causal"][causal][-1].append(recipe)

        dump_path_filehandler.write(json.dumps(new_doc, ensure_ascii=False) + "\n")
    dump_path_filehandler.close()

