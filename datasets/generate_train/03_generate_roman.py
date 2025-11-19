import os
import re
import json
from tqdm import tqdm

from utils import DATA_ROOT, dump_jsonl
from utils import load_jsonl_iteratively


def merge_data_with_qa_ja(datasets_fn, response_fn):
    items = {}
    for doc_item in tqdm(load_jsonl_iteratively(datasets_fn), desc=f"Reading documents from preprocessed data: {datasets_fn}"):
        items[doc_item['docid']] = doc_item
    non_match_doc, less_5_qa = [], []
    non_match_text = []
    qaffix_affix = r"#+\s*(?:\d+\.\s*)?(?:[\u3040-\u30FF\u4E00-\u9FAF\uFF66-\uFF9F]{2,12})?(?:質問|問題|问题|质問|質问|质问|賨問|要問|膪問|贅問|背景的な質問|質急|质询|資問|落川|賶問)(?:\d+)?(?:\s*\d+)?(?::|：)"
    qaffix_pattern = qaffix_affix + r"(?!\s*\.)"

    for i, res_item in tqdm(enumerate(load_jsonl_iteratively(response_fn)), desc="Processing QA response"):
        assert res_item['id'][9:] in items
        generation = res_item['choices'][0]['message']['content']
        if "</think>" in generation:
            qa_text = generation.split("</think>")[1].strip()
        elif "\n\n\n" in generation and len(generation.split("\n\n\n")) == 2:
            qa_text = generation.split("\n\n\n")[1].strip()
        elif sum(1 for _ in re.finditer(qaffix_pattern, generation)) >= 2:
            qa_text = generation[list(re.finditer(qaffix_pattern, generation))[0].start():]
        elif sum(1 for _ in re.finditer(qaffix_pattern, generation)) >= 2:
            qa_text = generation[list(re.finditer(qaffix_pattern, generation))[0].start():]
        else:
            non_match_doc.append(generation)
            continue

        start_indices = []
        for match in re.finditer(qaffix_pattern, qa_text):
            start_indices.append(match.start())
        start_indices.append(-1)
        
        if len(start_indices) <= 2:
            less_5_qa.append((start_indices, qa_text))
            continue
        
        qaffix_affix = r"#+\s*(?:\d+\.\s*)?(?:[\u3040-\u30FF\u4E00-\u9FAF\uFF66-\uFF9F]{2,12})?(?:質問|問題|问题|质問|質问|质问|賨問|要問|膪問|贅問|背景的な質問|質急|质询|資問|落川|問|賶問|)(?:\d+)?(?:\s*\d+)?(?::|：)\s*"
        aaffix = r'#+\s*(?:\d+\.\s*)?(?:回答|答案|回應|回问|回问答)(?:\d+)?(?:\s*\d+)?(?::|：)\s*'
        qa_pairs = []
        for start, end in zip(start_indices[:-1], start_indices[1:]):
            this_qa_text = qa_text[start:end].strip()
            try:
                match = re.match(qaffix_affix + r"\s*(.*?)\n*" + aaffix + r"(.*)", this_qa_text, re.DOTALL)
                if match is not None:
                    q = match.group(1).strip()
                    a = match.group(2).strip()
                    assert len(q) > 0 and len(a) > 0
                    qa_pairs.append((q, a))
                else:
                    non_match_text.append(((this_qa_text, qa_text)))
            except Exception as e:
                non_match_text.append(((this_qa_text, qa_text)))
        items[res_item['id'][9:]]['qa'] = qa_pairs
    print(len(non_match_doc), len(less_5_qa), len(non_match_text))
    return items

def merge_data_with_qa_zh(datasets_fn, response_fn):
    items = {}
    for doc_item in tqdm(
        load_jsonl_iteratively(datasets_fn), 
        desc=f"Reading documents from preprocessed data: {datasets_fn}"):
        items[doc_item['docid']] = doc_item
    non_match_doc, less_5_qa = [], []
    non_match_text = []
    qaffix_affix = r"#+\s*(?:\d+\.\s*)?(?:提问|提問|提答|提考|問題|供稿|提請|提交问|问题)(?:\d+)?(?:\s*\d+)?(?::|：)"
    qaffix_pattern = qaffix_affix + r"(?!\s*\.)"

    for i, res_item in tqdm(enumerate(load_jsonl_iteratively(response_fn)), desc="Processing QA response"):
        assert int(res_item['id'][9:]) in items
        generation = res_item['choices'][0]['message']['content']
        if "</think>" in generation:
            qa_text = generation.split("</think>")[1].strip()
        elif "\n\n\n" in generation and len(generation.split("\n\n\n")) == 2:
            qa_text = generation.split("\n\n\n")[1].strip()
        elif sum(1 for _ in re.finditer(qaffix_pattern, generation)) >= 5:
            qa_text = generation[list(re.finditer(qaffix_pattern, generation))[0].start():]
        elif sum(1 for _ in re.finditer(qaffix_pattern, generation)) >= 5:
            qa_text = generation[list(re.finditer(qaffix_pattern, generation))[0].start():]
        else:
            non_match_doc.append(generation)
            continue

        start_indices = []
        for match in re.finditer(qaffix_pattern, qa_text):
            start_indices.append(match.start())
        start_indices.append(-1)
        
        if len(start_indices) <= 2:
            less_5_qa.append((start_indices, qa_text))
            continue
        
        aaffix = r'#+\s*(?:\d+\.\s*)?(?:回答|答案|回應|回问|回问答)(?:\d+)?(?:\s*\d+)?(?::|：)\s*'
        qa_pairs = []
        for start, end in zip(start_indices[:-1], start_indices[1:]):
            this_qa_text = qa_text[start:end].strip()
            try:
                match = re.match(qaffix_affix + r"\s*(.*?)\n*" + aaffix + r"(.*)", this_qa_text, re.DOTALL)
                if match is not None:
                    q = match.group(1).strip()
                    a = match.group(2).strip()
                    assert len(q) > 0 and len(a) > 0
                    qa_pairs.append((q, a))
                else:
                    non_match_text.append(this_qa_text)
            except Exception as e:
                non_match_text.append(this_qa_text)
                continue
        items[int(res_item['id'][9:])]['qa'] = qa_pairs
    return items

def merge_data_with_qa_en(datasets_fn, response_fn):
    items = {}
    for doc_item in tqdm(
        load_jsonl_iteratively(datasets_fn), 
        desc=f"Reading documents from preprocessed data: {datasets_fn}"):
        items[doc_item['docid']] = doc_item
    non_match_doc, less_5_qa = [], []
    non_match_text = []
    qaffix_affix = r'(?:#+\s*)?(?:\*\*)?(?:question|Question)(?:\d+)?(?:\*\*)?(?:\s*\d+)?\s*(?::|：)\s*'
    qaffix_pattern = qaffix_affix + r"(?!\s*\.)"

    for i, res_item in tqdm(enumerate(load_jsonl_iteratively(response_fn)), desc="Processing QA response"):
        assert str(res_item['id'][9:]) in items
        generation = res_item['choices'][0]['message']['content']
        if "</think>" in generation:
            qa_text = generation.split("</think>")[1].strip()
        elif "\n\n\n" in generation and len(generation.split("\n\n\n")) == 2:
            qa_text = generation.split("\n\n\n")[1].strip()
        elif sum(1 for _ in re.finditer(qaffix_pattern, generation)) >= 5:
            qa_text = generation[list(re.finditer(qaffix_pattern, generation))[0].start():]
        elif sum(1 for _ in re.finditer(qaffix_pattern, generation)) >= 5:
            qa_text = generation[list(re.finditer(qaffix_pattern, generation))[0].start():]
        else:
            non_match_doc.append(generation)
            continue

        start_indices = []
        for match in re.finditer(qaffix_pattern, qa_text):
            start_indices.append(match.start())
        start_indices.append(-1)
        
        if len(start_indices) <= 2:
            less_5_qa.append((start_indices, qa_text))
            continue
        
        aaffix = r'(?:#+\s*)?(?:\*\*)?(?:Answer|answer)(?:\d+)?(?:\*\*)?(?:\s*\d+)?\s*(?::|：)\s*'
        qa_pairs = []
        for start, end in zip(start_indices[:-1], start_indices[1:]):
            this_qa_text = qa_text[start:end].strip()
            try:
                match = re.match(qaffix_affix + r"\s*(.*?)\n*" + aaffix + r"(.*)", this_qa_text, re.DOTALL)
                if match is not None:
                    q = match.group(1).strip()
                    a = match.group(2).strip()
                    assert len(q) > 0 and len(a) > 0
                    qa_pairs.append((q, a))
                else:
                    non_match_text.append(this_qa_text)
            except Exception as e:
                non_match_text.append(this_qa_text)
                continue
        items[str(res_item['id'][9:])]['qa'] = qa_pairs
    return items

import jieba
import cutlet
from pypinyin import pinyin, Style

katsu = cutlet.Cutlet()


def chinese_to_pinyin(text):
    segments = re.findall(r'[\u4e00-\u9fff]+|[^\u4e00-\u9fff]+', text)
    pinyin_segments = []
    for segment in segments:
        if re.match(r'[\u4e00-\u9fff]+', segment):
            words = jieba.lcut(segment)
            pys = []
            for word in words:
                _pys = pinyin(word, style=Style.NORMAL)
                pys.append(''.join([py[0] for py in _pys]))
            pinyin_word = ' '.join(pys)
            pinyin_segments.append(pinyin_word)
        else:
            pinyin_segments.append(segment)
    return re.sub(r'\s+([^\w\s])', r'\1', ' '.join(pinyin_segments))

def romanize(text, lang):
    if lang == 'zh':
        return chinese_to_pinyin(text)
    elif lang == 'ja':
        try:
            return katsu.romaji(text)
        except Exception as e:
            return None

def romanize_doc(item, lang):
    roman_item = {}
    for keyword in item:
        if keyword == "docid": 
            continue
        elif isinstance(item[keyword], str):
            roman_item[keyword] = romanize(item[keyword], lang)
        elif keyword in ["keywords", "subjects", "sentences", "conclu"]:
            roman_item[keyword] = [romanize(text, lang) for text in item[keyword]]
        elif keyword in ["gmrc"]:
            roman_item[keyword] = {}
            for key, text in item[keyword].items():
                roman_item[keyword][key] = romanize(text, lang)
        elif keyword in ["causal"]:
            roman_item[keyword] = {}
            for key in item[keyword].keys():
                roman_item[keyword][key] = []
                text_list = item[keyword][key]
                for text_pair in text_list:
                    roman_item[keyword][key].append([romanize(text_pair[0], lang), romanize(text_pair[1], lang)])
        elif keyword == "qa":
            roman_item[keyword] = []
            for q, a in item[keyword]:
                roman_item[keyword].append((romanize(q, lang), romanize(a, lang)))
    return roman_item
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="zh", choices=['zh', 'ja', 'en', 'en_jstage', 'en-ja', 'en-zh', 'zh-ja'])
    parser.add_argument("--trans_type", type=str, default="zh", choices=['balanced_trans', 'science_trans'])
    parser.add_argument("--rewrite", action="store_true")
    args = parser.parse_args()

    
    if args.lang in ['zh', 'ja']:
        datasets_fn = os.path.join(DATA_ROOT, 'datasets', "medical", args.lang, "preprocessed.jsonl")
        response_fn = os.path.join(DATA_ROOT, 'response', "medical", args.lang, "response.full.jsonl")
        dump_fn = os.path.join(DATA_ROOT, 'datasets', "medical", args.lang, "full.jsonl")
        if args.lang == 'zh':
            items = merge_data_with_qa_zh(datasets_fn, response_fn)
        elif args.lang == 'ja':
            items = merge_data_with_qa_ja(datasets_fn, response_fn)
        else:
            items = merge_data_with_qa_en(datasets_fn, response_fn)

        if not args.rewrite:
            docids = set()
            if os.path.exists(dump_fn):
                for item in load_jsonl_iteratively(dump_fn):
                    docids.add(item['docid'])
        print(f"Writing romanized items to {dump_fn}")

        mode = "w" if args.rewrite else "a"
        with open(dump_fn, mode, encoding="utf8") as fp:
            for docid in tqdm(items.keys(), total=len(items)):
                if not args.rewrite and docid in docids:
                    continue
                
                roman_item = {}
                item = items[docid]
                roman_item = romanize_doc(item, args.lang)
                raw_item = {key:value for key, value in item.items() if key != "docid"}
                all_item = {
                    "docid": item["docid"],
                    "raw": raw_item,
                    "roman": roman_item,
                }
                string = json.dumps(all_item, ensure_ascii=False)
                fp.write(f"{string}\n")
    elif args.lang == 'en' or args.lang == 'en_jstage':
        datasets_fn = os.path.join(DATA_ROOT, 'datasets', "medical", args.lang, "preprocessed.jsonl")
        response_fn = os.path.join(DATA_ROOT, 'response', args.lang, "response.full.jsonl")
        dump_fn = os.path.join(DATA_ROOT, 'datasets', "medical", args.lang, "full.jsonl")
        items = merge_data_with_qa_en(datasets_fn, response_fn)

        if not args.rewrite:
            docids = set()
            if os.path.exists(dump_fn):
                for item in load_jsonl_iteratively(dump_fn):
                    docids.add(item['docid'])
        
        print(f"Writing romanized items to {dump_fn} ...")
        mode = "w" if args.rewrite else "a"
        with open(dump_fn, mode, encoding="utf8") as fp:
            for docid in tqdm(items.keys(), total=len(items)):
                if not args.rewrite and docid in docids:
                    continue
                
                item = items[docid]
                raw_item = {key:value for key, value in item.items() if key != "docid"}
                all_item = {
                    "docid": item["docid"],
                    "raw": raw_item,
                }
                string = json.dumps(all_item, ensure_ascii=False)
                fp.write(f"{string}\n")
    else:
        folder = "balanced_bilingual" if args.trans_type == 'balanced_trans' else 'scientific_bilingual'
        datasets_fn = os.path.join(DATA_ROOT, 'datasets', folder, args.lang, "data.jsonl")
        dump_fn = os.path.join(DATA_ROOT, 'datasets', folder, args.lang, "full.jsonl")
        lang1, lang2 = args.lang.split("-")
        
        if not args.rewrite:
            docids = set()
            if os.path.exists(dump_fn):
                for item in load_jsonl_iteratively(dump_fn):
                    docids.add(item['docid'])
        print(f"Writing romanized items to {dump_fn}")

        mode = "w" if args.rewrite else "a"
        outfp = open(dump_fn, mode, encoding="utf8")
        for i, item in tqdm(enumerate(load_jsonl_iteratively(datasets_fn)), desc="Romanizing and Writing"):
            if not args.rewrite and str(i) in docids:
                continue
            
            if lang1 != 'en':
                text = romanize(item[lang1], lang1)
                item[f'{lang1}-roman'] = text
            if lang2 != 'en':
                text = romanize(item[lang2], lang2)
                item[f'{lang2}-roman'] = text
            item['docid'] = str(i)
            
            string = json.dumps(item, ensure_ascii=False)
            outfp.write(f"{string}\n")
    
    
