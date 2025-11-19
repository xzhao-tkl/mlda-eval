import re
import nltk
from itertools import product


# Summarizaion
# Text Completion
# Keywords extraction
# Keyword to sentence
# Conclusion
# Romanization
# Casual effect
# Question-answering
# GMRC


gmrc_keywords = {
    "en": {
        "goal": ["Objective", "Objectives", "Purpose", "Purposes", "Background", "Backgrounds", "Aim", "Aims"],
        "method": ["Method", "Methods", "Patients and Methods"],
        "result": ["Result", "Results"],
        "conclu": ["Conclusion", "Conclusions",]
    },
    "zh": {
        "goal": ["目的：", "目的:", "目的 ", "目　的", "目的", "背景: ", "背景 "],
        "method": ["方法：", "方法:", "方法 ", "方 法", "方法", "対象和方法"],
        "result": ["结果：", "结果:", "结果 ", "结 果", "结果", "结果与结论"],
        "conclu": ["结论：", "结论:", "结论 ", "结 论", "结论"]
    },
    "ja": {
        "goal": ["目的", "背景", "目　的"],
        "method": ["方法","方　法","対象と方法", ],
        "result": ["実験結果", "実験　結果", "結果", "結　果", ],
        "conclu": ["結論", "考察", "結　論", "考　察","結果、結論", "結果,結論", "まとめ"]
    }
}


def generate_gmrc_pattern(keywords):
    se_formats = []
    for keyword in keywords:
        formats = [
            fr"\[\s*{keyword}\s*\]", fr"\[\s*{keyword}\s*\]:", fr"【\s*{keyword}\s*】",
            fr"\(\s*{keyword}\s*\)", fr"（\s*{keyword}\s*）", fr"「\s*{keyword}\s*」",
            fr"{keyword}:", fr"{keyword}：", fr"{keyword}　", fr"{keyword} "
        ]
        se_formats.extend(formats)
    se_pattern = "|".join(se_formats)
    return se_pattern

def match_gmrc_advers(text, lang):
    gmrc_matches = {}
    start_indices, gmrc_paras = [], []
    gmrc_keys = gmrc_keywords[lang]
    for para, keywords in gmrc_keys.items():
        pattern = generate_gmrc_pattern(keywords)
        compiled_pattern = re.compile(pattern)
        matches = [match.start() for match in compiled_pattern.finditer(text)]
        if len(matches) == 1:    
            start_indices.append(matches[0])
            gmrc_paras.append(para)
    
    start_indices.append(-1)
    for i, para in enumerate(gmrc_paras):
        gmrc_matches[para] = text[start_indices[i]: start_indices[i+1]]
    return gmrc_matches


import hanlp
split_sent = hanlp.load(hanlp.pretrained.eos.UD_CTB_EOS_MUL)
def split_document(document, lang='en'):
    if lang == 'en' or lang == 'en_jstage':
        sentences = nltk.tokenize.sent_tokenize(document)
    elif lang == 'ja' or lang == 'zh':
        sentences = split_sent(document)
    return sentences


conclu_suffixs = [",", "、", "，", "、", ":", "：", ":", "：", " ", "　"]
conclusion_ja = [
    "以上の結果から", "以上の結果に基づき", "以上の結果", "これらの結果から",
    "以上の考察から", "以上の考察に基づき", "以上の考察", "これらの考察から",
    "本研究を通して", "本研究をとおして",
    "それに基づいた考察において", "結論として", "結論について",
    "結論", "考察", "結　論", "考　察","結果、結論", "結果,結論", "まとめ",
    "【結論】", "【考察】", "【結　論】", "【考　察】","【結果、結論】", "【結果,結論】", "【まとめ】",
    "（結論）", "（考察）", "（結　論）", "（考　察）","（結果、結論）", "（結果,結論）", "（まとめ）",
]

conclusion_zh = ["结论"]

conclusion_en = [
    "In conclusion: ", "In conclusion, ", 
    "To conclude: ", "To conclude, ",
    "Conclusion: ", "Conclusion, ", 
    "As a result, ", "As the result, ",
    "As a result: ", "As the result: ",
    "Major results are ", "Major results are ", 
    "As a result of the study, ", "As a result of this study, ", "As the result of this study, ",
    "Viewpoints of this study are ", "This study reveals that ", "By way of conclusion, ", 
    "The following findings are obtained", "Here is the conclusion, ",
    "Here is the result, ", "Here are the results, ", 
    "Here are the conclusions, ", "We conclude that ", 
    "In short, ", "In short: ", 
    "As one of the conclusions, ", "The main conclusion is, ",
]

conclusion_ja = [f"{middle}{suffix}" for middle, suffix in product(conclusion_ja, conclu_suffixs)]
conclusion_zh = [f"{middle}{suffix}" for middle, suffix in product(conclusion_zh, conclu_suffixs)]
conclusion_keywords = {
    "en": conclusion_en,
    "ja": conclusion_ja,
    "zh": conclusion_zh,
}

def match_conclusion_advers(text, lang):
    keywords = conclusion_keywords[lang]
    escaped_keywords = [re.escape(keyword) for keyword in keywords]
    pattern = r"|".join(escaped_keywords)    
    compiled_pattern = re.compile(pattern, re.IGNORECASE)
    matches = [(match.group(), match.start(), match.end()) for match in compiled_pattern.finditer(text) if match.start() > 0]
    if len(matches) >= 1:
        start = matches[0][1]
        return text[:start], text[start:]
    return None


casual_suffixs = [",", "、", "，", "、", ":", "：", ":", "：", " ", "　"]
casual_keywords = {
    "zh": {
        "entail": ["所以", "因此", "相应地", "因而", "于是", "由此可见", "从而", "这样一来", "结果", "最终"],
        "contradict": ["与此相反地", "与此相反", "但是", "然而", "不过", "尽管如此", "却", "反倒", "反而", "而", "虽说"],
        "neutral": ['除此之外', "此外", "另外", "同时", "而且", "并且", "不仅如此", "更重要的是", "值得一提的是", "除此之外还"]
    },
    "ja": {
        "entail": [
            "したがって", "そのため", "ゆえに", "よって", "これにより", "以上の所見より",
            "この結果より", "この結果により", "こうして", "結論として", "総じて言うと", "総じて", "必然的に", "これらの結果より"
            "調査の結果", "以上の結果より", "その結果", "以上のことから", "分析の結果", "この結果",
            "このことから", "以上より", "以上から", "原因として", "この実験から", "この事実より", "結論"],
        "contradict": ["しかしながら", "それに対して", "しかし", "一方で", "とはいえ", "それにもかかわらず", "ところが", "むしろ", "逆に", "対照的に", "とは反対に"],
        "neutral": [
            "加えて述べると", "加えて考慮すると", "加えて言えば", 
            "加えて", "さらに", "また", "同時に", "併せて", "一方",
            "付け加えると", "言い換えると", "なお"]
    },
    "en": {
        "entail": [
            "so", "therefore", "thus", "hence", "as a result", "consequently", 
            "accordingly", "because of this", "for this reason", "this means that", 
            "which implies", "resulting in", "leading to", "it follows that"
        ],
        "contradict": [
            "however", "but", "on the contrary", "in contrast", "conversely", 
            "nevertheless", "nonetheless", "yet", "despite this", "although", 
            "even though", "whereas", "while", "on the other hand", "instead"
        ],
        "neutral": [
            "additionally", "furthermore", "moreover", "also", "in addition", 
            "besides", "as well as", "not only", "apart from this", "similarly", 
            "likewise", "at the same time", "meanwhile", "in the meantime", 
            "on top of that", "what's more"
        ]
    },
}

def match_casuals(sentences, lang):
    if lang == 'zh' or lang == 'ja':
        return match_casuals_zh_ja(sentences, lang)
    elif lang == 'en':
        return match_casuals_en(sentences, 'en')

def match_casuals_zh_ja(sentences, lang):
    casual = {}
    keywords = casual_keywords[lang]
    for rel, keys in keywords.items():
        for i, sent in enumerate(sentences[1:]):
            for key in keys:
                if sent.lower().startswith(key):
                    sent = sent[len(key):]
                    if any([sent.lower().startswith(suffix) for suffix in casual_suffixs]):
                        sent = sent[1:].strip()
                    casual.setdefault(rel, []).append([sentences[i], sent])
                    break
    return casual


def split_sentence_by_keyword(sentence, keywords):
    """
    Split a sentence into two parts based on a keyword.
    """
    for keyword in keywords:
        for comma in [", ", ","]:
            keyword = comma + keyword
            if keyword in sentence.lower():
                parts = re.split(fr"\b{keyword}\b", sentence, flags=re.IGNORECASE)
                if len(parts) > 1:
                    return parts[0].strip(), parts[1].strip()
    return None, None

def match_casuals_en(sentences, lang):
    casual = {}
    keywords = casual_keywords[lang]

    # Check for relationships within a single sentence
    for sent in sentences:
        for rel, keys in keywords.items():
            cause, effect = split_sentence_by_keyword(sent, keys)
            if cause and effect:
                if any([effect.lower().startswith(suffix) for suffix in casual_suffixs]):
                    effect = effect[1:].strip()                    
                casual.setdefault(rel, []).append([cause, effect])
                
    # Check for relationships between adjacent sentences
    for i in range(len(sentences) - 1):
        for rel, keys in keywords.items():
            for key in keys:
                if sentences[i + 1].lower().startswith(key):
                    effect = sentences[i + 1][len(key):].lstrip()
                    if any([effect.lower().startswith(suffix) for suffix in casual_suffixs]):
                        effect = effect[1:].strip()
                    casual.setdefault(rel, []).append([sentences[i], effect])
                    break

    return casual

import os
import json
from tqdm import tqdm
from utils import DATA_ROOT, load_jsonl_iteratively

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="zh", choices=['zh', 'ja', 'en', 'en_jstage'])
    args = parser.parse_args()

    out_fn = os.path.join(DATA_ROOT, "datasets", "medical", args.lang, f"preprocessed.jsonl")
    items = load_jsonl_iteratively(os.path.join(DATA_ROOT, "datasets", "medical", args.lang, f"data.jsonl"))

    all_items = []
    num_gmrc = 0

    with open(out_fn, 'w', encoding="utf8") as fp:
        for i, item in tqdm(enumerate(items)):
            sents = split_document(item['abstract'], args.lang)
            item["sentences"] = sents

            lang_key = "en" if args.lang.startswith("en") else args.lang
            sents = item["sentences"]
            casuals = match_casuals(sents, lang_key)
            if casuals != {}:
                item["causal"] = casuals
                
            gmrc_matches = match_gmrc_advers(item['abstract'], lang_key)
            conclu_matches = match_conclusion_advers(item['abstract'], lang_key)
            
            if conclu_matches is not None:
                item["conclu"] = conclu_matches
            
            if len(gmrc_matches) > 1:
                item["gmrc"] = gmrc_matches

            fp.write(f"{json.dumps(item, ensure_ascii=False)}\n")





