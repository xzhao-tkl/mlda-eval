import os
import time
import json
from tqdm import tqdm
from pdb import set_trace
from collections import Counter
from utils import load_jsonl_iteratively 


def collect_lemmas():
    if os.path.exists(os.path.join(dataset_root, f"tokenization_merged.full.jsonl")):
        filenames = [os.path.join(dataset_root, f"tokenization_merged.full.jsonl")]
    else:
        filenames = [
            os.path.join(dataset_root, filename) 
            for filename in os.listdir(dataset_root) 
            if filename.startswith(f"tokenization-")]
    
    pos2char = {'NOUN': 'n', 'VERB': 'v', 'ADJ': 'a', 'ADV': 'r'}
    
    def _collect_lemmas(tokenized_sent):
        token_set = Counter()
        for token in tokenized_sent['tokens']:
            if token['pos'] in ['NOUN', 'VERB', 'ADJ', 'ADV']:
                token_set[(token['lemma'], pos2char[token['pos']])] += 1
        return token_set

    lemma_counter = Counter()
    for file_name in filenames:
        file_name = os.path.join(dataset_root, file_name)
        print("Checking ", file_name)
        for doc in tqdm(load_jsonl_iteratively(file_name, request_num=None), desc="Loading lemmas from tokenized documents"):
            doc = doc['token']
            
            lemma_counter.update(_collect_lemmas(doc['title']))
            lemma_counter.update(_collect_lemmas(doc['abstract']))
            for keyword in doc['keywords']:
                lemma_counter.update(_collect_lemmas(keyword))
            for subject in doc['subjects']:
                lemma_counter.update(_collect_lemmas(subject))
            for sent in doc['sentences']:
                lemma_counter.update(_collect_lemmas(sent))

            if 'qa' in doc:
                for qa in doc["qa"]:
                    for sent in qa:
                        lemma_counter.update(_collect_lemmas(sent))
            
            if "conclu" in doc:
                for sent in doc["conclu"]:
                    lemma_counter.update(_collect_lemmas(sent))

            if "gmrc" in doc:
                for gmrc, sent in doc["gmrc"].items():
                    lemma_counter.update(_collect_lemmas(sent))

            if "causal" in doc:
                for causal, sents_ls in doc["causal"].items():
                    for sents in sents_ls:
                        for sent in sents:
                            lemma_counter.update(_collect_lemmas(sent))
    return lemma_counter


lang2wn = {
    'ja': 'omw-ja:1.4',
    'zh': 'omw-cmn:1.4', 
}

def _get_word_from_synset(sense, lang):
    synonyms = set()
    for s in sense.translate(lexicon=lang2wn[lang]):
        synonym = s.word().lemma()
        synonym = synonym.replace('+', '') 
        synonyms.add(synonym)
    return synonyms

def get_all_synonyms(token, synsets, lang, only_first=False, more_than=None):
    syns = set()
    for i in range(len(synsets)):
        for sense in synsets[i].senses():
            if lang != 'en':
                syns.update(_get_word_from_synset(sense, lang))
            else:
                synonym = sense.word().lemma().replace('+', '')
                if synonym != token:
                    syns.add(synonym)

        if only_first and len(syns) > 0:
            break

        if more_than is not None and len(syns) > more_than:
            break
    return syns


def find_synonyms(lemma, pos, lang):
    synsets = wordnet.synsets(lemma, pos=pos)
    if len(synsets) == 0:
        return []
    synonyms = get_all_synonyms(lemma, synsets, lang, only_first=False, more_than=3)
    return synonyms
    
if __name__ == "__main__":
    import wn
    import json
    import argparse

    
    parser = argparse.ArgumentParser(description="Collect CUIs from J-STAGE dataset")
    parser.add_argument("--lang", type=str, default="ja", choices=["ja", "en", "zh"])
    parser.add_argument("--most_common", type=int, default=None)
    args = parser.parse_args()

    wordnet = wn.Wordnet('oewn:2024')
    dataset_root = "/data/xzhao/dataset/roman-pretrain/datasets/medical/en_jstage/wordnet"
    OUT_FILE = os.path.join(dataset_root, f"wordnet_{args.lang}.jsonl")
    print("====> Output file: ", OUT_FILE)

    lemma_counter = collect_lemmas()
    all_lemma_pos_pair = [(lemma, pos) for (lemma, pos), cnt in lemma_counter.most_common(args.most_common)]
    print(f"Collected {len(all_lemma_pos_pair)} CUIs")

    done_lemmas = set()
    if os.path.exists(OUT_FILE):
        for item in load_jsonl_iteratively(OUT_FILE):
            for pos in item['synonyms']:
                done_lemmas.add((item["lemma"], pos))

    request_lemma_pos_pairs = [lemma_pos for lemma_pos in all_lemma_pos_pair if lemma_pos not in done_lemmas]
    print(f"Already done {len(done_lemmas)} lemmas, total {len(request_lemma_pos_pairs)} lemmas to process.")

    print("====> Start processing lemmas...")
    processed_lemmas = {}
    # Open file for append
    start_time = time.time()
    with open(OUT_FILE, "a", encoding="utf-8") as fout:
        for i, (lemma, pos) in enumerate(request_lemma_pos_pairs):
            synonyms = find_synonyms(lemma, pos, lang=args.lang)
            if len(synonyms) == 0:
                continue
            fout.write(
                json.dumps({
                    "lemma": lemma,
                    "pos": pos,
                    "freq": lemma_counter[(lemma, pos)],
                    "synonyms": list(synonyms)
                }, ensure_ascii=False) + "\n")

            if i % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {i}/{len(request_lemma_pos_pairs)} CUIs in {elapsed:.2f} seconds")