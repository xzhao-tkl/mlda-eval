from fileinput import filename
import os
from pdb import set_trace
import time
import json
import requests
from tqdm import tqdm
from utils import load_jsonl_iteratively 
from collections import defaultdict, Counter


def collect_cuis(ner_tool):
    """
    Collect CUIs from the dataset and their surface forms.
    """
    assert ner_tool in ['bionlp', 'scibert']
    
    # Load the training dataset
    train_ds_root = f"/data/xzhao/experiments/roman-pretrain/datasets/kg-datasets/en_jstage-0.5/verified_docids.jsonl"
    with open(train_ds_root, "r") as f:
        docids = set([line.strip() for line in f.readlines()])

    cuis = defaultdict()
    seen_docs = set()
    
    if os.path.exists(os.path.join(dataset_root, f"{ner_tool}_merged.full.jsonl")):
        filenames = [os.path.join(dataset_root, f"{ner_tool}_merged.full.jsonl")]
    else:
        filenames = [
            os.path.join(dataset_root, filename) 
            for filename in os.listdir(dataset_root) 
            if filename.startswith(f"{ner_tool}-")]
        
    def _collect_cuis(nered_sent):
        for ent in nered_sent['entities']:
            if ent['ents'] == []: 
                continue
            for cui in ent['ents']:    
                if cui['cui'] not in cuis:
                    cuis[cui['cui']] = {
                        "freq": 1, 
                        "surface": {ent['text']},
                    }
                else:
                    cuis[cui['cui']]['freq'] += 1
                    cuis[cui['cui']]['surface'].add(ent['text'])

    for file_name in filenames:
        file_name = os.path.join(dataset_root, file_name)
        print("Checking ", file_name)
        for doc in tqdm(load_jsonl_iteratively(file_name)):
            if doc['docid'] not in docids and doc['docid'] not in seen_docs:
                continue
            seen_docs.add(doc['docid'])
            doc = doc['ner']
            
            _collect_cuis(doc['title'])
            _collect_cuis(doc['abstract'])
            for keyword in doc['keywords']:
                _collect_cuis(keyword)
            for subject in doc['subjects']:
                _collect_cuis(subject)
            for sent in doc['sentences']:
                _collect_cuis(sent)

            if 'qa' in doc:
                for qa in doc["qa"]:
                    for sent in qa:
                        _collect_cuis(sent)
            
            if "conclu" in doc:
                for sent in doc["conclu"]:
                    _collect_cuis(sent)
                    
            if "gmrc" in doc:
                for gmrc, sent in doc["gmrc"].items():
                    _collect_cuis(sent)
            
            if "causal" in doc:
                for causal, sents_ls in doc["causal"].items():
                    for sents in sents_ls:
                        for sent in sents:
                            _collect_cuis(sent)
            
            
    cui_counter = Counter()
    for cui, info in cuis.items():
        cui_counter[cui] = info['freq']
    return cui_counter


def fetch_translations(cui):
    if LANG != "ALL":
        url = f"https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}/atoms?language={LANG}"
    else:
        url = f"https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}/atoms?pageSize=1000"
    params = {
        "apiKey": API_KEY
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                data = r.json()
                item_trans = {}
                for atom in data.get("result", []):
                    item_trans.setdefault(atom["language"], []).append(atom["name"])
                return item_trans
            elif r.status_code == 429:  # Too many requests
                wait = 5 * attempt
                print(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            elif r.status_code == 404:
                # print(f"No Japanese atoms for cui: {cui}. HTTP {r.status_code} for {cui}: {r.text}")
                return []
            else:
                print(f"HTTP {r.status_code} for {cui}: {r.text}")
                return []
        except requests.RequestException as e:
            print(f"Error for {cui} (attempt {attempt}): {e}")
            time.sleep(2 * attempt)
    return []

if __name__ == "__main__":
    import json
    import argparse

    
    parser = argparse.ArgumentParser(description="Collect CUIs from J-STAGE dataset")
    parser.add_argument("--lang", type=str, default="ja", choices=["ja", "en", "all"])
    parser.add_argument("--ner_tool", type=str, default="scibert", choices=["scibert", "bionlp"])
    parser.add_argument("--most_common", type=int, default=None)
    parser.add_argument("--backup_file", type=str, default=None, help="Path to the backup file that may contain processed cuis")
    parser.add_argument("--api_key", type=str, default="default", choices=["default", "new"], help="API key for accessing external services")
    args = parser.parse_args()


    if args.lang == "ja":    
        LANG = "JPN" 
    elif args.lang == "en":
        LANG = "ENG"
    elif args.lang == "all":
        LANG = "ALL"

    dataset_root = "/data/xzhao/dataset/roman-pretrain/datasets/medical/en_jstage/code-switch"
    OUT_FILE = os.path.join(dataset_root, f"cui_{args.lang}-{args.ner_tool}.jsonl")
    print("====> Output file: ", OUT_FILE)

    cui_counter = collect_cuis(args.ner_tool)
    all_cuis = [cui for cui, _ in cui_counter.most_common(args.most_common)]
    print(f"Collected {len(all_cuis)} CUIs")

    if args.api_key == "default":
        API_KEY = "<API_KEY1>"
    elif args.api_key == "new":
        API_KEY = "<API_KEY2>"
        
    SLEEP_BETWEEN_REQUESTS = 0.05  # to avoid hitting API too fast
    MAX_RETRIES = 3

    # Read CUIs
    done_cuis = set()
    if os.path.exists(OUT_FILE):
        for item in load_jsonl_iteratively(OUT_FILE):
            done_cuis.add(item["cui"])
    request_cuis = [cui for cui in all_cuis if cui not in done_cuis]
    print(f"Already done {len(done_cuis)} CUIs, total {len(request_cuis)} CUIs to process.")

    print("====> Start processing CUIs...")
    processed_cuis = {}
    if args.backup_file is not None and os.path.exists(args.backup_file):
        print("Load processed CUIs from backup file...")
        processed_cuis = {item["cui"]: item for item in load_jsonl_iteratively(args.backup_file)}
        print(f"Loaded {len(processed_cuis)} processed CUIs from backup file.")

    # Open file for append
    start_time = time.time()
    with open(OUT_FILE, "a", encoding="utf-8") as fout:
        for i, cui in enumerate(request_cuis, 1):
            if cui in done_cuis:
                continue
                
            if cui in processed_cuis:
                translations = processed_cuis[cui]["translations"]
            else:
                translations = fetch_translations(cui)
            fout.write(json.dumps({
                    "cui": cui, 
                    "freq": cui_counter[cui],
                    "translations": translations
                }, ensure_ascii=False) + "\n")
            fout.flush()

            if i % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {i}/{len(request_cuis)} CUIs in {elapsed:.2f} seconds")
            time.sleep(SLEEP_BETWEEN_REQUESTS)