import os
import json
import time
import spacy
from tqdm import tqdm
from utils import load_jsonl_iteratively

def get_token_by_sent(sent):
    tokens = []
    try:
        _doc = nlp(sent)
        for token in _doc:
            tokens.append({
                "text": token.text,
                "start": token.idx,
                "end": token.idx + len(token),
                "lemma": token.lemma_,
                "pos": token.pos_,
            })
    except Exception as e:
        print(f"Error processing sentence: {sent}")
        print(e)
    return {"text": sent, "tokens": tokens}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate NER evaluation dataset for J-STAGE")
    parser.add_argument("--start_indice", type=int, default=0)
    parser.add_argument("--request_num", type=int, default=None)
    
    args = parser.parse_args()

    print("===> 1. Loading evaluation documents...")
    dump_root = "/data/xzhao/dataset/roman-pretrain/datasets/medical/en_jstage/tokenization"
    dataset_path = "/data/xzhao/dataset/roman-pretrain/datasets/medical/en_jstage/full.jsonl"
    os.makedirs(dump_root, exist_ok=True)
    # dump_path = os.path.join(dump_root, f'tokenization.jsonl')
    request_num = args.request_num if args.request_num is not None else 9999999
    dump_path = os.path.join(dump_root, f'tokenization-{args.start_indice}-{args.start_indice + request_num}.jsonl')
    print("Write processed documents to:", dump_path)
    dump_docs = set()
    if os.path.exists(dump_path):
        dump_docs = set([doc['docid'] for doc in load_jsonl_iteratively(dump_path)])

    print("Dumped docs: ", len(dump_docs))
    
    all_docs = []
    for doc in tqdm(load_jsonl_iteratively(dataset_path, start_indice=args.start_indice, request_num=request_num), "Loading documents"):
        docid = doc["docid"]
        if docid not in dump_docs:
            all_docs.append(doc)
    print(f"Loaded documents: {len(all_docs)}")
    print(f"===> 2. Loaded spacy model ...")
    # nlp = spacy.load("en_core_web_sm")
    nlp = spacy.load("/data/xzhao/spacy_libs/en_core_sci_scibert/en_core_sci_scibert-0.5.4")
    print(f"===> 3. Processing documents ...")
    docs = []
    
    start_time = time.time()
    with open(dump_path, 'a', encoding="utf8") as fp:
        for idx, doc in enumerate(all_docs):
            tokenized_doc = {}
            
            # Process title and abstract
            tokenized_doc["title"] = get_token_by_sent(doc["raw"]["title"])
            tokenized_doc["abstract"] = get_token_by_sent(doc["raw"]["abstract"])
            tokenized_doc["subjects"] = [get_token_by_sent(keyword) for keyword in doc["raw"]["subjects"]]
            tokenized_doc["keywords"] = [get_token_by_sent(keyword) for keyword in doc["raw"]["keywords"]]
            tokenized_doc["sentences"] = [get_token_by_sent(sent) for sent in doc["raw"]["sentences"]]
            
            if "qa" in doc["raw"]:
                tokenized_doc["qa"] = []
                for qa_pair in doc["raw"]["qa"]:
                    tokenized_doc["qa"].append([get_token_by_sent(sent) for sent in qa_pair])

            if "gmrc" in doc["raw"]:
                tokenized_doc["gmrc"] = {key: get_token_by_sent(sent) for key, sent in doc["raw"]["gmrc"].items()}

            if "conclu" in doc["raw"]:
                tokenized_doc["conclu"] = [get_token_by_sent(sent) for sent in doc["raw"]["conclu"]]

            if "causal" in doc["raw"]:
                tokenized_doc["causal"] = {}
                for causal_type, causal_pairs in doc["raw"]["causal"].items():
                    for causal_pair in causal_pairs:
                        ner_pair = [get_token_by_sent(sent) for sent in causal_pair]
                        tokenized_doc["causal"].setdefault(causal_type, []).append(ner_pair)
            
            if idx % 100 == 0:
                print(f"Processed {idx+1} documents in {time.time() - start_time:.2f} seconds")
            string = json.dumps({"docid": doc["docid"], "token": tokenized_doc}, ensure_ascii=False)
            fp.write(f"{string}\n")
            docs.append(doc)

        print(f"Processed {len(all_docs)} documents in {time.time() - start_time:.2f} seconds")