import os
import json
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

import spacy
import scispacy
from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector
from scispacy.hyponym_detector import HyponymDetector
from scispacy.custom_tokenizer import combined_rule_tokenizer

from utils import load_jsonl_iteratively

def get_ner_by_sent(sent):
    entities = []
    try:
        _doc = nlp(sent)
        for entity in _doc.ents:
            kg_ents = []
            for umls_ent in entity._.kb_ents:
                kg_ents.append({
                    "cui": umls_ent[0],
                    "score": umls_ent[1],
                    "tui": linker.kb.cui_to_entity[umls_ent[0]].types,
                    "aliases": linker.kb.cui_to_entity[umls_ent[0]].aliases
                })
            assert sent[entity.start_char:entity.end_char] == entity.text, "Mismatch in entity text extraction"
            entities.append({
                "text": entity.text,
                "label": entity.label_,
                "start": entity.start_char,
                "end": entity.end_char,
                "ents": kg_ents
            })
    except Exception as e:
        print(f"Error processing sentence: {e}")
        entities = []

    return {"text": sent, "entities": entities}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate NER evaluation dataset for J-STAGE")
    parser.add_argument("--ner_tool", type=str, default="scibert", choices=["scibert", "bionlp"])
    parser.add_argument("--lang", type=str, default="en", choices=["en", "ja"],)
    parser.add_argument("--start_indice", type=int, default=0)
    parser.add_argument("--request_num", type=int, default=None)
    args = parser.parse_args()

    if args.lang == "ja":
        raise NotImplementedError(
            "Japanese NER processing is not implemented in this script. Please use /home/xzhao/workspace/MedNERN-CR-JA/my_predict.py on llm-eval server for processing.")

    print("===> 1. Loading evaluation documents...")
    dump_root = "/data/xzhao/dataset/roman-pretrain/datasets/medical/en_jstage/code-switch"
    dataset_path = "/data/xzhao/dataset/roman-pretrain/datasets/medical/en_jstage/full.jsonl"
    request_num = args.request_num if args.request_num is not None else 9999999
    dump_path = os.path.join(dump_root, f'{args.ner_tool}-{args.start_indice}-{args.start_indice + request_num}.jsonl')
    print("Write processed documents to:", dump_path)
    dump_docs = set()
    if os.path.exists(dump_path):
        dump_docs = set([doc['docid'] for doc in load_jsonl_iteratively(dump_path)])

    print("Dumped docs: ", len(dump_docs))
    print("Start Indice: ", args.start_indice)
    print("Request Number: ", request_num)
    print("NER tools: ", args.ner_tool)

    all_docs = []
    for doc in tqdm(load_jsonl_iteratively(dataset_path, start_indice=args.start_indice, request_num=request_num), "Loading documents"):
        docid = doc["docid"]
        if docid not in dump_docs:
            all_docs.append(doc)
    print(f"Loaded documents: {len(all_docs)}")

        # import importlib
    # import thinc.compat
    # import thinc.util

    # importlib.reload(thinc.compat)
    # importlib.reload(thinc.util)

    
    # spacy.require_gpu()

    print(f"===> 2. Loaded spacy model ...")
    # Enforce GPU usage for spaCy
    sent_spliter = spacy.load("/data/xzhao/spacy_libs/en_core_sci_scibert/en_core_sci_scibert-0.5.4")
    if args.ner_tool == "scibert":
        nlp = spacy.load("/data/xzhao/spacy_libs/en_core_sci_scibert/en_core_sci_scibert-0.5.4")
        nlp.tokenizer = combined_rule_tokenizer(nlp)

        nlp.add_pipe("abbreviation_detector")
        nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        nlp.add_pipe("hyponym_detector", last=True, config={"extended": True})

    elif args.ner_tool == "bionlp":
        nlp = spacy.load("/data/xzhao/spacy_libs/en_ner_bionlp13cg_md/en_ner_bionlp13cg_md-0.5.4")
        nlp.tokenizer = combined_rule_tokenizer(nlp)

        nlp.add_pipe("abbreviation_detector")
        nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        nlp.add_pipe("hyponym_detector", last=True, config={"extended": True})

    else:
        raise ValueError(f"Unsupported NER tool: {args.ner_tool}")
    
    print(f"===> 3. Processing documents ...")
    linker = nlp.get_pipe("scispacy_linker")
    docs = []
    
    start_time = time.time()
    with open(dump_path, 'a', encoding="utf8") as fp:
        for idx, doc in enumerate(all_docs):
            ner_doc = {}
            
            # Process title and abstract
            ner_doc["title"] = get_ner_by_sent(doc["raw"]["title"])
            ner_doc["abstract"] = get_ner_by_sent(doc["raw"]["abstract"])
            ner_doc["subjects"] = [get_ner_by_sent(keyword) for keyword in doc["raw"]["subjects"]]
            ner_doc["keywords"] = [get_ner_by_sent(keyword) for keyword in doc["raw"]["keywords"]]
            ner_doc["sentences"] = [get_ner_by_sent(sent) for sent in doc["raw"]["sentences"]]
            
            if "qa" in doc["raw"]:
                ner_doc["qa"] = []
                for qa_pair in doc["raw"]["qa"]:
                    ner_doc["qa"].append([get_ner_by_sent(sent) for sent in qa_pair])

            if "gmrc" in doc["raw"]:
                ner_doc["gmrc"] = {key: get_ner_by_sent(sent) for key, sent in doc["raw"]["gmrc"].items()}

            if "conclu" in doc["raw"]:
                ner_doc["conclu"] = [get_ner_by_sent(sent) for sent in doc["raw"]["conclu"]]

            if "causal" in doc["raw"]:
                ner_doc["causal"] = {}
                for causal_type, causal_pairs in doc["raw"]["causal"].items():
                    for causal_pair in causal_pairs:
                        ner_pair = [get_ner_by_sent(sent) for sent in causal_pair]
                        ner_doc["causal"].setdefault(causal_type, []).append(ner_pair)
            
            if idx % 100 == 0:
                print(f"Processed {idx+1} documents in {time.time() - start_time:.2f} seconds")
            string = json.dumps({"docid": doc["docid"], "ner": ner_doc}, ensure_ascii=False)
            fp.write(f"{string}\n")
            docs.append(doc)

        print(f"Processed {len(all_docs)} documents in {time.time() - start_time:.2f} seconds")

    # with open(dump_path, 'a', encoding="utf8") as fp:
    #     for item in tqdm(processed_docs, "Writing documents"):
    #         string = json.dumps(item, ensure_ascii=False)
    #         fp.write(f"{string}\n")


    # import multiprocessing as mp
    # mp.set_start_method("spawn", force=True)
    # with ProcessPoolExecutor(max_workers=4) as executor:
    #     futures = []
    #     for i in tqdm(range(0, len(all_docs), 80000), desc="Submitting tasks"):
    #         batch_docs = all_docs[i:i+80000]
    #         futures.append(executor.submit(process_docs, batch_docs, args.ner_tool))
        
    #     docs = []
    #     for future in tqdm(futures, "Waiting for returning the processed documents"):
    #         partial_docs = future.result()
    #         with open(dump_path, 'w', encoding="utf8") as fp:
    #             for item in tqdm(partial_docs, "Writing documents"):
    #                 string = json.dumps(item, ensure_ascii=False)
    #                 fp.write(f"{string}\n")

    



