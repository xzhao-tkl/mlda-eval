from tqdm import tqdm
from utils import dump_jsonl, load_jsonl, load_jsonl_iteratively

import nltk
import spacy
import scispacy
from spacy import displacy

from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector
from scispacy.hyponym_detector import HyponymDetector
from scispacy.custom_tokenizer import combined_rule_tokenizer


def process_docs(sent_spliter, nlp, eval_docs):
    linker = nlp.get_pipe("scispacy_linker")
    docs = []
    for doc in tqdm(eval_docs):
        data = doc["abstract"]
        sentences = [sent.text for sent in sent_spliter(data).sents]
        sent_info = []
        for sent in sentences:
            entities = []
            _doc = nlp(sent)
            
            abbreviation = []
            for abrv in _doc._.abbreviations:
                abbreviation.append({
                    "text": abrv.text,
                    "start": abrv.start_char,
                    "end": abrv.end_char,
                    "long_form": abrv._.long_form.text
                })
            
            for entity in _doc.ents:
                kg_ents = []
                if entity.label_ == "ORGANISM":
                    continue
        
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

            sent_info.append({
                "text": sent,
                "entities": entities,
                "abbreviations": abbreviation
            })
        doc["sentences"] = sent_info
        docs.append(doc)
    
    return docs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate NER evaluation dataset for J-STAGE")
    parser.add_argument("--ner_tool", type=str, default="scispacy", choices=["scibert", "bionlp"])
    parser.add_argument("--lang", type=str, default="en", choices=["en", "ja"],)
    args = parser.parse_args()


    if args.lang == "ja":
        raise NotImplementedError(
            "Japanese NER processing is not implemented in this script. Please use /home/xzhao/workspace/MedNERN-CR-JA/my_predict.py on llm-eval server for processing.")

    print("===> 1. Loading evaluation documents...")
    jstage_root = "/data/xzhao/experiments/roman-pretrain/datasets/kg-datasets/ja-0.5/eval_qa/01_fact_check"
    docid_path = f"{jstage_root}/docids.jsonl"
    dataset_path = "/data/xzhao/dataset/roman-pretrain/datasets/medical/en_jstage/data.jsonl"
    eval_docids = set(load_jsonl(docid_path))

    eval_docs = []
    for doc in tqdm(load_jsonl_iteratively(dataset_path)):
        docid = doc["docid"]
        if docid in eval_docids:
            eval_docs.append(doc)
    assert len(eval_docs) == len(eval_docids), "Mismatch in number of documents"

    print(f"===> 2. Loaded spacy model ...")
    spacy.prefer_gpu()

    sent_spliter = spacy.load("/data/xzhao/spacy_libs/en_core_sci_scibert/en_core_sci_scibert-0.5.4")
    if args.ner_tool == "scibert":
        nlp = spacy.load("/data/xzhao/spacy_libs/en_core_sci_scibert/en_core_sci_scibert-0.5.4")
        nlp.tokenizer = combined_rule_tokenizer(nlp)

        nlp.add_pipe("abbreviation_detector")
        nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        nlp.add_pipe("hyponym_detector", last=True, config={"extended": True})

        dump_path = f"{jstage_root}/scibert_ners.jsonl"
    elif args.ner_tool == "bionlp":
        nlp = spacy.load("/data/xzhao/spacy_libs/en_ner_bionlp13cg_md/en_ner_bionlp13cg_md-0.5.4")
        nlp.tokenizer = combined_rule_tokenizer(nlp)

        nlp.add_pipe("abbreviation_detector")
        nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        nlp.add_pipe("hyponym_detector", last=True, config={"extended": True})

        dump_path = f"{jstage_root}/bionlp_ners.jsonl"
    else:
        raise ValueError(f"Unsupported NER tool: {args.ner_tool}")
    
    docs = process_docs(sent_spliter, nlp, eval_docs)
    dump_jsonl(docs, dump_path)




