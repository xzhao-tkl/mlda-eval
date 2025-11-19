from pdb import set_trace
import os
import math
from tqdm import tqdm 
import json
import requests
from utils import load_json

from agent_utils import robust_parse
from utils import EXP_ROOT, dump_jsonl, load_jsonl, load_jsonl_iteratively

from langchain.output_parsers import PydanticOutputParser

def extract_triples(lang="en"):
    eval_qa_root = os.path.join(EXP_ROOT, "datasets/kg-datasets/ja-0.5/eval_qa")
    if lang == "ja":
        docs = load_jsonl(os.path.join(eval_qa_root, "01_fact_check/ja_ners.jsonl"))
    elif lang == "en":
        docs = load_jsonl(os.path.join(eval_qa_root, "01_fact_check/bionlp_ners.jsonl"))
    docs = {doc['docid']: doc for doc in docs}
    model2triples = {}

    if lang == "ja":
        models = ["Qwen3-32B", "llm-jp-3.1", "Swallow-70B"]
    elif lang == "en":
        models = ["Qwen3-32B", "Llama3.3-70B", "DeepSeek-R1-70B"]
    
    for model in models:
        items = {}
        for item in load_jsonl_iteratively(os.path.join(eval_qa_root, f"02_triple_extract/{lang}_{model}.jsonl")):
            items[item['request_id']] = item
        model2triples[model] = items

    for model in models:
        request_ids = set(model2triples[model].keys())
        merged_ids = request_ids if "merged_ids" not in locals() else merged_ids.intersection(request_ids)

    print(f"Number of merged request_ids: {len(merged_ids)}")

    probs = []
    fact_scores = {}
    for request_id in merged_ids:
        fact_score = 0
        true_prob = 0
        
        for model in model2triples:
            prob = math.exp(model2triples[model][request_id]['logprob'])
            factuality = model2triples[model][request_id]['factuality']
            if factuality:
                fact_score += 1
                true_prob += prob

        fact_scores[request_id] = (fact_score, true_prob)
        probs.append(true_prob)
    sorted_reqids = sorted(fact_scores, key=lambda x: fact_scores[x][1], reverse=True)

    def entity_bimatch(target_entity, cand_entities):
        for entity in cand_entities:
            if entity in target_entity or target_entity in entity:
                return True
        return False

    extracted_triples = []

    if lang == "en":
        for req_id in tqdm(sorted_reqids):
            if fact_scores[req_id][0] <= 2:
                continue
            docid, sentid = req_id.split("_sentid:")
            # print(f"\n-------- Document ID: {docid}, Sentence ID: {sentid} --------")
            triples = []
            for model in model2triples:
                triple = model2triples[model][req_id]['triple']
                if triple is None:
                    continue
                subj_cuis = []
                obj_cuis = []

                try:
                    for entity in docs[docid]['sentences'][int(sentid)]['entities']:
                        ent_aliases = [entity['text'].strip().lower()]
                        for ent in entity['ents']:
                            if ent['score']  < 0.6:
                                continue        
                            ent_aliases.extend([alias.strip().lower() for alias in ent["aliases"]])
                        
                        # if not isinstance(triple['subject'], str) or not isinstance(triple['object'], str):
                        #     print(f"Skipping triple with non-string subject or object: {triple}")
                        #     continue

                            if entity_bimatch(triple['subject'].strip().lower(), ent_aliases):
                                subj_cuis.extend([(ent['cui'], entity['text']) for ent in entity['ents']])
                            if entity_bimatch(triple['object'].strip().lower(), ent_aliases):
                                obj_cuis.extend([(ent['cui'], entity['text']) for ent in entity['ents']])
                    
                    triples.append({
                        "model": model,
                        "subj": triple['subject'],
                        "obj": triple['object'],
                        "relation": model2triples[model][req_id]['triple']['relation'],
                        "subj_cuis": subj_cuis,
                        "obj_cuis": obj_cuis,
                        "prob": math.exp(model2triples[model][req_id]['logprob'])
                    })
                except Exception as e:
                    print(f"Error processing entity matching: {e}")
                    continue    
                
            # print(f"subj: {subj}, matched CUIs: {subj_cuis}")
            # print(f"obj: {obj}, matched CUIs: {obj_cuis}")
            item = {
                "request_id": req_id,
                "docid": docid,
                "sentid": sentid,
                "abstract": docs[docid]['abstract'],
                "sentence": docs[docid]['sentences'][int(sentid)]['text'],
                "triple": triples,
            }
            extracted_triples.append(item)
    elif lang == "ja":
        for req_id in tqdm(sorted_reqids):
            if fact_scores[req_id][0] <= 2:
                continue
            
            docid, sentid = req_id.split("_sentid:")
            triples = []
            for model in model2triples:
                triple = model2triples[model][req_id]['triple']
                if triple is None:
                    continue
                try:
                    triples.append({
                        "model": model,
                        "subj": triple['subject'],
                        "obj": triple['object'],
                        "relation": model2triples[model][req_id]['triple']['relation'],
                        "prob": math.exp(model2triples[model][req_id]['logprob'])
                    })
                except Exception as e:
                    continue
            
                
            item = {
                "request_id": req_id,
                "docid": docid,
                "sentid": sentid,
                "abstract": docs[docid]['abstract'],
                "sentence": docs[docid]['sentences'][int(sentid)]['text'],
                "triple": triples,
            }
            extracted_triples.append(item)
    return extracted_triples

def generate_requests(extracted_triples, lang, dump_file):
    if os.path.exists(dump_file) and os.path.getsize(dump_file) > 0:
        print(f"Dump file {dump_file} already exists, skipping generation.")
        return load_jsonl(dump_file)
    print(f"Generating requests for {dump_file}...")
    
    import spacy

    instructions = load_json("./instructions.json")
    system_prompt = instructions["generate-qa"][lang]['system']
    user_template = instructions["generate-qa"][lang]['user']

    openai_requests = []
    nlp = spacy.load("en_core_web_sm")

    def count_match_word_ratio(sentence_lemmas, entity_lemmas):
        entity_lemmas = set(entity_lemmas)
        match_count = len(entity_lemmas.intersection(sentence_lemmas))
        return match_count / len(entity_lemmas) if entity_lemmas else 0

    for request_item in tqdm(extracted_triples):
        if lang == "en":
            triples = request_item['triple']
            cand_triples = []
            for triple in triples:
                triple['subj_lemmas'] = [token.lemma_ for token in nlp(triple['subj'])]
                triple['obj_lemmas'] = [token.lemma_ for token in nlp(triple['obj'])]
                all_lemmas = triple['subj_lemmas'] + triple['obj_lemmas']
                sentence_lemmas = [token.lemma_ for token in nlp(request_item['sentence'])]
                match_ratio = count_match_word_ratio(sentence_lemmas, all_lemmas)
                triple['match_ratio'] = match_ratio
                if match_ratio < 0.5:
                    continue
                cand_triples.append(triple)
            
            if len(cand_triples) == 0:
                cand_triples = triples

            # print(f"\n========== Request ID: {extract_triples[i]['request_id']} =========")
            min_len = 99999
            for item in cand_triples:
                length = len(item['subj_lemmas']) + len(item['obj_lemmas'])
                length = length * (1/item['match_ratio'])
                if length < min_len:
                    min_len = length
                    triple = item
            request_item['best_triple'] = triple    
            input_dict = {
                "sentences": request_item['sentence'],
                "triple": {
                    "subject": request_item['best_triple']['subj'],
                    "relation": request_item['best_triple']['relation'],
                    "object": request_item['best_triple']['obj']
                }
            }
        elif lang == "ja":
            for item in request_item['triple']:
                if item["model"] == "Qwen3-32B":
                    best_triple = item
                    break

            input_dict = {
                "sentences": request_item['sentence'],
                "triple": {
                    "subject": best_triple['subj'],
                    "relation": best_triple['relation'],
                    "object": best_triple['obj']
                }
            }

        user_prompt = user_template.format(input=json.dumps(input_dict, ensure_ascii=False, indent=4))
        # request = {
        #     "custom_id": request_item['request_id'],
        #     "method": "POST",
        #     "url": "/chat/completions",
        #     "body": {
        #         "model": os.getenv("AZURE_MODEL_NAME"), 
        #         "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], 
        #         "max_tokens": 1024,
        #         "temperature": 0.5,
        #         "top_p": 0.9,
        #     }}
        # batch_requests.append(request)
        openai_requests.append({
            "request_id": request_item['request_id'],
            "message": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "metadata": {
                "input": input_dict,
                "docid": request_item['docid'],
                "sentid": request_item['sentid'],
                "abstract": request_item['abstract'],
                "sentence": request_item['sentence']
            }})
    
    dump_jsonl(openai_requests, dump_file)
    return openai_requests


from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class QAGenExtract(BaseModel):
    triple: Dict[str, str] = Field(
        description="The factual knowledge triple in JSON format with keys: subject, relation, object"
    )
    fill_in_blank: str = Field(
        description="Generated fill-in-the-blank query for the given triple and sentence"
    )
    answer: str = Field(
        description="The correct answer to the fill-in-the-blank question, which is the object of the triple"
    )
    distractors: List[str] = Field(
        description="List of distractors that are plausible but incorrect answers to the question and corrent object"
    )
    question: str = Field(
        description="Question paraphrased from the original sentence, asking for the object of the triple"
    )

def parser_qa_generation(response, message, request_id):
    assert request_id in id2request, f"Request ID {request_id} not found in id2request."
    generation = robust_parse(response.content, parser)
    assert "[BLANK]" in generation["fill_in_blank"]
    assert "subject" in generation["triple"] and "relation" in generation["triple"] and "object" in generation["triple"]
    assert len(generation["distractors"]) == 3, f"Distractors length is not 3: {generation['distractors']}"
    assert len(generation["triple"]) == 3, f"Triple does not have 3 elements: {generation['triple']}"
    result = {
        'request_id': request_id, 
        'message': message,
        "generation": generation,
        "metadata": id2request[request_id]['metadata']}
    return result

if __name__ == "__main__":
    import os
    import time
    import asyncio
    from langchain_openai import AzureChatOpenAI
    from agent_utils import adaptive_batch_process

    import argparse
    aug_parser = argparse.ArgumentParser(description="Generate QA pairs from triples.")
    aug_parser.add_argument("--lang", type=str, choices=["en", "ja"], required=True, help="Language for the QA generation.")
    aug_parser.add_argument("--num_requests", type=int, default=-1, help="Number of requests to process.")
    aug_parser.add_argument("--request_file", type=str, default=None, help="Path to the request file. If provided, will use this file instead of generating requests from triples.")
    aug_parser.add_argument("--response_file", type=str, default=None, help="Path to the response file. If provided, will use this file instead of generating requests from triples.")
    args = aug_parser.parse_args()

    parser = PydanticOutputParser(pydantic_object=QAGenExtract, include_raw=True, strict=False)
    qa_root = os.path.join(EXP_ROOT, "datasets/kg-datasets/ja-0.5/eval_qa/03_en_qa")
    os.makedirs(qa_root, exist_ok=True)

    if not args.request_file:
        extracted_triples = extract_triples(lang=args.lang)
        all_requests = generate_requests(extracted_triples, lang=args.lang, dump_file=os.path.join(qa_root, f"{args.lang}_requests.jsonl"))
    else:
        request_file = os.path.join(qa_root, args.request_file)
        print(f"Using provided request file: {request_file}")
        all_requests = load_jsonl(request_file)
    id2request = {item['request_id']: item for item in all_requests}

    os.environ["AZURE_MODEL_NAME"] = "gpt-4.1-2025-04-14"
    os.environ["OPENAI_API_TYPE"] = "azure_ad"
    os.environ["OPENAI_API_VERSION"] = "2025-04-01-preview"
    os.environ["AZURE_OPENAI_API_KEY"] = "<API_KEY>"
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://llm-jp-openai-mmwg-01.openai.azure.com"

    CONCURRENCY = 1

    llm = AzureChatOpenAI(
            deployment_name=os.environ["AZURE_MODEL_NAME"],
            temperature=0,
            top_p=1,
            max_retries=1,
            timeout=120,
            logprobs=True,
            max_tokens=1024,
        )
    
    # filter_requestid = [
    #     "fpj1944@@92/1/92_1_17_sentid:2",
    #     "jfsc@@125/0/125_498_sentid:0",
    #     "chemotherapy1995@@44/4/44_4_227_sentid:0",
    #     "jjsa@@82/11/82_1976_sentid:1",
    #     "tennenyuki@@56/0/56_Poster23_sentid:1",
    #     "rikusui1931@@58/4/58_4_349_sentid:7",
    #     "jvma1951@@38/2/38_2_108_sentid:2",
    #     "jpan@@20/1/20_22_sentid:1",
    #     "jjspc1994@@14/4/14_4_393_sentid:3",
    #     "gee1973b@@28/4/28_4_685_sentid:5",
    #     "gee1973b@@50/2/50_2_230_sentid:3",
    #     "jjpn@@28/2/28_oa.2015.0069_sentid:1",
    #     "nskkk@@67/11/67_430_sentid:8",
    #     "chemotherapy1995@@43/1/43_1_12_sentid:2",
    #     "jvss@@2019/0/2019_3Hp03_sentid:1",
    #     "jjsnr@@28/5/28_20050802003_sentid:2",
    #     "tigg1989@@14/76/14_76_87_sentid:0"
    # ]
    while True:
        generated_reqids = []
        if args.response_file:
            response_path = os.path.join(qa_root, args.response_file)
            print(f"Using provided response file: {response_path}")
            if os.path.exists(response_path):
                raise ValueError(f"Response file {response_path} already exists. Please remove it or specify a different file.")
        else:
            response_path = os.path.join(qa_root, f"{args.lang}_generation.jsonl")
        print(f"Response file will be saved to: {response_path}")
        
        if os.path.exists(response_path):
            generated = load_jsonl(response_path)
            for generated_item in generated:
                generated_reqids.append(generated_item['request_id'])
        generated_reqids = set(generated_reqids)
        
        # all_requests = [req for req in all_requests if req['request_id'] in filter_requestid]
        if args.num_requests > 0:
            all_requests = all_requests[:args.num_requests]

        print(f"=== Total requests to process: {len(all_requests)}")
        print(f"=== The number of processed requests: {len(generated_reqids)}")
        
        requests = []
        for request in all_requests:
            if request["request_id"] in generated_reqids:
                continue
            requests.append((request["request_id"], request['message']))

        if len(requests) == 0:
            print("=== All requests have been processed. Exiting.")
            break
        
        print(f"=== Number of requests to process: {len(requests)}")
        try:
            print("=== Starting processing requests with concurrency:", CONCURRENCY)
            asyncio.run(
                adaptive_batch_process(
                    llm, requests, 
                    dump_file=response_path, 
                    process_request_func=parser_qa_generation,
                    start_concurrency=1, max_concurrency=5, step=1,
                    failed_log_file=os.path.join(qa_root, f"failed_{args.lang}_generation_requests.log"),
                    do_resample=False)
            )
        except Exception as e:
            print(f"Error during processing: {e}")
            print("Retrying in 10 seconds...")
            time.sleep(10)
            continue