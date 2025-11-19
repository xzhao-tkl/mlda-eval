import os
import json
from pdb import set_trace
from typing import Dict, List
from tqdm import tqdm
from pydantic import BaseModel, Field
from agent_utils import robust_parse
from langchain.output_parsers import PydanticOutputParser
from utils import EXP_ROOT, dump_jsonl, load_json, load_jsonl, load_jsonl_iteratively

class InLangFilterExtract(BaseModel):
    explanations: Dict[str, str] = Field(
        description="Provide brief explanations for each of your assessments above."
    )
    fill_in_blank_quality: str = Field(
        description="Quality of the fill-in-the-blank questions. Answer with 'Good' or 'Poor'."
    )
    paraphrase_quality: str = Field(
        description="Quality of the question paraphrase. Answer with 'Good' or 'Poor'."
    )
    options_quality: str = Field(
        description="Quality of the answer and distractors. Answer with 'Good' or 'Poor'."
    )

class CrossLangFilterExtract(BaseModel):
    explanation: str = Field(
        description="Provide a brief explanation for each of your assessments above."
    )
    associated_sentences: List[str] = Field(
        description="List of sentences associated with the question. Provide all relevant sentences."
    )
    score: int = Field(
        description="Overall score for the question and answer pair based on the context. Answer with an integer from 1 to 5, where 1 is poor and 5 is excellent."
    )


def parser_inlang_filtering_generation(response, message, request_id):
    assert request_id in id2request, f"Request ID {request_id} not found in id2request."
    generation = robust_parse(response.content, inlang_parser)
    result = {
        'request_id': request_id, 
        'message': message,
        "generation": generation,
    }
    return result

def parser_crosslang_filtering_generation(response, message, request_id):
    assert request_id in id2request, f"Request ID {request_id} not found in id2request."
    generation = robust_parse(response.content, crosslang_parser)
    result = {
        'request_id': request_id, 
        'message': message,
        "generation": generation,
    }
    return result


def load_qa_instance(lang, data_root):
    data_path = os.path.join(data_root, f"{lang}_generation.jsonl")
    assert os.path.exists(data_path), f"Data path {data_path} does not exist."
    instances = []
    for generation in load_jsonl_iteratively(data_path):
        instances.append({
            "request_id": generation['request_id'],
            "sentence": generation['metadata']["sentence"],
            "triple": generation['generation']["triple"],
            "fill_in_blank": generation['generation']['fill_in_blank'],
            "question": generation['generation']['question'],
            "answer": generation['generation']['answer'],
            "distractors": generation['generation']["distractors"],
        })
    return instances

def generate_requests(instances, lang, dump_file):
    """ Generate OpenAI requests for the given instances and language. """
    # if os.path.exists(dump_file) and os.path.getsize(dump_file) > 0:
    #     print(f"Dump file {dump_file} already exists, skipping generation.")
    #     return load_jsonl(dump_file)
    print(f"Generating requests for {dump_file}...")
    
    instructions = load_json("./instructions.json")
    system_prompt = instructions["inlang-filter"][lang]['system']
    user_template = instructions["inlang-filter"][lang]['user']

    openai_requests = []
    for request_item in tqdm(instances):
        input_dict = {
            "sentence": request_item['sentence'],
            "triple": request_item['triple'],
            "fill_in_blank": request_item['fill_in_blank'],
            "question": request_item['question'],
            "answer": request_item['answer'],
            "distractors": request_item['distractors'],
        }
        user_prompt = user_template.format(input=json.dumps(input_dict, ensure_ascii=False, indent=4))
        openai_requests.append({
            "request_id": request_item['request_id'],
            "message": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ], 
            "metadata": request_item
        })

    dump_jsonl(openai_requests, dump_file)
    return openai_requests

def load_crosslang_instances(lang, data_root, filtered_reqids):
    ner_root = os.path.join(EXP_ROOT, "datasets/kg-datasets/ja-0.5/eval_qa/01_fact_check")
    crosslang_ner_path = os.path.join(ner_root, "bionlp_ners.jsonl") if lang == "ja" else os.path.join(ner_root, "ja_ners.jsonl")
    
    tgtlang_instances = {}
    for item in load_jsonl(crosslang_ner_path):
        tgtlang_instances[item['docid']] = item['abstract']

    inlang_instances = load_jsonl(os.path.join(data_root, f"{lang}_generation.jsonl"))
    instances = []
    for item in inlang_instances:
        if item['request_id'] not in filtered_reqids:
            continue
        assert item['metadata']['docid'] in tgtlang_instances, f"Missing docid {item['docid']} in en_docs"
        instances.append({
            "docid": item['metadata']['docid'],
            "crosslang-context": tgtlang_instances[item['metadata']['docid']],
            "request_id": item['request_id'],
            "sentence": item['metadata']["sentence"],
            "question": item['generation']['question'],
            "triple": item['generation']["triple"],
            "answer": item['generation']['answer'],
        })
    return instances

def generate_crosslang_requests(instances, lang, dump_file):
    """ Generate OpenAI requests for bilingual filtering. """

    if os.path.exists(dump_file) and os.path.getsize(dump_file) > 0:
        print(f"Dump file {dump_file} already exists, skipping generation.")
        return load_jsonl(dump_file)
    print(f"Generating bilingual filtering requests for {dump_file}...")
    
    instructions = load_json("./instructions.json")
    system_prompt = instructions["crosslang-filter"][lang]['system']
    user_template = instructions["crosslang-filter"][lang]['user']

    openai_requests = []
    for request_item in tqdm(instances):
        if lang == "en":
            input_dict = {
                "ja-context": request_item['crosslang-context'],
                "en-sentence": request_item['sentence'],
                "en-triple": request_item['triple'],
                "en-question": request_item['question'],
                "en-answer": request_item['answer'],
            }
        elif lang == "ja":
            input_dict = {
                "en-context": request_item['crosslang-context'],
                "ja-sentence": request_item['sentence'],
                "ja-triple": request_item['triple'],
                "ja-question": request_item['question'],
                "ja-answer": request_item['answer'],
            }
        else:
            raise ValueError(f"Unsupported language: {lang}")
        user_prompt = user_template.format(input=json.dumps(input_dict, ensure_ascii=False, indent=4))
        openai_requests.append({
            "request_id": request_item['request_id'],
            "message": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ], 
            "metadata": request_item
        })
    dump_jsonl(openai_requests, dump_file)
    return openai_requests


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
    aug_parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    aug_parser.add_argument("--crosslang", action="store_true", help="Enable cross-language filtering mode.")

    args = aug_parser.parse_args()

    inlang_parser = PydanticOutputParser(pydantic_object=InLangFilterExtract, include_raw=True, strict=False)
    crosslang_parser = PydanticOutputParser(pydantic_object=CrossLangFilterExtract, include_raw=True, strict=False)
    dump_root = os.path.join(EXP_ROOT, "datasets/kg-datasets/ja-0.5/eval_qa/04_filtering")
    os.makedirs(dump_root, exist_ok=True)
    data_root = os.path.join(EXP_ROOT, "datasets/kg-datasets/ja-0.5/eval_qa/03_en_qa")
    assert os.path.exists(data_root), f"Data root {data_root} does not exist."

    if not args.crosslang:
        instances = load_qa_instance(lang=args.lang, data_root=data_root)
        all_requests = generate_requests(
            instances, lang=args.lang, 
            dump_file=os.path.join(dump_root, f"{args.lang}_requests.jsonl"))
    else:
        inlang_generations = load_jsonl(os.path.join(dump_root, f"{args.lang}_generation.jsonl"))
        filtered_reqids = set([
            item["request_id"] 
            for item in inlang_generations 
            if item['generation']['fill_in_blank_quality'] == "Good" \
                and item['generation']['paraphrase_quality'] == "Good" \
                and item['generation']['options_quality'] == "Good"])
        print(f"=== The number of filtered request IDs from {args.lang}: {len(filtered_reqids)}")
        instances = load_crosslang_instances(lang=args.lang, data_root=data_root, filtered_reqids=filtered_reqids)
        dump_jsonl(instances, os.path.join(dump_root, f"{args.lang}_crosslang_instances.jsonl"))
        # raise Exception("Stop here for crosslang instance generation.")
        all_requests = generate_crosslang_requests(
            instances, lang=args.lang, 
            dump_file=os.path.join(dump_root, f"{args.lang}_requests.crosslang.jsonl"))
    
    # raise Exception("Stop here for crosslang instance generation.")
    if not args.crosslang:
        response_path = os.path.join(dump_root, f"{args.lang}_generation.jsonl")
    else:
        response_path = os.path.join(dump_root, f"{args.lang}_generation.crosslang.jsonl")
    
    
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
            logprobs=False,
            max_tokens=4096,
        )
    
    while True:
        generated_reqids = []
        print(f"Response file will be saved to: {response_path}")
        
        if os.path.exists(response_path):
            generated = load_jsonl(response_path)
            for generated_item in generated:
                generated_reqids.append(generated_item['request_id'])
        generated_reqids = set(generated_reqids)
        
        if args.num_requests > 0:
            all_requests = all_requests[:args.num_requests]

        print(f"=== Total requests to process: {len(all_requests)}")
        print(f"=== The number of processed requests: {len(generated_reqids)}")
        
        requests = []
        for request in all_requests:
            if request["request_id"] in generated_reqids:
                continue
            if args.debug: 
                import pandas as pd
                df = pd.read_csv(f"./annotation/{args.lang}_evaluation.csv")
                cand_ids = df['request_id'].unique().tolist()
                if request["request_id"] not in cand_ids:
                    continue
            requests.append((request["request_id"], request['message']))

        if len(requests) == 0:
            print("=== All requests have been processed. Exiting.")
            break
        
        print(f"=== Number of requests to process: {len(requests)}")
        try:
            parser_filtering_generation = parser_inlang_filtering_generation if not args.crosslang else parser_crosslang_filtering_generation
            print("=== Starting processing requests with concurrency:", CONCURRENCY)
            asyncio.run(
                adaptive_batch_process(
                    llm, requests, 
                    dump_file=response_path,
                    process_request_func=parser_filtering_generation,
                    start_concurrency=1, max_concurrency=3, step=1,
                    failed_log_file=os.path.join(dump_root, f"failed_{args.lang}_generation_requests.log"),
                    do_resample=False)
            )
        except Exception as e:
            print(f"Error during processing: {e}")
            print("Retrying in 10 seconds...")
            set_trace()
            time.sleep(10)
            continue