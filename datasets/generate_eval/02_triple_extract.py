from functools import partial
import os
import time
import asyncio
import tokenize
from tqdm import tqdm
from collections import Counter

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

from agent_utils import robust_parse, measure_prob, adaptive_batch_process
from utils import EXP_ROOT, load_jsonl, load_json, load_jsonl_iteratively

CONCURRENCY = 5

def get_requests(predumped_file, lang, repeated_request=1):
    """Generator function to yield sentences with more than 2 entities."""
    request_counter = Counter()

    if os.path.exists(predumped_file):
        for item in tqdm(load_jsonl_iteratively(predumped_file, start_indice=0), desc="Loading dumped sentences"):
            request_counter[item["request_id"]] += 1

    request_ids = set()
    for doc in docs:
        if "sentences" not in doc:
            continue
        
        for i, sentence in enumerate(doc["sentences"]):
            entities = []
            for entity in sentence["entities"]:
                if lang == "en":
                    ents.add(entity['text'].strip().lower())
                    for ent in entity["ents"]:
                        match = False
                        for tui in ent["tui"]:
                            if tui in filter_tuis:
                                continue
                            match = True
                        if match:
                            entities.append(ent)
                elif lang == "ja":
                    ent_type = entity["type"]
                    if ent_type.startswith("time") or ent_type.startswith("cc_") or ent_type.startswith("t-val_") or ent_type.startswith("t-key_other") or ent_type.startswith("f_"):
                        continue
                    entities.append(entity)
            if len(entities) <= 2:
                continue
            request_id = f"{doc['docid']}_sentid:{i}"
            assert request_id not in request_ids, f"Duplicate request_id found: {request_id}"
            request_ids.add(request_id)
            if request_counter[request_id] >= repeated_request:
                continue
            
            dumped_cnt = request_counter[request_id] if request_id in request_counter else 0
            for _ in range(repeated_request - dumped_cnt):
                yield request_id, sentence['text']
    
# rate_limiter = InMemoryRateLimiter(requests_per_second=10, check_every_n_seconds=1, max_bucket_size=10)

from typing import Optional
from pydantic import BaseModel, Field

class EnTripleExtract(BaseModel):
    factuality: bool = Field(
        description="A Boolean value indicating whether the sentence contains generalizable triple-like factual knowledge."
    )
    triple: Optional[dict] = Field(
        description="A nested JSON object with three fields: subject, relation, and object representing the extracted fact. Please note that if factuality is false, make sure this field is null."
    )
    reason: str = Field(
        description="A brief explanation for why the sentence was or wasn't considered factual, referring to the criteria provided."
    )

class JaTripleExtract(BaseModel):
    factuality: bool = Field(
        description="文が一般化可能なトリプル形式の事実的知識を含むかどうかを示す真偽値。"
    )
    triple: Optional[dict] = Field(
        description="抽出された事実を表す主語、関係、目的語の3つのフィールドを持つネストされたJSONオブジェクト。factualityがfalseの場合、このフィールドはnullであること。"
    )
    reason: str = Field(
        description="文が事実的と判断された理由、またはそうでなかった理由を、指示された基準に基づいて簡潔に説明する文字列。"
    )

# structured_llm = llm.with_structured_output(method="json_mode")
# structured_llm = llm.with_structured_output(schema=EnTripleExtract, method="json_schema")
# structured_llm = llm.with_structured_output(schema=EnTripleExtract, method="json_schema", include_raw=True, strict=False)

def parser_factuality_check(response, message, request_id, tokenize_pattern):
    result = robust_parse(response.content, parser)
    factuality, prob = measure_prob(response, pattern=tokenize_pattern)
    assert factuality == result['factuality'], f"Factuality mismatch between prob-based and output-based output: {factuality} != {result['factuality']} for request {request_id}"
    new_result = {'request_id': request_id, "logprob": prob}
    new_result.update(result)
    new_result['request'] = message
    return new_result

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="Filter factual sentences from a dataset.")
    parser.add_argument("--lang", type=str, required=True, choices=["en", "ja"])
    parser.add_argument(
        "--model", type=str, required=True, 
        choices=["Qwen3-32B", "Llama3.3-70B", "DeepSeek-R1-70B", "llm-jp-3.1", "Swallow-70B"],)
    parser.add_argument(
        "--node", type=str, required=True,
        help="The node to use for the LLM API, e.g., 'gpu-node3'.")
    parser.add_argument("--temperature", type=float, default=0.0, help="The temperature for the LLM API, default is 0.0.")
    parser.add_argument("--port", type=int, default=8080, help="The port for the LLM API, default is 8080." )
    parser.add_argument("--reverse", action="store_true",
                        help="If set, the script will reverse the order of processing requests.")
    args = parser.parse_args()
    
    eval_qa_root = os.path.join(EXP_ROOT, "datasets/kg-datasets/ja-0.5/eval_qa")
    
    if args.lang == "en":
        doc_path = os.path.join(eval_qa_root, "01_fact_check/bionlp_ners.jsonl") 
    elif args.lang == "ja":
        doc_path = os.path.join(eval_qa_root, "01_fact_check/ja_ners.jsonl") 
    docs = load_jsonl(doc_path)


    filter_tuis = ["T073", "T098", "T097"]
    cuis = set()
    ents = set()

    if args.model == "llm-jp-3.1":
        tokenize_pattern = "pattern2"
    else:
        tokenize_pattern = "pattern1"

    print(f"=== Using tokenize pattern: {tokenize_pattern} for model {args.model}")

    llm = ChatOpenAI(
        model="0068_QA-Gen",
        openai_api_key="",
        openai_api_base=f"http://{args.node}:{args.port}/v1",
        temperature=args.temperature,
        top_p=1,
        max_retries=1,
        timeout=120,
        logprobs=True,
        max_tokens=1024,
    )

    instructions = load_json("./instructions.json")
    system_prompt = instructions["factuality-check"][args.lang]['system']
    user_prompt = instructions["factuality-check"][args.lang]['user']
    TripleExtract = EnTripleExtract if args.lang == "en" else JaTripleExtract

    parser = PydanticOutputParser(pydantic_object=TripleExtract, include_raw=True, strict=False)
    template = PromptTemplate(
        template=f'{system_prompt}\n{user_prompt}',
        input_variables=["sentence"],
        partial_variables={"schema_instruction": parser.get_format_instructions()},
    )

    dump_folder = os.path.join(eval_qa_root, "02_triple_extract")
    os.makedirs(dump_folder, exist_ok=True)
    if args.reverse:
        dump_file = os.path.join(dump_folder, f"{args.lang}_{args.model}-reverse.jsonl")
    else:
        dump_file = os.path.join(dump_folder, f"{args.lang}_{args.model}.jsonl")

    print(f"=== Dumping results to {dump_file}")
    failed_log_file = os.path.join(dump_folder, f"failed_requests_{args.model}.log")

    while True:
        request_items = []
        for request_id, sentence in get_requests(dump_file, lang=args.lang, repeated_request=1):
            request_items.append((request_id, template.format(sentence=sentence)))
        
        if args.reverse:
            request_items.reverse()
        
        if len(request_items) == 0:
            print("No requests to process, exiting.")
            break

        print(f"=== Total requests to process: {len(request_items)}")
        if os.path.exists(dump_file):
            print(f"=== The number of processed requests: {sum(1 for _ in open(dump_file))}")

        try:
            print("=== Starting processing requests with concurrency:", CONCURRENCY)
            asyncio.run(
                adaptive_batch_process(
                    llm, request_items, dump_file, 
                    process_request_func=partial(parser_factuality_check, tokenize_pattern=tokenize_pattern),
                    start_concurrency=CONCURRENCY, max_concurrency=20, step=2,
                    failed_log_file=failed_log_file, do_resample=False)
            )
        except Exception as e:
            print(f"Error during processing: {e}")
            print("Retrying in 10 seconds...")
            time.sleep(10)
            continue
        

