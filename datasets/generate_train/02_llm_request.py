import os
import time
import logging
import asyncio
import aiohttp
import argparse
from tqdm import tqdm
import multiprocessing as mp

from utils import DATA_ROOT, load_json, load_jsonl_iteratively

def setup_logging(log_file):
    """Setup logging configuration."""
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def form_request(abs_item, instruction, instruct_type):
    """Formulate the request payload and headers."""
    headers = {
        "Content-Type": "application/json"
    }
    if instruct_type == "qa-gen":
        content = f"{instruction} {abs_item['title']}: {abs_item['abstract']}"
        model_name = "0068_QA-Gen"
    elif instruct_type.endswith("-trans"):
        content = instruction.format(title=abs_item['title'], abstract=abs_item['abstract'])
        model_name = "0068_QA-Gen"
    elif instruct_type == "prompt-gen":
        content = f"{instruction} {abs_item['abstract']}"
        model_name = "0068_QA-Gen"
        
    payload = {
        "request_id": str(abs_item["docid"]),
        "model": model_name,
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": content}
        ]
    }
    return headers, payload


async def send_request(session, endpoint, headers, payload):
    try:
        async with session.post(endpoint, headers=headers, json=payload) as response:
            response.raise_for_status()  # Raise an exception for HTTP errors
            return await response.text()
    except aiohttp.ClientError as e:
        logging.info(f"Request failed: {e}")
        return ""


async def handle_request(session, endpoint, headers, payload, docid, queue):
    response_data = await send_request(session, endpoint, headers, payload)
    await queue.put((docid, response_data))
    

async def request_sender(endpoint, instruct_type, abs_items, instruction, queue, rsp_items):
    async with aiohttp.ClientSession() as session:
        tasks = []
        skip = 0
        for abs_item in abs_items:
            if str(abs_item['docid']) in rsp_items:
                skip += 1
                continue
            if skip > 0:
                logging.info(f"Skip {skip} documents")
            logging.info(f"Sending request {abs_item['docid']}")
            headers, payload = form_request(abs_item, instruction, instruct_type)
            task = asyncio.create_task(handle_request(session, endpoint, headers, payload, abs_item['docid'], queue))
            tasks.append(task)
            await asyncio.sleep(shared_interval.value)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logging.info(f"Task error: {result}")


def response_processor(queue, rsp_fn, shared_interval, num_requests):
    num_response, prev_num = 0, 0
    """Process responses from the queue."""
    rsp_fp = open(rsp_fn, 'a+', encoding="utf8")

    count = 0
    while True:
        if not queue.empty():
            docid, response_data = queue.get()
            if response_data is None:  # Sentinel value to stop the loop
                break
            num_response += 1
            
            if num_response == 10:
                prev_time = time.time()
                prev_num = num_response
            
            if num_response - prev_num == 10:
                per_response = (time.time() - prev_time)/(num_response - prev_num)
                shared_interval.value = min(max(per_response - 0.1, 0), 10)
                prev_time = time.time()
                prev_num = num_response
                logging.info(f"============ Set interval as {shared_interval.value:.2f}s ============")
            if response_data.strip() != "":
                rsp_fp.write(f"{response_data}\n")
            logging.info(f"Receive response for docid {docid}, {num_response} doc processed")
            
            count += 1
            if count % 100 == 0:
                rsp_fp.close()            
                rsp_fp = open(rsp_fn, 'a+', encoding="utf8")
            
        else:
            time.sleep(0.1)  # Sleep briefly to avoid busy-waiting
    rsp_fp.close()

def run_request_sender(endpoint, instruct_type, abs_items, instruction, queue, rsp_items):
    """Run the request sender in an asyncio event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(request_sender(endpoint, instruct_type, abs_items, instruction, queue, rsp_items))
    except Exception as e:
        logging.info(f"Asyncio error: {e}")
    finally:
        queue.put((None, None))  # Sentinel value to signal the end of processing
        loop.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="zh", choices=['zh', 'ja', 'en', 'en_pair'])
    parser.add_argument("--endpoint", type=str, default="http://gpu-node13:8080/v1/chat/completions")
    parser.add_argument("--start_indice", type=int, default=0)
    parser.add_argument("--request_num", type=int, default=100000)
    parser.add_argument("--instruct_type", type=str, choices=["qa-gen", "prompt-gen", "medical-trans", "science-trans"])
    args = parser.parse_args()

    if args.instruct_type == "science-trans" and args.lang != 'zh':
        raise NotImplementedError(f"science-trans is only supported for Chinese, but get language parameter as {args.lang}")
        
    rsp_dir = os.path.join(DATA_ROOT, "response", args.lang)
    os.makedirs(rsp_dir, exist_ok=True)
    
    rsp_fn = os.path.join(rsp_dir, f"{args.instruct_type}.response.s{args.start_indice}e{args.start_indice + args.request_num}.jsonl")
    log_fn = f"../logs/{args.instruct_type}.{args.lang}.s{args.start_indice}e{args.start_indice + args.request_num}.txt"
    setup_logging(log_fn)

    rsp_items = set()
    if os.path.exists(rsp_fn):
        for item in tqdm(load_jsonl_iteratively(rsp_fn)):
            rsp_items.add(item["id"][9:])

    print(f"{len(rsp_items)} docs are processed")
    instruct_lang = "en" if args.lang == "en_pair" else args.lang
    instruction = load_json("./instructions/llm-instructions.json")[args.instruct_type][instruct_lang]
    
    if args.instruct_type != "science-trans":
        abs_items = load_jsonl_iteratively(
            os.path.join(DATA_ROOT, "datasets", "medical", args.lang, f"data.jsonl"), 
            request_num=args.request_num, 
            start_indice=args.start_indice)
    else:
        abs_items = load_jsonl_iteratively(
            os.path.join(DATA_ROOT, "datasets", "scientific_bilingual", "en-zh", f"zh-only.jsonl"), 
            request_num=args.request_num, 
            start_indice=args.start_indice)
    
    # Create a multiprocessing queue
    queue = mp.Queue()

    initial_interval = 1
    shared_interval = mp.Value("d", initial_interval) 
    num_requests = mp.Value("d", 0) 

    # Start the response processor process
    response_process = mp.Process(target=response_processor, args=(queue, rsp_fn, shared_interval, num_requests))
    response_process.start()

    # Start the request sender process
    request_process = mp.Process(target=run_request_sender, args=(args.endpoint, args.instruct_type, abs_items, instruction, queue, rsp_items))
    request_process.start()

    # Wait for both processes to finish
    request_process.join()
    response_process.join()