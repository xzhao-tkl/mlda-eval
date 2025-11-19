import re
import json
import time
import openai
from pydantic import ValidationError

import asyncio
from tqdm import tqdm
from langchain_core.runnables import RunnableConfig

def extract_first_json_block(text: str) -> str:
    """
    Extract the first valid JSON object from a string with nested braces.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No opening brace found in text")

    brace_count = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                return text[start:i+1]
    raise ValueError("Braces do not match, incomplete JSON block")

def robust_parse(output_str, parser):
    """
    Preprocess LLM output string and try to parse it robustly using a LangChain parser.
    """
    cleaned = re.sub(r"^```(?:json)?|```$", "", output_str.strip(), flags=re.MULTILINE).strip()

    # Normalize casing for JSON literals
    cleaned = re.sub(r'\bNULL\b', 'null', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bTRUE\b', 'true', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bFALSE\b', 'false', cleaned, flags=re.IGNORECASE)

    try:
        json_block = extract_first_json_block(cleaned)
    except ValueError as e:
        print("⚠️ Failed to extract valid JSON block")
        raise RuntimeError(str(e))
    
    try:
        return parser.parse(json_block).model_dump()
    except (ValidationError, json.JSONDecodeError) as e:
        print("⚠️  Parser failed after cleanup.")
        raise RuntimeError(f"Robust parsing failed: {e}")


factuality_check_schema = {
    "pattern1": {
        "false": ['fact', 'uality', '":', 'Ġfalse'],
        "true": ['fact', 'uality', '":', 'Ġtrue']
    },
    "pattern2": {
        "false": ['▁"', 'f', 'actual', 'ity', '":', "▁false"],
        "true": ['▁"', 'f', 'actual', 'ity', '":', "▁true"]
    }
}

def tokens_match(tokens, target_array):
    index = -1
    target_length = len(target_array)
    for i in range(len(tokens) - target_length + 1):
        if tokens[i:i + target_length] == target_array:
            return i + target_length - 1
    return index

def measure_prob(response, pattern="pattern1"):
    logprobs = response.response_metadata['logprobs']
    tokens = [item['token'] for item in logprobs['content']]
    probs = [item['logprob'] for item in logprobs['content']]

    match_true = tokens_match(tokens, factuality_check_schema[pattern]['true'])
    match_false = tokens_match(tokens, factuality_check_schema[pattern]['false'])

    if match_true == -1 and match_false == -1:
        raise ValueError("Neither true nor false token arrays matched, which is unexpected.")
    elif match_true != -1 and match_false != -1:
        raise ValueError("Both true and false token arrays matched, which is unexpected.")
    elif match_true != -1:
        prob = probs[match_true]
        return True, prob
    elif match_false != -1:
        prob = probs[match_false]
        return False, prob
    
async def adaptive_batch_process(
        llm, request_items, dump_file, process_request_func,
        start_concurrency=1, max_concurrency=20, step=2,
        failed_log_file=None, do_resample=False):
    global CONCURRENCY
    concurrency = start_concurrency
    responses = []

    queue = asyncio.LifoQueue()
    for item in request_items[::-1]:
        queue.put_nowait(item)
    
    tqdm_bar = tqdm(total=queue.qsize(), desc="Processing requests")
    while not queue.empty():
        size = min(concurrency * 5, queue.qsize())
        current_batch = [queue.get_nowait() for _ in range(size)]
        if not current_batch:
            break
        
        config = RunnableConfig(max_concurrency=concurrency)
        start_time = time.time()
        batch_results = []
        
        failed = 0
        request_ids, messages = zip(*current_batch)
        
        async for idx, response in llm.abatch_as_completed(messages, config=config):
            try:
                new_result = process_request_func(response, messages[idx], request_ids[idx])
                batch_results.append(new_result)
                with open(dump_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(new_result, ensure_ascii=False) + '\n')
                if do_resample:
                    tqdm_bar.update(1)
            except Exception as e:
                print(f"Error parsing response for {request_ids[idx]}: {e}")
                if do_resample:
                    queue.put_nowait((request_ids[idx], messages[idx]))
                failed += 1

                if failed_log_file is not None:
                    with open(failed_log_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({
                            "request_id": request_ids[idx],
                            "error": str(e),
                            "response": response.content
                        }, ensure_ascii=False) + '\n')
            if not do_resample:
                tqdm_bar.update(1)

        duration = time.time() - start_time
        throughput = len(batch_results) / duration
        print(f"Current concurrency: {concurrency}, Processed {len(batch_results)} items in {duration:.2f}s (Throughput: {throughput:.2f} req/s, Failed: {failed} items)")

        
        responses.extend(batch_results)
        prev_throughput = throughput if 'prev_throughput' not in locals() else prev_throughput
        if throughput >= prev_throughput:  # Tune this threshold as needed
            concurrency = min(concurrency + step, max_concurrency)
        else:
            concurrency = max(1, concurrency - step)
        CONCURRENCY = concurrency  # Update global concurrency variable

        prev_throughput = throughput
        await asyncio.sleep(0.1)  # Optional pause to avoid rate limits

    return responses