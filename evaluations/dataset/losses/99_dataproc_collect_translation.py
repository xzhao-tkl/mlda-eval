import os

from cv2 import error
from utils import load_jsonl, load_jsonl_iteratively

en_instruction = """### Instruction: 
You are an expert in biomedical natural language processing and Japanese scientific translation.
You will be given a list of sentences, in order, from a biomedical research abstract.
For each sentence, generate five different versions according to the following rules, while strictly preserving the biomedical meaning.
Note that certain sentences such as punctuation marks and numerical dates cannot be translated or paraphrased and should be copied directly to fields.

### Rules for Each Version
1. syntactical
    - Only change syntax (word order, clause structure, voice, etc.).
    - Do not replace words with synonyms unless it is strictly required for grammaticality.
    - Keep all biomedical terminology unchanged (e.g., “apoptosis”, “metastasis”, “mTOR pathway”).
2. lexical
    - Keep syntax as close as possible to the original.
    - Replace content words (verbs, nouns, adjectives) with suitable synonyms or equivalent biomedical expressions.
    - Maintain accuracy of biomedical terminology: Use widely accepted synonyms from standard biomedical lexicons (e.g., UMLS, MeSH, JMedTerm). Do not use informal terms for specialized concepts.
3. semantical
    - Change both syntax and vocabulary to express the same meaning in a different way.
    - You may merge or split clauses for clarity.
    - Ensure biomedical concepts remain exactly equivalent — no meaning shift, no over-generalization.
4. word_aligned_translation
    - Translate into Japanese word-by-word, preserving the original sentence structure as much as possible.
    - Retain biomedical loanwords (e.g., “apoptosis” → アポトーシス) in katakana when common in Japanese biomedical writing.
    - Keep technical precision even if the result is less natural.
5. flexible_translation
    - Translate into natural, fluent Japanese suitable for a professional biomedical paper.
    - Reorder elements for Japanese readability.
    - Replace foreign biomedical terms with their standard Japanese equivalents when they exist in established usage.

### Formatting Rules
- Output a JSON array.
- Each input sentence corresponds to one JSON object with the keys:
    - "original": The original sentence in English.
    - "syntactical": The syntactically modified version.
    - "lexical": The lexically modified version.
    - "semantical": The semantically modified version.
    - "word_aligned_translation": The strict translation version.
    - "flexible_translation": The flexible translation version.    
- Maintain the input order exactly.
- All output strings must be plain text without markdown formatting.

### Input Sentences:
```json
[
    "Add next sentences to the following text.", 
    "The Joint United Nations Programme on HIV/AIDS (USAIDS) reports that an estimated 370,000 children contracted HIV during the perinatal and breastfeeding period in 2009.", 
    "In Japan, there are 51 documented cases of mother-to-child transmission of HIV. ", 
    "We saw a 5-year-old boy who was introduced for recurrent otitis media with vertical HIV transmission."
]
```

### Output:
```json
[
    {
        "original": "Add next sentences to the following text.",
        "syntactical": "Add the following sentences to the text.",
        "vocabulary": "Insert the subsequent sentences into the given passage.",
        "semantical": "Include the upcoming sentences in the text provided.",
        "word_aligned_translation": "次の文を以下の文章に追加してください。",
        "flexible_translation": "以下の文章に続けて文を追加してください。"
    },
    {
        "original": "The Joint United Nations Programme on HIV/AIDS (USAIDS) reports that an estimated 370,000 children contracted HIV during the perinatal and breastfeeding period in 2009.",
        "syntactical": "According to the Joint United Nations Programme on HIV/AIDS (USAIDS), an estimated 370,000 children contracted HIV during the perinatal and breastfeeding period in 2009.",
        "vocabulary": "The Joint United Nations Programme on HIV/AIDS (USAIDS) states that about 370,000 children acquired HIV in the perinatal and lactation stages in 2009.",
        "semantical": "In 2009, approximately 370,000 children were infected with HIV during birth or breastfeeding, according to the Joint United Nations Programme on HIV/AIDS (USAIDS).",
        "word_aligned_translation": "国連合同エイズ計画（USAIDS）は、報告する推定370,000子どもが周産期と授乳期の期間にHIVに感染した2009年。",
        "flexible_translation": "国連合同エイズ計画（USAIDS）の報告によると、2009年には約37万人の子どもが周産期または授乳期にHIVに感染した。"
    },
    {
        "original": "In Japan, there are 51 documented cases of mother-to-child transmission of HIV.",
        "syntactical": "There are 51 documented cases of mother-to-child transmission of HIV in Japan.",
        "vocabulary": "Japan has 51 recorded instances of vertical HIV transmission.",
        "semantical": "In Japan, 51 cases have been officially reported where HIV was transmitted from mother to child.",
        "word_aligned_translation": "日本では、あります51記録された症例の母子感染のHIV。",
        "flexible_translation": "日本ではHIVの母子感染が51例報告されている。"
    },
    {
        "original": "We saw a 5-year-old boy who was introduced for recurrent otitis media with vertical HIV transmission.",
        "syntactical": "A 5-year-old boy with vertical HIV transmission was introduced for recurrent otitis media.",
        "vocabulary": "We examined a 5-year-old male patient with repeated otitis media and congenital HIV infection.",
        "semantical": "We treated a 5-year-old boy who had HIV from birth and suffered from repeated middle ear infections.",
        "word_aligned_translation": "私たちは見た5歳の男児が再発性中耳炎のために紹介された垂直HIV伝播",
        "flexible_translation": "私たちは、生まれつきHIVに感染しており、繰り返す中耳炎のため紹介された5歳の男児を診察した。"
    }
]
"""

ja_instruction = """### Instruction: 
You are an expert in biomedical natural language processing and English scientific translation.
You will be given a list of sentences, in order, from a biomedical research abstract written in Japanese.
For each sentence, generate five different versions according to the following rules, while strictly preserving the biomedical meaning.
Note that certain sentences such as punctuation marks and numerical dates cannot be translated or paraphrased and should be copied directly to fields.

### Rules for Each Version
1. syntactical
    - Only change syntax (word order, clause structure, voice, etc.).
    - Do not replace words with synonyms unless it is strictly required for grammaticality.
    - Keep all biomedical terminology unchanged (e.g., “apoptosis”, “metastasis”, “mTOR pathway”).
2. lexical
    - Keep syntax as close as possible to the original.
    - Replace content words (verbs, nouns, adjectives) with suitable synonyms or equivalent biomedical expressions.
    - Maintain accuracy of biomedical terminology: Use widely accepted synonyms from standard biomedical lexicons (e.g., UMLS, MeSH, JMedTerm). Do not use informal terms for specialized concepts.
3. semantical
    - Change both syntax and vocabulary to express the same meaning in a different way.
    - You may merge or split clauses for clarity.
    - Ensure biomedical concepts remain exactly equivalent — no meaning shift, no over-generalization.
4. word_aligned_translation
    - Translate the Japanese sentence into English word-by-word while preserving the original word order as much as possible.
    - Retain Japanese biomedical loanwords in transliteration when they are common in English biomedical writing.
    - Maintain technical precision even if the result is ungrammatical in English.
5. flexible_translation
    - Translate into natural, fluent English suitable for a professional biomedical paper.
    - Reorder elements for English readability.
    - Replace Japanese biomedical terms with their standard English equivalents when they exist in established usage.

### Formatting Rules
- Output a JSON array.
- Each input sentence corresponds to one JSON object with the keys:
    - "original": The original sentence in Japanese.
    - "syntactical": The syntactically modified version.
    - "lexical": The lexically modified version.
    - "semantical": The semantically modified version.
    - "word_aligned_translation": The strict translation version.
    - "flexible_translation": The flexible translation version.    
- Maintain the input order exactly.
- All output strings must be plain text without markdown formatting.

### Input Sentences:
```json
[
    "以下の文章に続けて文を追加してください。",
    "日本ではHIVの母子感染が51例報告されている。",
    "2009年には、周産期および授乳期に約37万人の子どもがHIVに感染したと国連合同エイズ計画が報告している。",
    "我々は、生まれつきHIVに感染しており、繰り返す中耳炎のため紹介された5歳の男児を診察した。"
]
```

### Output:
```json
[
    {
        "original": "以下の文章に続けて文を追加してください。",
        "syntactical": "以下の文章に文を続けて追加してください。",
        "lexical": "下の文章に次の文を挿入してください。",
        "semantical": "このテキストの末尾に次の文を付け加えてください。",
        "word_aligned_translation": "below 's text to continuing sentence add please .",
        "flexible_translation": "Please add the following sentences to the text below."
    },
    {
        "original": "日本ではHIVの母子感染が51例報告されている。",
        "syntactical": "HIVの母子感染が日本では51例報告されている。",
        "lexical": "日本国内ではHIVの垂直感染が51件記録されている。",
        "semantical": "日本では母親から子へのHIV感染が51例確認されている。",    
        "word_aligned_translation": "In Japan HIV's mother-child transmission 51 cases have been reported.",
        "flexible_translation": "There are 51 reported cases of mother-to-child transmission of HIV in Japan."
    },
    {
        "original": "2009年には、周産期および授乳期に約37万人の子どもがHIVに感染したと国連合同エイズ計画が報告している。",
        "syntactical": "国連合同エイズ計画は、2009年に周産期および授乳期に約37万人の子どもがHIVに感染したと報告している。",
        "lexical": "2009年には、周産期および授乳期に約37万人の児童がHIVに罹患したと国連合同エイズ計画が報告している。",
        "semantical": "国連合同エイズ計画は、2009年に出生時または授乳期を通じて約37万人の子どもがHIVに感染したと述べている。",
        "word_aligned_translation": "In 2009, perinatal period and breastfeeding period in about 370,000 children HIV to infected that United Nations Joint AIDS Programme reports.",
        "flexible_translation": "In 2009, about 370,000 children became infected with HIV during the perinatal or breastfeeding periods, according to the Joint United Nations Programme on HIV/AIDS."
    },
    {
        "original": "我々は、生まれつきHIVに感染しており、繰り返す中耳炎のため紹介された5歳の男児を診察した。",
        "syntactical": "我々は、繰り返す中耳炎のため紹介された、生まれつきHIVに感染している5歳の男児を診察した。",
        "lexical": "私たちは、先天的にHIVに感染しており、再発性の中耳炎で紹介された5歳の男児を診察した。",
        "semantical": "生まれつきHIV陽性で、頻回に中耳炎を起こすため紹介された5歳男児を我々は診察した。",
        "word_aligned_translation": "We, from birth HIV to infected, recurring otitis media for referred 5 years old boy examined.",
        "flexible_translation": "We examined a 5-year-old boy who was born with HIV and was referred for recurrent middle ear infections."
    }
]
```

### Input:
```json
{input}
```

### Output:
"""

user_prompt = """### Input:
```json
{input}
```

### Output:
"""

import re
from pydantic import ValidationError

def extract_first_json_block(text: str) -> str:
    """
    Extract the first valid JSON array (list of dictionaries) from a string with nested brackets/braces.
    """
    start = text.find("[")
    if start == -1:
        raise ValueError("No opening bracket '[' found in text")

    bracket_count = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            bracket_count += 1
        elif text[i] == "]":
            bracket_count -= 1
            if bracket_count == 0:
                return text[start:i+1]
    raise ValueError("Brackets do not match, incomplete JSON array block")

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

from typing import List
from pydantic import BaseModel, Field, RootModel
from langchain.output_parsers import PydanticOutputParser

class SentRewriteObject(BaseModel):
    original: str = Field(..., description="The original sentence in the source language.")
    syntactical: str = Field(..., description="Syntactically modified version of the original sentence.")
    lexical: str = Field(..., description="Lexically modified version of the original sentence.")
    semantical: str = Field(..., description="Semantically modified version of the original sentence.")
    word_aligned_translation: str = Field(..., description="Strict translation version aligned with the original syntax.")
    flexible_translation: str = Field(..., description="Flexible translation version suitable for professional writing.")

class SentRewriteObjectList(RootModel[List[SentRewriteObject]]):
    pass

import asyncio
import time
import json
from tqdm import tqdm
from langchain_core.runnables import RunnableConfig

async def adaptive_batch_process(
        llm, requests, dump_file, error_doc_path=None,
        start_concurrency=1, max_concurrency=20, step=2,
        failed_log_file=None, do_resample=False):
    global CONCURRENCY
    concurrency = start_concurrency
    responses = []
    queue = asyncio.LifoQueue()
    for request in requests[::-1]:
        queue.put_nowait(request)

    if error_doc_path is not None and not os.path.exists(error_doc_path):
        with open(error_doc_path, 'w', encoding='utf-8') as f:
            pass
    
    tqdm_bar = tqdm(total=queue.qsize(), desc="Processing requests")
    while not queue.empty():
        error_docs = []
        if error_doc_path is not None:
            with open(error_doc_path, 'r', encoding='utf-8') as f:
                error_docs = [int(doc.strip()) for doc in f.readlines()]
                print(f"Loaded {len(error_docs)} error documents:")

        size = min(concurrency, queue.qsize())
        current_batch = [queue.get_nowait() for _ in range(size)]
        current_batch = [item for item in current_batch if item['request_id'] not in error_docs]
        if not current_batch:
            break
        
        config = RunnableConfig(max_concurrency=concurrency)
        start_time = time.time()
        batch_results = []
        
        failed = 0
        messages = [item['message'] for item in current_batch]
        request_ids = [item['request_id'] for item in current_batch]
        async for idx, response in llm.abatch_as_completed(messages, config=config):
            try:
                new_result = parser_generation(current_batch[idx], response)
                batch_results.append(new_result)
                with open(dump_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(new_result, ensure_ascii=False) + '\n')
                if do_resample:
                    tqdm_bar.update(1)
            except Exception as e:
                print(f"Error parsing response for {request_ids[idx]}: {e}")
                # queue.put_nowait((request_ids[idx], messages[idx]))
                if do_resample:
                    print("Resampling due to error...")
                    queue.put_nowait((request_ids[idx], messages[idx]))
                failed += 1

                if failed_log_file is not None:
                    with open(failed_log_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({
                            "request_id": request_ids[idx],
                            "error": str(e),
                            "request": requests[idx],
                            "response": response.content
                        }, ensure_ascii=False) + '\n')
                if error_doc_path is not None:
                    with open(error_doc_path, 'a', encoding='utf-8') as f:
                        f.write(f"{request_ids[idx]}\n")
            
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
        await asyncio.sleep(0.5)  # Optional pause to avoid rate limits

    return responses

def parser_generation(request, response):
    generation = robust_parse(response.content, parser)
    assert len((request['metadata']['sentences'])) == len(generation), \
        f"Mismatch in number of sentences: {len(request['metadata']['sentences'])} != {len(generation)}"

    result = {
        'request_id': request["request_id"],
        'message': request['message'],
        "generation": generation,
        "metadata": request['metadata']}
    return result


if __name__ == "__main__":
    import json
    import random
    import argparse
    from langchain_openai import AzureChatOpenAI

    parser = argparse.ArgumentParser(description="Process translation requests.")
    parser.add_argument("--lang", type=str, default="en_jstage", help="Language code", choices=["en_jstage", "ja", "en"])
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for LLM generation")
    parser.add_argument("--skip_error_doc", action="store_true", help="Skip documents with errors")
    
    args = parser.parse_args()
    # os.environ["AZURE_MODEL_NAME"] = "gpt-4.1-2025-04-14"
    os.environ["AZURE_MODEL_NAME"] = "gpt-4.1-mini-2025-04-14"
    os.environ["OPENAI_API_TYPE"] = "azure_ad"
    os.environ["OPENAI_API_VERSION"] = "2025-04-01-preview"
    os.environ["AZURE_OPENAI_API_KEY"] = "c11f07d09c2e41d28cd0b3494e37aa4f"
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://llm-jp-openai-mmwg-01.openai.azure.com"

    llm = AzureChatOpenAI(
        deployment_name=os.environ["AZURE_MODEL_NAME"],
        temperature=args.temperature,
        top_p=1,
        max_retries=2,
        timeout=120,
        logprobs=False
    )
    
    CONCURRENCY = 1
    parser = PydanticOutputParser(pydantic_object=SentRewriteObjectList, include_raw=True, strict=False)

    lang = args.lang
    train_data_root = "/data/xzhao/experiments/med-eval/dataset/losses/train_samples"
    data_path = os.path.join(train_data_root, f"random_{lang}_2k.jsonl")

    all_requests = []
    for item in load_jsonl_iteratively(data_path, request_num=2000):
        all_requests.append({
            "request_id": item['id'],
            "message": [
                {"role": "system", "content": ja_instruction if lang == "ja" else en_instruction},
                {"role": "user", "content": user_prompt.format(input=json.dumps(item['sentences'], indent=4))}
            ],
            "metadata": item
        })

    
    response_root = "/data/xzhao/experiments/med-eval/dataset/losses/resources"
    failed_log_file = os.path.join(response_root, f"translation_{lang}_2k_failed.jsonl")
    response_path = os.path.join(response_root, f"translation_{lang}_2k.jsonl")

    error_doc_path = None
    if args.skip_error_doc:
        error_doc_path = os.path.join(response_root, f"translation_{lang}_2k_error_docs.txt")
                
    while True:
        generated_reqids = []
        
        print(f"Response file will be saved to: {response_path}")
        
        if os.path.exists(response_path):
            generated = load_jsonl(response_path)
            for generated_item in generated:
                generated_reqids.append(generated_item['request_id'])
        
        generated_reqids = set(generated_reqids)
        print(f"=== The number of generated requests: {len(generated_reqids)}")
        print(f"=== Total requests to process: {len(all_requests)}")
        print(f"=== The number of processed requests: {len(generated_reqids)}")
        
        requests = []
        for request in all_requests:
            if request["request_id"] in generated_reqids:
                continue
            requests.append(request)

        if len(requests) == 0:
            print("=== All requests have been processed. Exiting.")
            break
        
        random.shuffle(requests)
        print([req['request_id'] for req in requests])
        print(f"=== Number of requests to process: {len(requests)}")
        try:
            print("=== Starting processing requests with concurrency:", CONCURRENCY)
            asyncio.run(
                adaptive_batch_process(
                    llm, requests, 
                    dump_file=response_path,
                    error_doc_path=error_doc_path,
                    start_concurrency=1, max_concurrency=5, step=1,
                    failed_log_file=failed_log_file,
                    do_resample=False)
            )
        except Exception as e:
            print(f"Error during processing: {e}")
            print("Retrying in 10 seconds...")

            if failed_log_file is not None:
                with open(failed_log_file, 'a', encoding='utf-8') as f:
                    for req in requests:
                        f.write(json.dumps({
                            "request_id": req['request_id'],
                            "error": str(e),
                            "request": req,
                            "response": None
                        }, ensure_ascii=False) + '\n')
            
            time.sleep(10)
            # continue