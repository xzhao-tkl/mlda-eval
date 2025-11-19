#!/bin/bash

python3 tokenization.py --lang ja --tokenizer llm-jp/llm-jp-3-7.2b
python3 tokenization.py --lang ja --tokenizer meta-llama/Meta-Llama-3-8B
python3 tokenization.py --lang ja --tokenizer Qwen/Qwen2.5-7B

# python3 tokenization.py --lang ja --tokenizer llm-jp/llm-jp-3-7.2b --file_pattern science_
# python3 tokenization.py --lang ja --tokenizer meta-llama/Meta-Llama-3-8B --file_pattern science_
# python3 tokenization.py --lang ja --tokenizer llm-jp/llm-jp-3-7.2b --file_pattern balanced_roman
# python3 tokenization.py --lang ja --tokenizer meta-llama/Meta-Llama-3-8B --file_pattern balanced_roman

# python3 tokenization.py --lang zh --tokenizer Qwen/Qwen2.5-7B --file_pattern balanced_romanization
# python3 tokenization.py --lang zh --tokenizer meta-llama/Meta-Llama-3-8B --file_pattern balanced_romanization
# python3 tokenization.py --lang zh --tokenizer meta-llama/Meta-Llama-3-8B --file_pattern balanced_roman2en

python3 tokenization.py --lang en --tokenizer llm-jp/llm-jp-3-7.2b
python3 tokenization.py --lang en --tokenizer Qwen/Qwen2.5-7B 

python3 tokenization.py --lang zh --tokenizer meta-llama/Meta-Llama-3-8B
python3 tokenization.py --lang zh --tokenizer Qwen/Qwen2.5-7B
