## Data augmentation of medical domain corpora

This package augment medical domain corpora of Chinese, Japanese and English by creating instructions from the plain corpora, via three aspects:

1. Regex-based instruction creation
2. LLM-based QA-instruction generation
3. Romanization
4. Instruction generation


### 1. Regex-based instruction creation

```shell
python3 generate_regex --lang en
python3 generate_regex --lang en_jstage
python3 generate_regex --lang ja
python3 generate_regex --lang zh
```

Note that, the regex-based data augmentation of ja and zh requires a small deep model to split sentences. 
It is recommanded to run the command on GPU, as shown in `run_regex_gpu.sh`. 


### 2. LLM-based QA-instruction generation

Here, we use Deepseek-R1 distilled models to instruct LLMs generate five question-answer pairs for each document.

Specifically, we use [deepseek-ai/DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) for Chinese QA pairs generation, [rinna/qwen2.5-bakeneko-32b-instruct](https://huggingface.co/rinna/qwen2.5-bakeneko-32b-instruct) for Japanese generation. 
We use vllm to load models on llm-nvlink clusters and sending `chat-completion` request to the running models. 

This work requires three steps:

1. Instal `vecinf`. 
To run LLMs on llm-nvlink cluster, we utilize [vector-interence](https://github.com/VectorInstitute/vector-inference), which is an easy-to-use solution to run inference servers on Slurm-managed computing clusters using vLLM.
As the original package cannot serve our requirements, please install the modified version under the project root by: 
```shell
cd <project_root>/vector-inference
bash venv.sh
```

2. Start the LLM server. 
Here are the commond to run them on llm-nvlink clusters:

```shell
DeepSeek-R1-Distill-Llama-70B: vec-inf launch DeepSeek-R1-Distill-Llama-70B --partition gpu --model-family DeepSeek-R1 --num-nodes 1 --num-gpus 8 --model-weights-parent-dir /data/xzhao/huggingface/models/DeepSeek-R1-Distill-Llama-70B --log-dir /home/xzhao/workspace/roman-pretrain/datasets/logs --time 48:00:00

vec-inf launch DeepSeek-R1-Distill-Qwen-JP-32B --partition gpu --model-family DeepSeek-R1 --num-nodes 1 --num-gpus 4 --model-weights-parent-dir /data/xzhao/huggingface/hub/models--rinna--qwen2.5-bakeneko-32b-instruct/snapshots/ca18bc08ae045b74fd592293aee1d5404bd32d9c --log-dir /home/xzhao/workspace/roman-pretrain/datasets/logs --time 48:00:00

Llama-3.3-70B-Instruct: vec-inf launch Llama-3.3-70B-Instruct --partition gpu --model-family Llama-3.3  --num-nodes 1 --num-gpus 8 --model-weights-parent-dir /data/xzhao/huggingface/models/Llama-3.3-70B-Instruct/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --log-dir /home/xzhao/workspace/roman-pretrain/datasets/logs --time 48:00:00

Qwen3-32B: vec-inf launch Qwen3-32B --partition gpu --model-family Qwen3  --num-nodes 1 --num-gpus 8 --model-weights-parent-dir /data/xzhao/huggingface/models/Qwen3-32B/models--Qwen--Qwen3-32B/snapshots/d47b0d4ae4b48fde975756bf360a63a9cca8d470 --log-dir /home/xzhao/workspace/roman-pretrain/datasets/logs --time 48:00:00


llm-jp-3.1: vec-inf launch llm-jp-3.1-8x13b-instruct4 --partition gpu --num-nodes 1 --num-gpus 8 --log-dir /home/xzhao/workspace/roman-pretrain/datasets/logs --time 48:00:00 --model-weights-parent-dir /data/xzhao/huggingface/hub/models--llm-jp--llm-jp-3.1-8x13b-instruct4/snapshots/d1ee76f9439f045e4499551c4dd686e9b4001c8f

Swallow-70B: vec-inf launch Llama-3.3-Swallow-70B-Instruct-v0.4 --partition gpu --num-nodes 1 --num-gpus 8 --log-dir /home/xzhao/workspace/roman-pretrain/datasets/logs --time 48:00:00 --model-weights-parent-dir /data/xzhao/huggingface/hub/models--tokyotech-llm--Llama-3.3-Swallow-70B-Instruct-v0.4/snapshots/f99e99588303e8a52b88076d3a5f24db19f757a6

```

3. Query the LLMs to get response.
First, find the endpoint used by the running LLMs by checking the sbatch log. 
Since there are many Japanese documents, run multiple LLMs and divide the documents into several parts for QA generation. 
Use the `start_indice` and `request_num` parameters to specify the starting and ending indices of the documents.
Here is an example: 

```shell
python3 llm_request.py --lang ja --endpoint http://gpu-node2:8080/v1/chat/completions --start_indice 100000 --request_num 100000
```

Note that:
1. We only run QA-instruction generation for Chinese and Japanese as English corpora are too large to process all of the documents. 
2. VLLM occasionally fails to generate responses or produces problematic ones. Multiple generation runs may be necessary to ensure QA pairs are created for each document. Refer to `process_generation.ipynb` for format checking procedures.
3. After format verification, rename the generation files from `qa-gen.response.s0exxxx.jsonl` to `response.full.jsonl` for subsequent processing.

### 3. Romanization

Run romanization on Chinese and Japanese using code below. 
This command also parse QA response from the previous step.

```shell
python3 generate_roman.py --lang ja
python3 generate_roman.py --lang zh
python3 generate_roman.py --lang en_jstage
```

Note that, this process step only merge QA generations to the preprocessed.jsonl for `en_jstage` rather than doing romanization, as English data cannot be romanized. 

Until here, you will see three or files under `DATA_ROOT/medical/LANG/`

- `data.jsonl`: The original paper abstract document
- `preprocessed.jsonl`: The instruction data by augmentating `data.jsonl` with regex-based instruction creation process
- `full.jsonl`: The structured data organized by document IDs, along with regex-based augmented instructions, LLM-generated Q&A, and romanization of the above.
- `trans.jsonl` (optional): This file contains the English version of all documents, only available for `ja` and `zh`.

### 4. Instruction generation

Finally, after extending plain corpora into structured datasets after running three steps, create the final instructions with the below command:

To generate instruction without romanization scripts: 

```shell
python3 generate_instructions.py --lang zh --xx
python3 generate_instructions.py --lang ja --xx
python3 generate_instructions.py --lang en --xx
```

The `xxx` specify the type of generated instructions. 
Here are the possible types: 

- medical_native: medical instructions. Available for `ja`, `en`, `zh`
- medical_roman: romanization of medical instructions. Available for `ja`, `zh`
- medical_en2roman: translation from English to LANG-romanization. Available for `ja`, `zh`
- medical_roman2en: translation from LANG-romanization to English. Available for `ja`, `zh`
- medical_native2roman: translation from LANG to English. Available for `ja`, `zh`
- medical_roman2native: translation from English to LANG. Available for `ja`, `zh`
- medical_trans: combination of medical_native2roman and medical_roman2native
- medical_halfroman: cross-lingual instructions with LANG as input and LANG-romanization as output
- medical_halfroman_reverse: cross-lingual instructions with LANG-romanization as input and LANG as output

- balanced_native
- balanced_roman
- balanced_en2roman
- balanced_roman2en
- balanced_native2roman
- balanced_roman2native
- balanced_trans

- science_native
- science_roman
- science_en2roman
- science_roman2en
- science_native2roman
- science_roman2native
- science_trans

All the data will be saved to `DATA_ROOT/instructions/<lang>/`

Each line is represented with the format: 

```json
{
    "text": <text>,
    "docid": <docid>,
    "type": <instruction_type>
}
```

Based on these files, you can create your own dataset with different recipes.


### 5. Denoising Instruction Generation

To generate denoising instructions, you first need to prepare the base files that introduce noise.  
We support different types of noise, including:  
- Code-switching  
- Syntax noise  

---

#### 5.1 Code-switching

Code-switching noise replaces certain words with candidates from predefined vocabularies.  
Our code supports three types of code-switching:  

**UMLS-based synonyms**  

UMLS provides multilingual aliases for biomedical entities.  
To prepare the UMLS-based multilingual code-switching dataset, follow the scripts in `umls_codeswitch`:  

1. Recognize UMLS entities in the training corpus:  
   ```bash
   bash 05_01_process_umls_ner.sh
   ```
2. Retrieve multilingual UMLS aliases for entities in the training corpus using the UMLS API: `05_03_download_cuis_codes.py`

**Multilingual-Wordnet based synonym**: 
TO BE DONE

**Random words**: 

We create denoising instructions in the format: `Restore the noisy text: {input}\n{output}`
- `{input}`: a noisy sentence (split and reconstructed into a full document)
- `{output}`: the original, clean version of the sentence

After preparing the resources for code-switching, you can generate denoising instructions with:

```shell
python3 04_denoising_instructions.py \
--umls_noise_ratio x 
--wordnet_noise_ratio x 
--random_word_ratio x`
--repeat 5
```

**Notes:**
- The three noise resources are not compatible; you can only use one at a time to construct denoising instructions.
- The `*_noise_ratio` parameters specify the percentage of words to replace with noise.
- The `repeat` parameter controls how many different instructions to generate for each noisy sentence.

#### 5.2 Syntax noise

We combine two syntax noising strategies: `reordering` and `deletion`.
To simplify parameterization, the process first reorders words and then deletes a portion of them.
You can control the ratio of affected words with the following command:

```shell
python3 04_denoising_instructions.py \
--word_disorder_ratio 0.16
```
**Notes:**
- The `word_disorder_ratio` parameter is compatible with the code-switching strategy. Syntax noise is applied after code-switching.
- After generating denoising instructions, remember to perform tokenization for token counting.