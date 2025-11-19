import re
import os
import json
import torch
import random
from tqdm import tqdm
from typing import Tuple
from utils import DATA_ROOT, load_json, load_jsonl_iteratively

TOKEN_PATTERN = r"""
    # 6) Dates & times
    \b\d{4}-\d{2}-\d{2}\b |
    \b\d{1,2}/\d{1,2}/\d{2,4}\b |
    \b\d{1,2}:\d{2}(?:\s?[ap]\.?m\.?)?(?=\W|$) |
    # 7) Multi-part abbrevs: U.S.A., Ph.D., e.g.
    \b(?:[A-Za-z]{1,4}\.){2,}(?=\W|$) |
    # 8) Single-part honorifics: Mr., Dr., Prof.
    \b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Mt)\.(?=\W|$) |
    # 9) C++ / C#
    \b[A-Za-z]\+\+(?=\W|$)|\b[A-Za-z]\#(?=\W|$) |
    # 10) Numbers
    (?<!\w)-\d+(?:\.\d+)? |        # negatives
    \b\d+/\d+\b |                  # simple fractions
    \b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b |  # 1,234,567.89
    |
    [\$€£¥₹₩]\d+(?:,\d{3})*(?:\.\d+)?  # $19.99
    |
    \b\d+(?:\.\d+)?%                   # 50%
    |
    \b\d+(?:st|nd|rd|th)\b             # 1st, 22nd
    |
    # numeric + unit (time/metric/imperial/storage/temps)
    \b\d+(?:\.\d+)?(?:
        [KkMmBb]|
        ms|s|sec|secs|min|mins|h|hr|hrs|
        cm|mm|km|m|g|kg|lb|lbs|
        kb|mb|gb|tb|KB|MB|GB|TB|
        KiB|MiB|GiB|TiB|
        °C|°F
    )\b
    |
    # 11) Numeric ranges
    \b\d+(?:–|-)\d+\b |
    # 12) Hyphenated compounds
    \b[^\W_]+(?:-[^\W_]+)+\b |
    # 13) Contractions/possessives
    \b[^\W\d_]+(?:['’][^\W\d_]+)+\b |
    # 14) Quoted single-letter (rock 'n' roll)
    '(?:[A-Za-z])' |
    # 15) Mentions/hashtags/cashtags
    [#@]\w+|\$[A-Za-z]{1,6} |
    # 16) Plain words (letters; allow internal dots like e.g)
    \b[^\W\d_]+(?:\.[^\W\d_]+)*\b |
    # 17) Ellipses
    \.\.\.|… |
    # 18) Multi-char operators / common markers
    &&|\|\||<=|>=|==|!=|-- |
    # 19) Single-char operators / punctuation
    [^\w\s]
"""

TOKENIZER_RE = re.compile(TOKEN_PATTERN, re.VERBOSE)

def shuffle_list_by_steps(array_lens, steps, window_size, seed=None):
    """
    Shuffle a list by a given number of steps.
    """
    if steps <= 0 or steps > array_lens:
        raise ValueError("Steps must be greater than 0 and less than the length of the list. But got steps: {}, length: {}".format(steps, array_lens))


    if seed is not None:
        random.seed(seed)

    
    orders = list(range(array_lens))
    modified = 0
    while modified < steps:
        src = random.randint(0, array_lens - 1)
        window = [i for i in range(src - window_size, src + window_size) if i != src and 0 <= i < array_lens]
        if len(window) == 0:
            continue

        tgt = random.choice(window)
        # print("src: {}, tgt: {}; orders[src]: {}, orders[tgt]: {}".format(src, tgt, orders[src], orders[tgt]))

        if steps - modified == 1 and orders[src] == src and orders[tgt] == tgt:
            continue
            
        if orders[src] == src:
            modified += 2 if orders[tgt] == tgt else 1
            orders[src], orders[tgt] = orders[tgt], orders[src]
        else:
            if orders[tgt] == tgt:
                modified += 1
                orders[src], orders[tgt] = orders[tgt], orders[src]
            
    assert sum([i != j for i, j in zip(orders, range(array_lens))]) == steps, \
        "Shuffle steps do not match. Modification: {}, Expected: {}, Actual: {}".format(
        modified, steps, sum([i != j for i, j in zip(orders, range(array_lens))])
    )
    return orders

def reorder_sents(sents, distance, seed=None) -> Tuple[torch.Tensor, list, str]:
    """
    Reorder sentences in the text by a given number of steps.
    
    Return a tuple of (input_ids, sent_reorder, reconstructed_text).
    - input_ids: Tensor of input IDs for the model
    - sent_reorder: List of indices indicating the new order of sentences
    - reconstructed_text: The text after reordering sentences
    """
    if seed is not None:
        random.seed(seed)
    sent_reorder = shuffle_list_by_steps(len(sents), distance=distance, window_size=len(sents), seed=seed)
    reordered_sents = [sents[i] for i in sent_reorder]
    return reordered_sents

def tokenize(text: str):
    """
    Regex-based, punctuation-aware tokenizer with smart handling for URLs,
    abbrevs, dates/times, numbers+units, code spans, emoji, etc.
    """
    out = []
    raw = TOKENIZER_RE.findall(text)
    for tok in raw:
        if tok != '':
            out.append(tok)
    return out

def detokenize(tokens):
    """
    Combine tokens back into a natural sentence with proper spacing.
    """
    text = ""
    no_space_before = set(".,!?;:%)]}'”’")   # don’t put space before these
    no_space_after  = set("([{'“‘")          # don’t put space after these

    for i, tok in enumerate(tokens):
        if i == 0:
            text += tok
        elif tok in no_space_before:
            text += tok
        elif text[-1] in no_space_after:
            text += tok
        else:
            text += " " + tok
    return text

def noise_sentence_by_word(sent, disorder_ratio, window_size=4, seed=None) -> Tuple[torch.Tensor, list, str]:
    if seed is not None:
        random.seed(seed)

    tokens = tokenize(sent)
    shuffle_edit_distance = int(len(tokens) * disorder_ratio) + 1
    if shuffle_edit_distance >= 2:
        shuffled_token_orders = shuffle_list_by_steps(
            len(tokens), steps=shuffle_edit_distance, window_size=window_size, seed=seed)
    else:
        shuffled_token_orders = list(range(len(tokens)))
    
    deleted_token_orders = []
    for i, j in enumerate(shuffled_token_orders):
        if i == j and random.random() < disorder_ratio:
            pass
        else:
            deleted_token_orders.append(j)
    noisy_tokens = [tokens[idx] for idx in deleted_token_orders]
    return detokenize(noisy_tokens)

def _construct_instruction(template, input_text, output_text, tuning=False):
    assert "{input}" in template and "{output}" in template, \
        "Template must contain {input} and {output} placeholders."
    assert template.endswith("{output}"), "Template must end with {output} placeholder."
    
    if not tuning:
        return {"text": template.format(input=input_text, output=output_text)}
    else:
        return {"template": template, "noise": input_text, "raw": output_text}


def generate_denoising_abstract(
        item: dict, 
        index: int,
        sentence_reorder: float,
        word_reorder: float):
    denoises = []
    template = random.choice(templates["denoising"]["en"])
    if sentence_reorder + word_reorder == 0:
        raw_abstract = item["raw"]['abstract']
        noisy_abstract = item["noisy"][index]['abstract']
        denoises.append(template.format(input=noisy_abstract, output=raw_abstract))
    else:
        raise NotImplementedError("Currently only support denoising without syntax noise")
    return denoises 

def generate_denoising_sentences(
        item: dict, 
        index: int,
        code_switch: bool,
        monolingual: bool,
        word_reorder: float, 
        tuning: bool):
    """
    This function is only available for code-switching corpus,
    for the data files located at ROOT/en_jstage/code-switch.
    """
    template_type = "monolingual" if monolingual else "multilingual"
    denoises = []
    raw_sentences = item["raw"]['sentences']
    noisy_sentences = item["noisy"][index]['sentences'] if code_switch else item["raw"]['sentences']
    for raw_sentence, noisy_sentence in zip(raw_sentences, noisy_sentences):
        if word_reorder > 0.0:
            noisy_sentence = noise_sentence_by_word(
                noisy_sentence, disorder_ratio=word_reorder, window_size=4, seed=None)
        if noisy_sentence != raw_sentence:
            template = random.choice(templates["denoising"][template_type])
            instruction = _construct_instruction(template, noisy_sentence, raw_sentence, tuning=tuning)
            denoises.append(instruction)

    if 'qa' in item["raw"]:
        raw_qas = item["raw"]['qa']
        noisy_qas = item["noisy"][index]['qa'] if code_switch else item["raw"]['qa']
        for raw_qa, noisy_qa in zip(raw_qas, noisy_qas):
            assert len(raw_qa) == len(noisy_qa) == 2, f"Length mismatch: {len(raw_qa)}, {len(noisy_qa)}"
            raw_qa = f"{raw_qa[0]} {raw_qa[1]}"
            noisy_qa = f"{noisy_qa[0]} {noisy_qa[1]}"
            if word_reorder > 0.0:
                noisy_qa = noise_sentence_by_word(
                    noisy_qa, disorder_ratio=word_reorder, window_size=4, seed=None)
                
            if raw_qa != noisy_qa:
                template = random.choice(templates["denoising"][template_type])
                instruction = _construct_instruction(template, noisy_qa, raw_qa, tuning=tuning)
                denoises.append(instruction)
    return denoises

def serialize(instruction, tuning, docid=None, type=None):
    assert isinstance(instruction, dict), "Instruction must be a dictionary."
    if not tuning:
        assert "text" in instruction, "For training, instruction must contain 'text' field."
        if docid is None and type is None:
            return f"{json.dumps({'text': instruction['text']}, ensure_ascii=False)}\n"
        else:
            return f"{json.dumps({'docid': docid, 'text': instruction['text'], 'type': type}, ensure_ascii=False)}\n"
    else:
        assert "template" in instruction and "noise" in instruction and "raw" in instruction, \
            "For tuning, instruction must contain 'template', 'noise', and 'raw' fields."
        if docid is None and type is None:
            return f"{json.dumps({'template': instruction['template'], 'noise': instruction['noise'], 'raw': instruction['raw']}, ensure_ascii=False)}\n"
        else:
            return f"{json.dumps({'docid': docid, 'template': instruction['template'], 'noise': instruction['noise'], 'raw': instruction['raw'], 'type': type}, ensure_ascii=False)}\n"



if __name__ == "__main__":
    import random
    import argparse
    from multiprocessing import Pool

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--langs", type=str, nargs='+', 
        default=["en"], choices=["ja", "en"], help="List of target languages")
    parser.add_argument("--umls_noise_ratio", type=float, default=0.0)
    parser.add_argument("--wordnet_noise_ratio", type=float, default=0.0)
    parser.add_argument("--word_disorder_ratio", type=float, default=0.0)
    parser.add_argument("--rewrite", action="store_true")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--tuning", action="store_true")
    args = parser.parse_args()

    # assert float(args.umls_noise_ratio) + float(args.wordnet_noise_ratio) + float(args.word_disorder_ratio) > 0, "At least one noise argument must be set"
    do_code_switch = float(args.umls_noise_ratio) + float(args.wordnet_noise_ratio) > 0
    assert not (float(args.umls_noise_ratio) > 0 and float(args.wordnet_noise_ratio) > 0), "Only one type of vocabulary noise can be applied at a time"
    tuning_suffix = ".tuning" if args.tuning else ""
    if args.umls_noise_ratio > 0:
        if len(args.langs) == 1 and args.langs[0] == "en":
            filename_prefix = f"umls.monolingual.{str(int(args.umls_noise_ratio*100))}"
        else:
            filename_prefix = f"umls.multilingual.{'-'.join(args.langs)}.{str(int(args.umls_noise_ratio*100))}"
        full_items_fn = os.path.join(DATA_ROOT, "datasets", "medical", "en_jstage", "code-switch", f"{filename_prefix}.jsonl")
    elif args.wordnet_noise_ratio > 0:
        if len(args.langs) == 1 and args.langs[0] == "en":
            filename_prefix = f"wordnet.monolingual.{str(int(args.wordnet_noise_ratio*100))}"
        else:
            filename_prefix = f"wordnet.multilingual.{'-'.join(args.langs)}.{str(int(args.wordnet_noise_ratio*100))}"
        full_items_fn = os.path.join(DATA_ROOT, "datasets", "medical", "en_jstage", "wordnet", f"{filename_prefix}.jsonl")
    elif args.word_disorder_ratio > 0:
        filename_prefix = f"syntax.denoising"
        full_items_fn = os.path.join(DATA_ROOT, "datasets", "medical", "en_jstage", "full.jsonl")
    else:
        filename_prefix = f"baseline"
        full_items_fn = os.path.join(DATA_ROOT, "datasets", "medical", "en_jstage", "full.jsonl")
    
    assert os.path.exists(full_items_fn), f"File {full_items_fn} does not exist"
    
    templates = load_json("./instructions/instruction-templates.new.json")
    output_dir = os.path.join(DATA_ROOT, "instructions", "denoising", f"{filename_prefix}{tuning_suffix}")
    os.makedirs(output_dir, exist_ok=True)

    print("===> Output directory created at:", output_dir)

    def create_denoising_instruction_by_index(i):
        if args.word_disorder_ratio == 0.0:
            file_name = os.path.join(output_dir, f"denoising-{i}.jsonl")
        else:
            file_name = os.path.join(
                output_dir,
                f"denoising-{i}.word{int(args.word_disorder_ratio * 100)}.jsonl"
            )

        if os.path.exists(file_name) and args.rewrite is False:
            print(f"{file_name} already exists")
            return  # skip instead of sys.exit(0), otherwise only 1 process runs

        print(f"Generating instructions to {file_name}")
        with open(file_name, "w", encoding="utf8") as output_fp:
            for item in tqdm(
                load_jsonl_iteratively(full_items_fn, request_num=None),
                desc=f"Generating denoising instructions (i={i})"
            ):
                denoising_instructions = generate_denoising_sentences(
                    item=item,
                    index=i,
                    code_switch=do_code_switch,
                    monolingual=((len(args.langs) == 1 and args.langs[0] == "en") or args.wordnet_noise_ratio == 0 or args.umls_noise_ratio == 0),
                    word_reorder=args.word_disorder_ratio,
                    tuning=args.tuning
                )
                for denoising_instruction in denoising_instructions:
                    output_fp.write(
                        serialize(
                            denoising_instruction,
                            tuning=args.tuning,
                            docid=item["docid"],
                            type=f"denoising-{filename_prefix}",
                        )
                    )
    # create_denoising_instruction_by_index(0)
    # CODE_SWITCH_ROOT = "/data/xzhao/dataset/roman-pretrain/datasets/medical/en_jstage/code-switch"
    # assert CODE_SWITCH_ROOT in full_items_fn, "This function is only available for code-switching corpus, located at ROOT/en_jstage/code-switch."
    with Pool(processes=5) as pool:  # or fewer if you don’t want all 5 in parallel
        pool.map(create_denoising_instruction_by_index, range(5))
    