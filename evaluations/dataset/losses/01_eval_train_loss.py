from typing import Tuple
from polars import first
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch.nn.functional as F

def caluate_loss_single_per_token(model, tokenizer, items, lang, sequence_type, distortion):
    total_loss = 0
    seq2loss = []
    for item in tqdm(items):
        org_input_ids, _, _ = reconstruct_sequence(
            tokenizer=tokenizer, 
            item=item, 
            lang=lang,
            sequence_type="full",
            distortion="original")
        
        input_ids, token_labels, tgt_text = reconstruct_sequence(
            tokenizer=tokenizer, 
            item=item,
            lang=lang, 
            sequence_type=sequence_type,
            distortion=distortion)
        input_ids = input_ids.to(model.device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            per_token_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none'
            ).view(shift_labels.size())
        
        token_losses = []
        for label, loss in zip(shift_labels[0].tolist(), per_token_loss[0].tolist()):
            token_losses.append((tokenizer.decode(label), label, loss))
        token_losses = tuple(token_losses)
        
        loss = per_token_loss[0].mean().item()
        seq2loss.append({
            "id": item["id"],
            "text": tgt_text,
            "avg_loss": float(loss),
            "token_losses": token_losses,
            "metadata": {
                "org_text": item["text"],
                "org_input_ids": org_input_ids[0].tolist(),
                "sentences": item["sentences"],
                "model_name": model.name_or_path,
                "sequence_type": sequence_type,
                "distortion": distortion,
                "token_labels": token_labels,
            }
        })
        total_loss += loss
    average_loss = total_loss / len(items)
    return seq2loss, average_loss


def caluate_loss_single(
        model, tokenizer, items, lang,
        sequence_type="full", distortion='none'):
    model.eval()
    total_loss = 0
    seq2loss = []
    for item in tqdm(items):
        input_ids, tgt_text = reconstruct_sequence(
            tokenizer=tokenizer, 
            item=item,
            lang=lang,
            sequence_type=sequence_type,
            distortion=distortion)
        input_ids = input_ids.to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss

        seq2loss.append({
            "id": item["id"],
            "text": item["text"],
            "sentences": item["sentences"],
            "reconstructed_text": tgt_text,
            "sequence_type": sequence_type,
            "distortion": distortion,
            "avg_loss": float(loss),
        })
        total_loss += loss.item()

    average_loss = total_loss / len(items)
    return seq2loss, average_loss


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

UNK_TOKEN_ID = 0 # <unk>
REWRITE_DICT = {
    "syntax": "syntactical",
    "lexicon": "lexical",
    "semantic": "semantical",
    "wordtrans": "word_aligned_translation",
    "translation": "flexible_translation"
}
def mask_tokens(tokenizer, text, mask_prob, seed=None) -> Tuple[torch.Tensor, list, str]:
    """
    Mask tokens by <unk> token in the text with a given probability.
    Return a tuple of (input_ids, token_labels, reconstructed_text).
    - input_ids: Tensor of input IDs for the model
    - token_labels: List of token labels indicating whether the token is masked or not
    - reconstructed_text: The text after masking tokens
    """
    if seed is not None:
        random.seed(seed)
    token_ids = tokenizer(text, return_tensors="pt", truncation=True)['input_ids'][0].tolist()
    noise_token_ids = [token_ids[0]]
    token_labels = ["<token>"]
    for tok in token_ids[1:]:
        if random.random() < mask_prob:
            noise_token_ids.append(UNK_TOKEN_ID)
            token_labels.append("<unk>")
        else:
            noise_token_ids.append(tok)
            token_labels.append("<token>")

    return torch.tensor([noise_token_ids]), token_labels, "".join(tokenizer.decode(noise_token_ids))

def delete_tokens(tokenizer, text, mask_prob, seed=None) -> Tuple[torch.Tensor, list, str]:
    """
    Delete tokens in the text with a given probability.

    Return a tuple of (input_ids, token_labels, reconstructed_text).
    - input_ids: Tensor of input IDs for the model
    - token_labels: List of token labels indicating whether the token is deleted or not
    - reconstructed_text: The text after deleting tokens
    """
    if seed is not None:
        random.seed(seed)
    token_ids = tokenizer(text, return_tensors="pt", truncation=True)['input_ids'][0].tolist()
    noise_token_ids = [token_ids[0]]
    token_labels = ["<token>"]
    for tok in token_ids[1:]:
        if random.random() < mask_prob:
            token_labels.append("<deleted>")
        else:
            noise_token_ids.append(tok)
            token_labels.append("<token>")
    return torch.tensor([noise_token_ids]), token_labels, "".join(tokenizer.decode(noise_token_ids))

def replace_tokens(tokenizer, text, mask_prob, seed=None) -> Tuple[torch.Tensor, list, str]:
    """
    Replace tokens in the text with <unk> token with a given probability.

    Return a tuple of (input_ids, token_labels, reconstructed_text).
    - input_ids: Tensor of input IDs for the model
    - token_labels: List of token labels indicating whether the token is replaced or not
    - reconstructed_text: The text after replacing tokens
    """
    if seed is not None:
        random.seed(seed)
    
    vocab_ids = list(tokenizer.get_vocab().values())
    token_ids = tokenizer(text, return_tensors="pt", truncation=True)['input_ids'][0].tolist()
    noise_token_ids = [token_ids[0]]
    token_labels = ["<token>"]
    for tok in token_ids[1:]:
        if random.random() < mask_prob:
            noise_token_ids.append(random.choice(vocab_ids))
            token_labels.append("<random>")
        else:
            noise_token_ids.append(tok)
            token_labels.append("<token>")
    return torch.tensor([noise_token_ids]), token_labels, "".join(tokenizer.decode(noise_token_ids))

def synonyms_tokens(tokenizer, text, synonym_list, mask_prob, lang, seed=None) -> Tuple[torch.Tensor, list, str]:
    """
    Replace tokens in the text with synonyms with a given probability.
    If monolingual is True, use monolingual synonyms, otherwise use multilingual synonyms.
    
    Input:
    - tokenizer: The tokenizer to use for encoding the text
    - text: The original text to process
    - synonym_list: A list of synonyms to use for replacement
    - mask_prob: Probability of replacing a token with a synonym
    - lang: Language of the synonyms (e.g., "ja" for Japanese)
    - seed: Random seed for reproducibility

    Return a tuple of (input_ids, token_labels, reconstructed_text).
    - input_ids: Tensor of input IDs for the model
    - token_labels: List of token labels indicating whether the token is replaced or not
    - reconstructed_text: The text after replacing tokens with synonyms
    """
    assert lang in ["ja", "en"], "Language must be either 'ja' or 'en'."
    assert isinstance(synonym_list, list), "Synonym list must be a list of tuples. But got: {}".format(type(synonym_list))
    
    if seed is not None:
        random.seed(seed)
    
    token_ids = tokenizer(text, return_tensors="pt", truncation=True)['input_ids'][0].tolist()
    token_labels = ["<token>"]
    
    def get_synonym_text(synonyms, tgt_lang, prob):
        replaced_syns = []
        noisy_text = text
        synonyms = [syn for syn in synonyms if syn["synonyms"][tgt_lang] is not None]
        for syn in synonyms:
            if random.random() < prob: 
                replaced_syns.append(syn)

        replaced_indexes = []
        offset = 0
        for synonym in replaced_syns:
            replaced_indexes.append((synonym["index"][0] + offset, synonym["index"][0] + len(synonym["synonyms"][tgt_lang]) + offset))
            offset += len(synonym["synonyms"][tgt_lang]) - (synonym["index"][1] - synonym["index"][0])

        replaced_syns = sorted(replaced_syns, key=lambda x: x["index"][0], reverse=True)
        for synonym in replaced_syns:
            noisy_text = noisy_text[:synonym["index"][0]] + synonym["synonyms"][tgt_lang] + noisy_text[synonym["index"][1]:]
        return noisy_text, replaced_syns[::-1], replaced_indexes

    noisy_text, replaced_syns, replaced_indexes = get_synonym_text(synonym_list, lang, mask_prob)
    noisy_tokens = tokenizer(noisy_text, return_offsets_mapping=True)
    noisy_token_positions = noisy_tokens["offset_mapping"]
    noisy_tokens = noisy_tokens["input_ids"]

    token_labels = []
    for idx, (start, end) in enumerate(noisy_token_positions):
        for index in replaced_indexes:
            if start >= index[0] and end <= index[1]:
                token_labels.append("<synonym>")
                break
        if len(token_labels) <= idx:
            token_labels.append("<token>")
        
    assert len(noisy_tokens) == len(token_labels), \
        "The length of noisy tokens and token labels must be the same. But got: {} vs {}".format(len(noisy_tokens), len(token_labels))
    return torch.tensor([noisy_tokens]), token_labels, noisy_text

def reorder_tokens(tokenizer, text, shuffle_ratio, window_size, seed=None) -> Tuple[torch.Tensor, list, str]:
    """
    Reorder tokens in the text by a given number of steps.

    Return a tuple of (input_ids, token_reorder, reconstructed_text).
    - input_ids: Tensor of input IDs for the model
    - token_reorder: List of indices indicating the new order of tokens
    - reconstructed_text: The text after reordering tokens
    """
    assert 0 <= shuffle_ratio <= 1, "Shuffle ratio must be between 0 and 1."
    if seed is not None:
        random.seed(seed)
    token_ids = tokenizer(text, return_tensors="pt", truncation=True)['input_ids'][0].tolist()
    steps = max(int(len(token_ids) * shuffle_ratio), 2)
    assert steps >= 2, "Shuffle steps must be greater than 2. But got: {}, by shuffle_ratio: {} and sequence_length: {}".format(steps, shuffle_ratio, len(token_ids))
    token_orders = shuffle_list_by_steps(array_lens=len(token_ids[1:]), steps=steps, window_size=window_size, seed=seed)
    token_ids = [token_ids[0]] + [token_ids[1:][i] for i in token_orders]
    token_orders = [0] + [i + 1 for i in token_orders]  # +1 because the first token is not shuffled
    return torch.tensor([token_ids]), token_orders, "".join(tokenizer.decode(token_ids))

def reorder_sents(tokenizer, sents, steps, seed=None) -> Tuple[torch.Tensor, list, str]:
    """
    Reorder sentences in the text by a given number of steps.
    
    Return a tuple of (input_ids, sent_reorder, reconstructed_text).
    - input_ids: Tensor of input IDs for the model
    - sent_reorder: List of indices indicating the new order of sentences
    - reconstructed_text: The text after reordering sentences
    """
    if seed is not None:
        random.seed(seed)
    sent_reorder = shuffle_list_by_steps(len(sents), steps=steps, window_size=3, seed=seed)
    reordered_sents = [sents[i] for i in sent_reorder]
    new_sent = ''.join(reordered_sents)
    assert ''.join(sents) != new_sent, "Reordered sentences should not be the same as original sentences."
    tokens = tokenizer(new_sent, return_tensors="pt", truncation=True)['input_ids']
    return tokens, sent_reorder, new_sent

def rewrite_sents(tokenizer, sents, rewrites, ratio, rewrite_type, seed=None) -> Tuple[torch.Tensor, list, str]:
    """
    Rewrite sentences in the text by a given ratio.
    There are five types of rewriting patterns: 
    - Syntax: Keep the vocabulary and change the syntax
    - Lexicon: Keep the syntax and change the vocabulary
    - Semantic: Change the meaning of the text while keeping the syntax and vocabulary
    - Word Translation: Translate words to another language while keeping the syntax and structure
    - Text Generation: Generate new text based on the original text

    Return a tuple of (input_ids, sent_reorder, reconstructed_text).
    - input_ids: Tensor of input IDs for the model
    - rewrite_labels: List of labels indicating the type of rewriting applied to each sentence
    - reconstructed_text: The text after reordering sentences
    """
    if seed is not None:
        random.seed(seed)
    rewrite_sents, rewrite_labels = [], []
    for sent, rewrite in zip(sents, rewrites):
        if random.random() < ratio:
            rewrite_sents.append(rewrite)
            rewrite_labels.append(f"<{rewrite_type}>")
        else:
            rewrite_sents.append(sent)
            rewrite_labels.append("<sent>")
    tokens = tokenizer(' '.join(rewrite_sents), return_tensors="pt", truncation=True)['input_ids']
    return tokens, rewrite_labels, ''.join(rewrite_sents)

def reconstruct_sequence(
        tokenizer, item, lang,
        sequence_type="full", 
        distortion="original", 
        seed=None) -> Tuple[torch.Tensor, list, str]:
    """
    Reconstruct a sequence from the item based on the sequence type and distortion.
    - tokenizer: The tokenizer to use for encoding the text
    - item: A dictionary containing:
        - id: Unique identifier for the item
        - text: The original text
        - sentences: A list of sentences in the text
    - sequence_type: "full", "partial-start", "partial-middle", "partial-end"
    - distortion: "original", "mask-5", "delete-10", "replace-20", "resent-2", "retoken-5", etc.
    
    Returns a tuple of (input_ids, text).
    - input_ids: Tensor of input IDs for the model
    - text: The reconstructed text after applying the distortion
    """
    if seed is not None:
        random.seed(seed)
    
    hash_id = item["id"]
    text = item["text"]
    sents = item["sentences"]
    assert ''.join(sents) == text, "The sentences do not match the original text."

    # Truncate text with selected sentences
    sent_splits = sorted([i%4 for i in range(len(sents))])
    assert len(sents) >= 4
    if sequence_type.startswith("partial"):
        target_parts = [int(char)-1 for char in sequence_type.split("-")[1]]
        assert sequence_type.split("-")[1] in ["1", "2", "3", "4", "12", "13", "14", "23", "24", "34"]
        sents = [sent for sent, sent_id in zip(sents, sent_splits) if sent_id in target_parts]
    
    # Construct text from sentences and apply distortion
    text = "".join(sents)
    if distortion == "original":
        token_ids = tokenizer(text, return_tensors="pt", truncation=True)['input_ids']
        token_labels = ["<token>"] * len(token_ids[0])
        return token_ids, token_labels, text
    elif distortion.startswith("mask-"):
        noisy_prob = float(distortion.split("-")[1]) / 100
        return mask_tokens(tokenizer, text, noisy_prob, seed=hash_id)
    elif distortion.startswith("delete-"):
        noisy_prob = float(distortion.split("-")[1]) / 100
        return delete_tokens(tokenizer, text, noisy_prob, seed=hash_id)
    elif distortion.startswith("replace-"):
        noisy_prob = float(distortion.split("-")[1]) / 100
        return replace_tokens(tokenizer, text, noisy_prob, seed=hash_id)
    elif distortion.startswith("monosyn-"):
        noisy_prob = float(distortion.split("-")[1]) / 100
        assert "wordnet_synonyms" in item, "Synonym distortion requires 'wordnet_synonyms' in the item."
        syn_lang = "ja" if lang == "ja" else "en"
        return synonyms_tokens(tokenizer, text, item["wordnet_synonyms"]['syn_words'], noisy_prob, lang=syn_lang, seed=hash_id)
    elif distortion.startswith("mltlsyn-"):
        noisy_prob = float(distortion.split("-")[1]) / 100
        syn_lang = "en" if lang == "ja" else "ja"
        return synonyms_tokens(tokenizer, text, item["wordnet_synonyms"]['syn_words'], noisy_prob, lang=syn_lang, seed=hash_id)
    elif distortion.startswith("resent-"):
        steps = int(distortion.split("-")[1])
        return reorder_sents(tokenizer, sents, steps=steps, seed=hash_id)
    elif distortion.startswith("rewrite"):
        assert "sentence_rewrite" in item, "Rewrite distortion requires 'sentence_rewrite' in the item."
        noisy_prob = float(distortion.split("-")[1]) / 100
        rewrite_type = distortion.split("-")[0].split("@")[1]
        assert rewrite_type in REWRITE_DICT.keys(), \
            "Unknown rewrite type: {}. Available types: {}".format(rewrite_type, list(REWRITE_DICT.keys()))
        assert REWRITE_DICT[rewrite_type] in item["sentence_rewrite"], "Rewrite distortion requires valid 'rewrite_type' in the item, but got {}".format(rewrite_type)
        
        return rewrite_sents(
            tokenizer, sents, 
            rewrites=item["sentence_rewrite"][REWRITE_DICT[rewrite_type]], 
            rewrite_type=rewrite_type,
            ratio=noisy_prob, seed=hash_id)
    elif distortion.startswith("retoken-"):
        assert "-" in distortion, "Retoken distortion must contain a window size."
        assert "@" in distortion, "Retoken distortion must contain a window size."        
        shuffle_ratio = float(distortion.split("-")[1].split("@")[0]) / 100
        window_size = int(distortion.split("@")[1])
        assert 0 < shuffle_ratio <= 1, "Shuffle ratio must be between 0 and 1. But got: {}".format(shuffle_ratio)
        return reorder_tokens(tokenizer, text, shuffle_ratio=shuffle_ratio, window_size=window_size, seed=hash_id)
    else:
        raise ValueError(f"Unknown distortion: {distortion}")

if __name__ == "__main__":
    import os
    import argparse
    import itertools
    from utils import load_jsonl, dump_jsonl

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--iteration", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--lang", type=str, required=True, choices=["en", "ja", "en_jstage"])
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--method", type=str, default="per_token", choices=["average", "per_token"])
    parser.add_argument(
        "--sequence_types", nargs='+', default=["full"],
        choices=[
            "full", 
            "partial-1", "partial-2", "partial-3", "partial-4", 
            "partial-12", "partial-13", "partial-14", 
            "partial-23", "partial-24", "partial-34"])
    parser.add_argument(
        "--distortions", nargs='+', default=["original"],
        choices=[
            "original",
            "mask-2", "mask-4", "mask-8", "mask-16", "mask-32",
            "delete-2", "delete-4", "delete-8", "delete-16", "delete-32",
            "replace-2", "replace-4", "replace-8", "replace-16", "replace-32",
            "resent-2", "resent-3", "resent-4", 
            "retoken-2@3", "retoken-4@3", "retoken-8@3", "retoken-16@3", "retoken-32@3", 
            "retoken-4@1", "retoken-4@2", "retoken-4@4", "retoken-4@8", "retoken-4@16",
            "monosyn-8", "monosyn-16", "monosyn-21", "monosyn-32", "monosyn-44", "monosyn-64", "monosyn-94", "monosyn-100",
            "mltlsyn-8", "mltlsyn-16", "mltlsyn-32", "mltlsyn-34", "mltlsyn-64", "mltlsyn-72", "mltlsyn-100", 
            "rewrite@syntax-20", "rewrite@syntax-40", "rewrite@syntax-60", "rewrite@syntax-80", "rewrite@syntax-100", 
            "rewrite@lexicon-20", "rewrite@lexicon-40", "rewrite@lexicon-60", "rewrite@lexicon-80", "rewrite@lexicon-100", 
            "rewrite@semantic-20", "rewrite@semantic-40", "rewrite@semantic-60", "rewrite@semantic-80", "rewrite@semantic-100", 
            "rewrite@wordtrans-20", "rewrite@wordtrans-40", "rewrite@wordtrans-60", "rewrite@wordtrans-80", "rewrite@wordtrans-100", 
            "rewrite@translation-20", "rewrite@translation-40", "rewrite@translation-60", "rewrite@translation-80", "rewrite@translation-100", 
        ])

    args = parser.parse_args()
    assert len(args.sequence_types) == 1 or len(args.distortions) == 1, "Only one sequence type or distortion is allowed for now."

    print("========== args ==========")
    print(f"Model name: {args.model_name}")
    print(f"Iteration: {args.iteration}")
    print(f"Model path: {args.model_path}")
    print(f"Language: {args.lang}")
    print(f"Distortion: {args.distortions}")
    print(f"Sequence type: {args.sequence_types}")
    print(f"Method: {args.method}")
    
    params = []
    for sequence_type, distortion in itertools.product(args.sequence_types, args.distortions):
        dump_root = f"/data/xzhao/experiments/med-eval/dataset/losses/results/random_2k/{args.lang}_{sequence_type}_{distortion}"
        dump_root = os.path.join(dump_root, args.model_name)
        dump_file = os.path.join(dump_root, f"{args.iteration}.jsonl")

        if os.path.exists(dump_file):
            print(f"SKIP: {dump_file}")
        else:
            params.append((sequence_type, distortion))
    
    if len(params) == 0:
        exit(0)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")
    model.eval()

    assert UNK_TOKEN_ID == tokenizer.unk_token_id
    train_samples_root = "/data/xzhao/experiments/med-eval/dataset/losses/train_samples"
    filename_dict = {
        # "en": os.path.join(train_samples_root, "random_en_2k.withsyn.jsonl"),
        # "ja": os.path.join(train_samples_root, "random_ja_2k.withsyn.jsonl"),
        # "en_jstage": os.path.join(train_samples_root, "random_en_jstage_2k.withsyn.jsonl"),
        "en": os.path.join(train_samples_root, "random_en_2k.rewrite.jsonl"),
        "ja": os.path.join(train_samples_root, "random_ja_2k.rewrite.jsonl"),
        "en_jstage": os.path.join(train_samples_root, "random_en_jstage_2k.rewrite.jsonl"),
    }
    assert args.lang in filename_dict, f"Unknown language: {args.lang}"
    path = filename_dict[args.lang]
    if args.num_samples < 2000:
        items = load_jsonl(path)[:args.num_samples]
    else:
        items = load_jsonl(path)

    for sequence_type, distortion in params:
        dump_root = f"/data/xzhao/experiments/med-eval/dataset/losses/results/random_2k/{args.lang}_{sequence_type}_{distortion}"
        dump_root = os.path.join(dump_root, args.model_name)
        os.makedirs(dump_root, exist_ok=True)
        dump_file = os.path.join(dump_root, f"{args.iteration}.jsonl")
    
        print(f"\n=====> Evaluating distortion: {distortion}")
        print(f"Dump file: {dump_file}")
        print(f"Model name: {args.model_name}")
        print(f"Iteration: {args.iteration}")
        print(f"Model path: {args.model_path}")
        print(f"Language: {args.lang}")
        print(f"Distortion: {distortion}")
        print(f"Sequence type: {sequence_type}")
        print(f"Method: {args.method}")
        
        try:
            if args.method == "average":
                seq2loss, avg_loss = caluate_loss_single(
                    model, tokenizer, items, args.lang,
                    sequence_type=sequence_type, 
                    distortion=distortion)
            elif args.method == "per_token":
                seq2loss, avg_loss = caluate_loss_single_per_token(
                    model, tokenizer, items, args.lang,
                    sequence_type=sequence_type, 
                    distortion=distortion)
            
            dump_jsonl(seq2loss, dump_file, pretty=False)
            print(f"Dumped to {dump_file}")
        except Exception as e:
            import traceback
            print(f"Error occurred: {e}")
            traceback.print_exc()
            print(f"Failed to evaluate distortion: {distortion}")
            continue
