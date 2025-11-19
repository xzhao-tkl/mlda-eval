import os
from tqdm import tqdm

from utils import load_jsonl_iteratively


def get_model_loss(lang, eval_type, num_iter, sample_ids=None, crosslang=False):
    # Collect the loss information from adaxeval evaluation result.json
    id2loss = {}
    if not crosslang:
        exp_lang = "ja" if lang == "ja" else "en_jstage"
    else:
        exp_lang = "en_jstage" if lang == "ja" else "ja"
    model_root = f"/data/xzhao/experiments/roman-pretrain/exps/exp1-multi-{exp_lang}/hf_models"
    if num_iter == "last":
        checkpoint_folder = "iter_0015364" if exp_lang == "ja" else "iter_0015354"
    
    elif isinstance(int(num_iter), int):
        checkpoint_folder = f"iter_{int(num_iter):07d}"
    checkpoint_folder = os.path.join(model_root, checkpoint_folder)
    assert os.path.exists(checkpoint_folder), f"Checkpoint folder does not exist: {checkpoint_folder}"
    # print(f"Loading losses from {checkpoint_folder} ...")

    target_dataset = f"{lang}_knowledge_{eval_type}"
    result_filename = os.path.join(checkpoint_folder, "adaxeval.jsonl")
    for item in load_jsonl_iteratively(result_filename):
        sample_id = item["sample"]["sample_id"]
        if item["sample"]['dataset'] != target_dataset:
            continue
        
        if sample_ids is not None and sample_id not in sample_ids:
            continue

        id2loss[sample_id] = {
            "total_loss": item['losses'], 
            "norm_loss": item['norm_losses'],
            "answer_idx": item["sample"]["answer_idx"]
        }
    return id2loss