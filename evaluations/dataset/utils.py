import os
import re
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

prompt_tasks = [
        "nii_en5_mono_prompt-en", "nii_en5_mono_prompt-ja",
        "nii_en5_bi_prompt-en", "nii_en5_bi_prompt-ja",
        "nii_en5_tri_prompt-en", "nii_en5_tri_prompt-ja",
        "nii_ja5_mono_prompt-en", "nii_ja5_mono_prompt-ja",
        "nii_ja5_bi_prompt-en", "nii_ja5_bi_prompt-ja",
        "nii_ja5_tri_prompt-en", "nii_ja5_tri_prompt-ja"
        ]

def parse_train_info(model_name):
    if "exp6" in model_name or "exp7" in model_name or "exp8" in model_name:
        train_domain, train_lang = "jstage", "en"
    elif model_name.endswith("-ja"):
        train_domain, train_lang = "jstage", "ja"
    elif model_name.endswith("-en"):
        train_domain, train_lang = "pubmed", "en"
    elif model_name.endswith("-en_jstage"):
        train_domain, train_lang = "jstage", "en"
    elif model_name.endswith("exp2-multi"):
        train_domain, train_lang = "mixed", "mixed"
    elif model_name.endswith("-mix"):
        train_domain, train_lang = "mixed", "mixed"
    elif model_name.endswith("overlap"):
        train_domain, train_lang = "jstage", "ja"
    else:
        raise ValueError("Invalid model name {}".format(model_name))
    return train_domain, train_lang

def parse_eval_info_old(task_name):
    if task_name.startswith("nii_en"):
        eval_domain = "pubmed"
    elif task_name.startswith("nii_ja"):
        eval_domain = "jstage"
    else:
        raise ValueError("Invalid task name {}".format(task_name))
    
    if "mono" in task_name:
        eval_type = "KP"
    elif "bi" in task_name:
        eval_type = "LQ"
    elif "tri" in task_name:
        eval_type = "RQ"
    else:
        raise ValueError("Invalid task name {}".format(task_name))
    
    if task_name.endswith("-en"):
        eval_lang = "en"
    elif task_name.endswith("-ja"):
        eval_lang = "ja"
    else:
        raise ValueError("Invalid task name {}".format(task_name))
    return eval_domain, eval_type, eval_lang

def parse_eval_info(task_name):
    assert task_name.startswith("adaxeval-"), task_name
    task_name = task_name[len("adaxeval-"):]

    eval_domain = "jstage"
    if task_name.startswith("en_"):
        eval_lang = "en"
    elif task_name.startswith("ja"):
        eval_lang = "ja"
    else:
        raise ValueError("Invalid task name {}".format(task_name))
    
    if task_name.endswith("memorization"):
        eval_type = "mem"
    elif task_name.endswith("generalization"):
        eval_type = "gen"
    else:
        raise ValueError("Invalid task name {}".format(task_name))    
    return eval_domain, eval_type, eval_lang

# def get_task_name(train_lang, eval_lang, train_domain, eval_domain):
#     task_name = "AdaXEval"
#     if train_lang == eval_lang:
#         task_name += "-native"
#     else:
#         task_name += "-CLT"
    
#     if train_domain != eval_domain:
#         task_name += "-OOD"
#     return task_name

def get_task_name(train_lang, eval_lang, train_domain, eval_domain):
    task_name = "AdaXEval"
    if eval_lang == "ja":
        task_name += "-Ja"
    else:
        task_name += "-En"
    
    if train_domain != eval_domain:
        task_name += "-OOD"
    return task_name

def read_accs_from_folder(result_dir, model_name, is_ood=True, delta_acc=False, namedicts=None):
    accs = []
    for filename in os.listdir(result_dir):
        # print(f"Processing {filename} ...")
        if not filename.endswith(".csv"):
            print(f"Skip {filename} as it is not a csv file")
            continue
        result_filename = os.path.join(result_dir, filename)
        if not os.path.isfile(result_filename):
            print(f"Skip {filename} as it is not a file")
            continue 
        
        matches = re.match(r"iter_(\d+)-prompt\.csv", filename)
        if not matches:
            print(f"Skip {filename} as it does not match the expected pattern")
            continue

        iter_num = int(matches.group(1))
        if iter_num <= 100 and iter_num > 0:
            continue
        # print(model_name, result_filename)
        with open(result_filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            for row in reader:
                task, acc = row[0], row[3]
                train_domain, train_lang = parse_train_info(model_name)
                eval_domain, eval_type, eval_lang = parse_eval_info(task)
                # print(train_domain, eval_domain)
                # if train_domain is not "mixed":
                #     if not is_ood and eval_domain != train_domain:
                #         # print(f"Skip {filename} as eval_domain {eval_domain} is not equal to train_domain {train_domain}")
                #         continue
                #     elif is_ood and eval_domain == train_domain:
                #         # print(f"Skip {filename} as eval_domain {eval_domain} is equal to train_domain {train_domain}")
                #         continue
                accs.append({
                    "model_name": namedicts[model_name] if namedicts is not None else model_name,
                    "iter_num": iter_num,
                    "train_domain": train_domain,
                    "train_lang": train_lang,
                    "eval_domain": eval_domain,
                    "eval_type": eval_type,
                    "eval_lang": eval_lang,
                    "task": get_task_name(train_lang, eval_lang, train_domain, eval_domain),
                    "acc": round(float(acc), 4),
                })
    df = pd.DataFrame(accs)
    return df
    delta_accs = []
    for idx, row in df.iterrows():
        _df = df[
            (df["eval_type"] == row["eval_type"]) & 
            # (df["task"] == row["task"]) & 
            (df["eval_domain"] == row["eval_domain"]) & 
            (df["eval_lang"] == row["eval_lang"]) & 
            (df["iter_num"] == 0)]
        assert len(_df) == 1, f"There should be only one row for the baseline, {row['eval_type']} {row['eval_domain']} {row['eval_lang']} {row['iter_num']}"
        if delta_acc:
            delta_accs.append(row['acc'] - _df.iloc[0]['acc'])
        else:
            delta_accs.append(row['acc'])
    df["delta_acc"] = delta_accs
    return df

def plot_acc_merged(df, is_ood, delta_acc, title=None):
    # Create the line plot
    plt.figure(figsize=(5, 3.75), constrained_layout=True)
    sns.lineplot(
        data=df, 
        x='iter_num', 
        y='delta_acc', 
        hue='task', 
        style='eval_type', 
        markers=True,
        hue_order=['AdaXEval-native', 'AdaXEval-CLT'] if not is_ood else ['AdaXEval-native-OOD', 'AdaXEval-CLT-OOD'],
        palette={'AdaXEval-native': '#F9A037', 'AdaXEval-CLT': '#3D7AB3', 'AdaXEval-native-OOD': '#F9A037', 'AdaXEval-CLT-OOD': '#3D7AB3'}
    )

    # Add vertical dotted lines at specific iteration numbers
    # plt.axvline(x=3841, color='red', linestyle='dotted', label='1st Epoch')
    # plt.axvline(x=15327, color='red', linestyle='dotted', label='4th Epoch')

    plt.title(title)
    plt.xlabel("Iteration Number")
    if delta_acc:
        plt.ylabel("Accuracy Difference to Baseline")
    else:
        plt.ylabel("Accuracy")
    plt.grid(True)
    # Custom separate legends
    handles, labels = plt.gca().get_legend_handles_labels()

    # Separate handles/labels
    task_handles = handles[1:3]  # Adjust slicing based on your data
    task_labels = labels[1:3]
    eval_handles = handles[3:-1]   # Remaining ones
    eval_labels = labels[3:-1]
    epoch_handles = handles[-1:]
    epoch_labels = labels[-1:]
    
    # First legend for tasks
    legend1 = plt.legend(
        task_handles, 
        task_labels, 
        loc='lower center',          # Center it horizontally
        bbox_to_anchor=(0.7, -0.02),   # Place it below x-axis
        ncol=1,                      # Two columns
        fontsize=10
    )
    # Second legend for eval_type
    plt.gca().add_artist(legend1)  # Keep the first legend
    legend2 = plt.legend(eval_handles, eval_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.gca().add_artist(legend2)  # Keep the first legend
    plt.legend(epoch_handles, epoch_labels, loc='center left', bbox_to_anchor=(1, 0.25), fontsize=10)
    plt.tight_layout()
    plt.show()

import os
import json
from pdb import set_trace
from traitlets import default
import yaml

DATA_ROOT = "/data/xzhao/dataset/roman-pretrain"
EXP_ROOT = "/data/xzhao/experiments/roman-pretrain"
PROJECT_ROOT = "/home/xzhao/workspace/roman-pretrain"

def load_config(config_name):
    config_root = os.path.join(PROJECT_ROOT, "experiments", "configs", f"{config_name}.yaml")
    if not os.path.exists(config_root): 
        raise FileNotFoundError(f"Config file: {config_root} doesn't exist")
    with open(config_root) as stream:
        exp_confs = yaml.safe_load(stream)

    default_config_root = exp_confs["default_config"]
    with open(default_config_root) as stream:
        confs = yaml.safe_load(stream)
    
    def overwrite_dict(d1, d2):
        for k, v in d2.items():
            # FIXME: Patch used for medical-xxx field overwrite. The exp-config may set a dictionary to this field while the original is a float.
            if isinstance(v, dict) and k in d1 and not isinstance(d1[k], dict):
                d1[k] = v
                continue

            if isinstance(v, dict) and k in d1:
                overwrite_dict(d1[k], v)
            else:
                d1[k] = v
                
    overwrite_dict(confs, exp_confs)
    return confs
        
def load_jsonl(filename):
    assert os.path.exists(filename), filename
    items = []
    with open(filename, 'r', encoding="utf8") as fp:
        for line in fp:
            items.append(json.loads(line))
    return items

def load_jsonl_iteratively(filename, request_num=None, start_indice=0):
    assert os.path.exists(filename), filename
    i = 0
    with open(filename, 'r', encoding="utf8") as fp:
        for j, line in enumerate(fp):
            if j < start_indice:
                continue
            if request_num is not None and i>=request_num:
                break
            try:
                item = json.loads(line)
            except Exception as e:
                print(e, line)
                continue
            finally:
                i += 1
            
            yield item


def dump_jsonl(obj, filename, pretty=False):
    with open(filename, 'w', encoding="utf8") as fp:
        for item in obj:
            if pretty:
                string = json.dumps(item, indent=2, ensure_ascii=False)
            else:
                string = json.dumps(item, ensure_ascii=False)
            fp.write(f"{string}\n")


def load_json(filename, create_if_nonexist=False):
    if not os.path.exists(filename):
        if create_if_nonexist:
            dump_json({}, filename)
            return {}
        else:
            raise FileNotFoundError(f"{filename} is not found")
    
    with open(filename, 'r', encoding="utf8") as fp:
        return json.load(fp)

def dump_json(obj, filename, pretty=False):
    with open(filename, 'w', encoding="utf8") as fp:
        if pretty:
            json.dump(obj, fp, indent=2, ensure_ascii=False)
        else:
            json.dump(obj, fp, ensure_ascii=False)

