import os
import json
import yaml
from tqdm import tqdm

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
        
def load_jsonl(filename, verbose=False):
    assert os.path.exists(filename), filename
    items = []

    with open(filename, 'r', encoding="utf8") as fp:
        if verbose:
            for line in tqdm(fp, desc=f"Loading JSONL from {filename}"):
                items.append(json.loads(line))
        else:
            for line in fp:
                items.append(json.loads(line))
    return items

def load_jsonl_iteratively(filename, request_num=None, start_indice=0, verbose=False):
    assert os.path.exists(filename), filename
    i = 0
    with open(filename, 'r', encoding="utf8") as fp:
        if verbose:
            for j, line in enumerate(tqdm(fp)):
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
        else:
            for j, line in enumerate(fp):
                if j < start_indice:
                    continue
                if request_num is not None and i >= request_num:
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

