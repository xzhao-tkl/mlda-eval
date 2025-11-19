import os
import sys
import json
from tqdm import tqdm
from utils import DATA_ROOT, dump_json, load_jsonl_iteratively, load_config


def write_text(src, tgt, num_instructions, instruction_type, record_ids=False, docid_fn=None, filter_docids=None):
    """
    Write text to tgt file from src file, with a limit of num_instructions.
    - `docid_fn`: if record_ids is True, the docid_fn will be used to write the doc ids
    - `filter_docids`: if record_ids is False, the docids will be used to filter the items
    """
    if record_ids:
        assert filter_docids is None and instruction_type is not None, "docids shouldn't and lang should be provided if record_ids is True"
        docid_fp = open(docid_fn, 'a', encoding="utf8") if record_ids else None
        print(f"Writing doc ids to {docid_fn}")
    
    if filter_docids is not None:
        assert isinstance(filter_docids, set), "docids should be a set"
        assert record_ids is False, "docids shouldn't be provided if record_ids is True"
        

    with open(tgt, 'a', encoding="utf8") as fp:
        desc = f"DATA_ROOT{src[len(DATA_ROOT):]} → DATA_ROOT{tgt[len(DATA_ROOT):]}"
        for cnt, item in tqdm(enumerate(load_jsonl_iteratively(src, request_num=num_instructions)), desc=desc):                        
            if filter_docids is not None and item['docid'] not in filter_docids:
                continue
            docid_fp.write(f"{json.dumps({'docid': item['docid'], 'type': instruction_type}, ensure_ascii=False)}\n")
            new_item = {"docid": item['docid'], "type": instruction_type}
            new_item.update(item)
            string = json.dumps(new_item, ensure_ascii=False)
            fp.write(f"{string}\n")
    print(f"Finished writing {cnt} instructions to {tgt} from {src}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str)
    parser.add_argument("--update_config", action="store_true")
    args = parser.parse_args()    
    
    config = load_config(args.config_name)
    assert args.config_name == config['name'], f"config_name should be the same as the config file name, but got {args.config_name} vs. {config['name']}"
    data_root = config['dataset']['dataset-dir']
    
    exp_root = os.path.join(config['exp-dir'], config['name'])
    save_dir = os.path.join(exp_root, "dataset")
    os.makedirs(save_dir, exist_ok=True)
    dump_json(config, f"{exp_root}/config.json", pretty=True)

    if args.update_config:
        sys.exit(0)

    collections = config['dataset']['collection']
    assert 'instruction-tuning' in collections, "instruction-tuning should be in collections"
    tuning_config = collections['instruction-tuning']
    tokenizer_type = config['dataset']['tokenizer']['type']

    data_dirs = []
    for instruction_type in tuning_config.keys():
        data_suffix = None
        data_filepath = None
        filtered_docid_fn = None
        assert isinstance(tuning_config[instruction_type], dict), f"tuning_config for {instruction_type} should be a dict"
        this_config = tuning_config[instruction_type]
        assert 'data_filepath' in this_config
        if "doc_ids" in this_config:
            filtered_docid_fn = this_config['doc_ids']
        
        data_filepath = this_config['data_filepath']
        data_suffix = this_config['data_suffix']
        num_instructions = this_config['num_instructions']
        assert num_instructions is None or isinstance(num_instructions, int), f"num_instructions should be None or int, but got {num_instructions}"
        assert isinstance(data_suffix, str), f"data_suffix should be None or str, but got {data_suffix}"
        assert os.path.exists(data_filepath), f"data_filepath {data_filepath} doesn't exist"
        
        number = "all" if num_instructions is None else num_instructions
        data_dir = os.path.join(config['dataset']['it-dataset-dir'], f'{instruction_type}-{data_suffix}-{number}')
        os.makedirs(data_dir, exist_ok=True)

        src_data_fn = os.path.join(data_dir, "instructions.jsonl")
        src_docid_fn = os.path.join(data_dir, "doc_ids.jsonl")
        if not os.path.exists(src_data_fn):
            print(f"Instruction dataset not found at {data_dir}, generating...")
            print("===> Reading instructions from: ", data_filepath)
            if filtered_docid_fn is None:
                write_text(
                    data_filepath, src_data_fn, 
                    num_instructions=num_instructions, instruction_type=data_suffix, 
                    filter_docids=None, record_ids=True, docid_fn=src_docid_fn)
            else:
                filtered_docids = set([item['docid'] for item in load_jsonl_iteratively(filtered_docid_fn)])
                write_text(
                    data_filepath, src_data_fn, 
                    num_instructions=num_instructions, instruction_type=data_suffix, 
                    filter_docids=filtered_docids, record_ids=True, docid_fn=src_docid_fn)
                os.symlink(filtered_docid_fn, src_docid_fn)
        else:
            print(f"Instruction dataset already exists at {data_dir}")

        tgt_data_fn = os.path.join(save_dir, f"{instruction_type}-instructions.jsonl")
        if os.path.islink(tgt_data_fn):
            os.unlink(tgt_data_fn)
        os.symlink(src_data_fn, tgt_data_fn)
        print(f"Linking {src_data_fn} to {tgt_data_fn}")


