import os
import json


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