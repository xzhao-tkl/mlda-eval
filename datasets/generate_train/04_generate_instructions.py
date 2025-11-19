from pdb import set_trace
import re
import os
import sys
import json
import random
from typing import Tuple

import torch
from tqdm import tqdm
from utils import DATA_ROOT, load_json, load_jsonl_iteratively


def decide_text_type(native=False, roman=False, halfroman=False, halfroman_reverse=False, noisy=False):
    base_type = "noisy" if noisy else "raw"
    assert int(native) + int(roman) + int(halfroman) + int(halfroman_reverse) <= 1
    if halfroman_reverse:
        input_type, output_type = base_type, "roman"
    elif halfroman:
        input_type, output_type = "roman", base_type
    elif roman:
        input_type, output_type = "roman", "roman"
    else:
        input_type, output_type = base_type, base_type
    return input_type, output_type

def get_meta_for_process(item, lang, input_type, output_type):
    if input_type == "roman" and output_type == "roman":
        lang = lang + "-roman"
    
    input_data = item[input_type] if input_type != "noisy" else item[input_type][args.noisy_index]
    output_data = item[output_type] if output_type != "noisy" else item[output_type][args.noisy_index]
    return lang, input_data, output_data

def process_keywords(item):
    spliters = [',', '.', '/', '・', '、', '／']
    regex_pattern = '|'.join(map(re.escape, spliters))
    
    keywords = []
    for keyword in item['raw']['keywords']:
        _keywords = [key for key in re.split(regex_pattern, keyword) if len(key.strip()) > 0]
        keywords.extend(_keywords)
    item['raw']['keywords'] = keywords

    if "roman" not in item:
        return item

    keywords = []
    for keyword in item['roman']['keywords']:
        _keywords = [key for key in re.split(regex_pattern, keyword) if len(key.strip()) > 0]
        keywords.extend(_keywords)
    item['roman']['keywords'] = keywords
    return item
    
def generate_summarization(item, lang, input_type, output_type):
    lang, input_data, output_data = \
        get_meta_for_process(item, lang, input_type, output_type)

    template = random.choice(templates["summarization"][lang])
    if template.index("{abstract}") > template.index("{title}"):
        return template.format(
            abstract=output_data['abstract'],
            title=input_data['title'], )
    else:
        try:
            return template.format(
                abstract=input_data['abstract'],
                title=output_data['title'], )
        except Exception as e:
            set_trace()
        
def generate_textcomplete(item, lang, input_type, output_type):
    lang, input_data, output_data = \
        get_meta_for_process(item, lang, input_type, output_type)

    template = random.choice(templates["text-completion"][lang])
    indice = 1 if len(item['raw']['sentences']) == 2 else random.randint(1, len(item['raw']['sentences']) - 1)
    try:
        return template.format(
            front=" ".join(input_data['sentences'][:indice]),
            behind=" ".join(output_data['sentences'][indice:]))
    except Exception as e:
        print("Skip error in generate_textcomplete by return empty string")
        return ""
    
def generate_conclusion(item, lang, input_type, output_type):
    lang, input_data, output_data = \
        get_meta_for_process(item, lang, input_type, output_type)

    template = random.choice(templates["conclusion"][lang])
    return template.format(
        input=input_data['conclusion'][0], 
        abstract=output_data['conclusion'][1])

def generate_qa(item, lang, input_type, output_type, with_context=False):
    lang, input_data, output_data = \
        get_meta_for_process(item, lang, input_type, output_type)

    qa_instructions = []
    for i in range(len(item['raw']['qa'])):
        template = random.choice(templates["qa"][lang])
        question = input_data['qa'][i][0]
        answer = output_data['qa'][i][1]
        qa_instructions.append(template.format(question=question, answer=answer))
    if not with_context: 
        return qa_instructions
    else:
        return [input_data['abstract'] + "\n" + qa for qa in qa_instructions]

def generate_romanization(item, lang, native2roman=False):
    keyword = "native2roman" if native2roman else "roman2native"
    template = random.choice(templates[keyword][lang])
    romanization_instructions = []
    for native, roman in zip(item['raw']['sentences'], item['roman']['sentences']):
        template = random.choice(templates[keyword][lang])
        romanization_instructions.append(template.format(native=native, roman=roman))
    template = random.choice(templates[keyword][lang])
    romanization_instructions.append(template.format(native=item['raw']['title'], roman=item['roman']['title']))
    return romanization_instructions

def generate_pure_text(item, lang, roman):
    if not roman:
        return item[lang]
    else:
        return item[f'{lang}-roman']

def generate_text2key(item, lang, input_type, output_type):
    lang, input_data, output_data = \
        get_meta_for_process(item, lang, input_type, output_type)

    template = random.choice(templates["keywords-extraction"][lang])
    return template.format(
        abstract=f"{input_data['title']}. {input_data['abstract']}", 
        keywords=", ".join(output_data['keywords']))
    
def generate_key2text(item, lang, input_type, output_type):
    lang, input_data, output_data = \
        get_meta_for_process(item, lang, input_type, output_type)

    template = random.choice(templates["keywords-to-text"][lang])
    return template.format(
        abstract=f"{output_data['title']}. {output_data['abstract']}", 
        keywords=", ".join(input_data['keywords']))

def generate_causal(item, lang, input_type, output_type):
    lang, input_data, output_data = \
        get_meta_for_process(item, lang, input_type, output_type)

    causal_instructions = []
    for rel in item['raw']['causal']:
        for i, pair in enumerate(item['raw']['causal'][rel]):
            template = random.choice(templates[f"causal-{rel}"][lang])
            causal_instructions.append(
                template.format(
                    sentence1=input_data['causal'][rel][i][0], 
                    sentence2=output_data['causal'][rel][i][1]))
    return causal_instructions

def generate_gmrc(item, lang, input_type, output_type):
    lang, input_data, output_data = \
        get_meta_for_process(item, lang, input_type, output_type)

    causal_instructions = []
    for rel in item['raw']['gmrc']:
        template = random.choice(templates[f"gmrc-{rel}"][lang])
        target = output_data['gmrc'][rel]
        others = [input_data['gmrc'][_rel] for _rel in input_data['gmrc'] if _rel != rel]
        causal_instructions.append(
            template.format(input=" ".join(others), output=target))
    return causal_instructions

def generate_medical_translation(item, lang, en_text, roman, native2en=False):
    output_type = "roman" if roman else "raw"
    ja_text = f"{item[output_type]['title']} {item[output_type]['abstract']}"
    if native2en:
        temp = 'roman2en' if roman else 'native2en'
        template = random.choice(templates[temp][lang])
        return template.format(input=ja_text, output=en_text)
    else:
        temp = 'en2roman' if roman else 'en2native'
        template = random.choice(templates[temp][lang])
        return template.format(input=en_text, output=ja_text)

def generate_other_translation(item, lang, roman=False, x2en=True):
    x_text = item[lang] if roman is False else item[f'{lang}-roman']
    en_text = item['en']

    if x2en:
        temp = 'roman2en' if roman else 'native2en'
        template = random.choice(templates[temp][lang])
        return template.format(input=x_text, output=en_text)
    else:
        temp = 'en2roman' if roman else 'en2native'
        template = random.choice(templates[temp][lang])
        return template.format(input=en_text, output=x_text)

def generate_other_romanization(item, lang, roman2native):
    if roman2native:
        template = random.choice(templates["roman2native"][lang])
    else:
        template = random.choice(templates["native2roman"][lang])
    return template.format(native=item[lang], roman=item[f'{lang}-roman'])

def serialize(text, docid=None, type=None):
    if docid is None and type is None:
        return f"{json.dumps({'text': text}, ensure_ascii=False)}\n"
    else:
        return f"{json.dumps({'docid': docid, 'text': text, 'type': type}, ensure_ascii=False)}\n"


if __name__ == "__main__":
    import random
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang", type=str, default="zh", 
        choices=['zh', 'ja', 'en', 'en_jstage'])
    parser.add_argument("--rewrite", action="store_true")
    parser.add_argument("--noisy_index", type=int, default=0)

    parser.add_argument("--medical_noninstr", action="store_true")
    parser.add_argument("--medical_native", action="store_true")
    parser.add_argument("--medical_roman", action="store_true")
    parser.add_argument("--medical_en2roman", action="store_true")
    parser.add_argument("--medical_roman2en", action="store_true")
    parser.add_argument("--medical_native2roman", action="store_true")
    parser.add_argument("--medical_roman2native", action="store_true")
    parser.add_argument("--medical_trans", action="store_true")
    parser.add_argument("--medical_halfroman", action="store_true")
    parser.add_argument("--medical_halfroman_reverse", action="store_true")

    parser.add_argument("--balanced_native", action="store_true")
    parser.add_argument("--balanced_roman", action="store_true")
    parser.add_argument("--balanced_en2roman", action="store_true")
    parser.add_argument("--balanced_roman2en", action="store_true")
    parser.add_argument("--balanced_native2roman", action="store_true")
    parser.add_argument("--balanced_roman2native", action="store_true")
    parser.add_argument("--balanced_trans", action="store_true")
    
    parser.add_argument("--science_native", action="store_true")
    parser.add_argument("--science_roman", action="store_true")
    parser.add_argument("--science_en2roman", action="store_true")
    parser.add_argument("--science_roman2en", action="store_true")
    parser.add_argument("--science_native2roman", action="store_true")
    parser.add_argument("--science_roman2native", action="store_true")
    parser.add_argument("--science_trans", action="store_true")
    parser.add_argument("--noisy_instruction", action="store_true")
    parser.add_argument("--noisy_data_path", type=str, default="")
    args = parser.parse_args()

    assert int(args.medical_native) + int(args.medical_roman) + int(args.medical_en2roman) + int(args.medical_roman2en) + \
            int(args.medical_native2roman) + int(args.medical_roman2native) + int(args.medical_trans) + \
            int(args.medical_halfroman) + int(args.medical_halfroman_reverse) + \
            int(args.balanced_native) + int(args.balanced_roman) + int(args.balanced_en2roman) + int(args.balanced_roman2en) + \
            int(args.balanced_native2roman) + int(args.balanced_roman2native) + int(args.balanced_trans) + \
            int(args.science_native) + int(args.science_roman) + int(args.science_en2roman) + int(args.science_roman2en) + \
            int(args.science_native2roman) + int(args.science_roman2native) + int(args.science_trans) == 1

    if args.noisy_instruction: 
        assert args.medical_native is True, "Currently only medical_native noisy instruction generation is supported"
    
    if not args.noisy_instruction:
        if args.lang == "en":
            full_items_fn = os.path.join(DATA_ROOT, "datasets", "medical", args.lang, "preprocessed.jsonl")
        elif args.lang == "ja" or args.lang == "zh" or args.lang == "en_jstage":
            full_items_fn = os.path.join(DATA_ROOT, "datasets", "medical", args.lang, "full.jsonl")
        else:
            raise NotImplementedError(f"Language {args.lang} is not supported")
    else:
        assert args.noisy_data_path != "", "Please provide --noisy_data_path for noisy instruction generation"
        full_items_fn = args.noisy_data_path

    assert os.path.exists(full_items_fn), f"File {full_items_fn} does not exist"
    if not args.medical_native:
        assert args.lang in ["ja", "zh"], f"Language {args.lang} is not supported for non-native instruction creation"
    
    templates = load_json("./instructions/instruction-templates.new.json")
    
    output_dir = os.path.join(DATA_ROOT, "noisy_instructions", args.lang) if args.noisy_instruction else os.path.join(DATA_ROOT, "instructions", args.lang)
    
    os.makedirs(output_dir, exist_ok=True)
    print("===> Output directory created at:", output_dir)

    # if args.lang != "ja" and (args.native2en or args.roman2en):
    #     raise ValueError(f"Translation can only be applied to Japanese, but not {args.lang}")
    
    if args.medical_native:
        file_name = os.path.join(output_dir, "medical_native.jsonl")
    elif args.medical_noninstr:
        file_name = os.path.join(output_dir, "medical_noninstr.jsonl")
    elif args.medical_roman:
        file_name = os.path.join(output_dir, "medical_roman.jsonl")
    elif args.medical_en2roman:
        file_name = os.path.join(output_dir, "medical_en2roman.jsonl")
    elif args.medical_roman2en:
        file_name = os.path.join(output_dir, "medical_roman2en.jsonl")
    elif args.medical_native2roman:
        file_name = os.path.join(output_dir, "medical_native2roman.jsonl")
    elif args.medical_roman2native:
        file_name = os.path.join(output_dir, "medical_roman2native.jsonl")
    elif args.medical_trans:
        file_name = os.path.join(output_dir, "medical_trans.jsonl")
    
    elif args.medical_halfroman:
        file_name = os.path.join(output_dir, "medical_halfroman.jsonl")
    elif args.medical_halfroman_reverse:
        file_name = os.path.join(output_dir, "medical_halfroman_reverse.jsonl")
    

    elif args.balanced_native:
        file_name = os.path.join(output_dir, "balanced_native.jsonl")
    elif args.balanced_roman:
        file_name = os.path.join(output_dir, "balanced_roman.jsonl")
    elif args.balanced_en2roman:
        file_name = os.path.join(output_dir, "balanced_en2roman.jsonl")
    elif args.balanced_roman2en:
        file_name = os.path.join(output_dir, "balanced_roman2en.jsonl")
    elif args.balanced_native2roman:
        file_name = os.path.join(output_dir, "balanced_native2roman.jsonl")
    elif args.balanced_roman2native:
        file_name = os.path.join(output_dir, "balanced_roman2native.jsonl")
    elif args.balanced_trans:
        file_name = os.path.join(output_dir, "balanced_trans.jsonl")
    
    elif args.science_native:
        file_name = os.path.join(output_dir, "science_native.jsonl")
    elif args.science_roman:
        file_name = os.path.join(output_dir, "science_roman.jsonl")
    elif args.science_en2roman:
        file_name = os.path.join(output_dir, "science_en2roman.jsonl")
    elif args.science_roman2en:
        file_name = os.path.join(output_dir, "science_roman2en.jsonl")
    elif args.science_native2roman:
        file_name = os.path.join(output_dir, "science_native2roman.jsonl")
    elif args.science_roman2native:
        file_name = os.path.join(output_dir, "science_roman2native.jsonl")
    elif args.science_trans:
        file_name = os.path.join(output_dir, "science_trans.jsonl")
    else:
        raise NotImplementedError
    
    if args.noisy_instruction:
        file_basename = os.path.basename(args.noisy_data_path)
        os.makedirs(os.path.join(output_dir, file_basename), exist_ok=True)
        file_name = os.path.join(output_dir,  file_basename, f"{args.noisy_index}.jsonl")
        print(f"Noisy instruction generation, output file: {file_name}")

    if os.path.exists(file_name) and args.rewrite is False:
        print(f"{file_name} already exist")
        sys.exit(0)    

    print(f"Generating instructions to {file_name}")
    output_fp = open(file_name, "w", encoding="utf8")
    
    if args.medical_en2roman or args.medical_roman2en or args.medical_trans \
        or args.medical_native2roman or args.medical_roman2native:
        trans_fn = os.path.join(DATA_ROOT, "datasets", "medical", args.lang, "trans.jsonl")
        en_items = {}
        for item in load_jsonl_iteratively(trans_fn):
            en_items[item["docid"]] = {
                "title": item["title"],
                "abstract": item["abstract"]
            }
            
        for item in tqdm(load_jsonl_iteratively(full_items_fn)):
            if item["docid"] not in en_items:
                continue

            en_title = en_items[item["docid"]]["title"]
            en_abstract = en_items[item["docid"]]["abstract"]
            en_text = f"{en_title} {en_abstract}"
            x2en = generate_medical_translation(item, args.lang, en_text, roman=args.medical_roman2en, native2en=True)
            en2x = generate_medical_translation(item, args.lang, en_text, roman=args.medical_en2roman, native2en=False)
            if args.medical_trans:
                output_fp.write(serialize(x2en, docid=item["docid"], type="native2en"))
                output_fp.write(serialize(en2x, docid=item["docid"], type="en2native"))
            elif args.medical_roman2en:
                output_fp.write(serialize(x2en, docid=item["docid"], type="roman2en"))
            elif args.medical_en2roman:
                output_fp.write(serialize(en2x, docid=item["docid"], type="en2roman"))
            elif args.medical_native2roman:
                romans = generate_romanization(item, args.lang, native2roman=True)
                for roman in romans:
                    output_fp.write(serialize(roman, docid=item["docid"], type="medical_native2roman"))
            elif args.medical_roman2native:
                romans = generate_romanization(item, args.lang, native2roman=False)
                for roman in romans:
                    output_fp.write(serialize(roman, docid=item["docid"], type="medical_roman2native"))
            else:
                raise ValueError
        output_fp.close()
        sys.exit()

    if args.balanced_trans or args.balanced_native or args.balanced_roman \
            or args.balanced_en2roman or args.balanced_roman2en \
            or args.balanced_native2roman or args.balanced_roman2native:
        full_fn = os.path.join(DATA_ROOT, "datasets", "balanced_bilingual", f"en-{args.lang}", "full.jsonl")
        for item in tqdm(load_jsonl_iteratively(full_fn)):
            if args.balanced_trans:
                x2en = generate_other_translation(item, args.lang, roman=False, x2en=True)
                en2x = generate_other_translation(item, args.lang, roman=False, x2en=False)
                output_fp.write(serialize(x2en, docid=item["docid"], type="balanced_native2en"))
                output_fp.write(serialize(en2x, docid=item["docid"], type="balanced_en2native"))
            elif args.balanced_roman2en:
                x2en = generate_other_translation(item, args.lang, roman=True, x2en=True)
                output_fp.write(serialize(x2en, docid=item["docid"], type="balanced_roman2en"))
            elif args.balanced_en2roman:
                en2x = generate_other_translation(item, args.lang, roman=True, x2en=False)
                output_fp.write(serialize(en2x, docid=item["docid"], type="balanced_en2roman"))
            elif args.balanced_native or args.balanced_roman:
                text = generate_pure_text(item, args.lang, roman=args.balanced_roman)
                type = 'balanced_native' if args.balanced_native else 'balanced_romain'
                output_fp.write(serialize(text, docid=item["docid"], type=type))
            elif args.balanced_native2roman:
                native2roman = generate_other_romanization(item, args.lang, roman2native=False)
                output_fp.write(serialize(native2roman, docid=item["docid"], type="balanced_native2roman"))
            elif args.balanced_roman2native:
                roman2native = generate_other_romanization(item, args.lang, roman2native=True)
                output_fp.write(serialize(roman2native, docid=item["docid"], type="balanced_roman2native"))
            else:
                raise ValueError
        output_fp.close()
        sys.exit(0)
    
            

    if args.science_trans or args.science_roman or \
        args.science_en2roman or args.science_roman2en or args.science_native or \
        args.science_native2roman or args.science_roman2native:
        
        # if args.lang != 'ja':
        #     raise NotImplementedError(f"science_x is only supported for Japanese, but got {args.lang}")
        
        full_fn = os.path.join(DATA_ROOT, "datasets", "scientific_bilingual", f"en-{args.lang}", "full.jsonl")
        for item in tqdm(load_jsonl_iteratively(full_fn), desc="Generating science_x instructions"):
            if args.science_trans:
                x2en = generate_other_translation(item, args.lang, roman=False, x2en=True)
                en2x = generate_other_translation(item, args.lang, roman=False, x2en=False)
                output_fp.write(serialize(x2en, docid=item["docid"], type="science_trans"))
                output_fp.write(serialize(en2x, docid=item["docid"], type="science_en2native"))
            elif args.science_en2roman:
                en2x = generate_other_translation(item, args.lang, roman=True, x2en=False)
                output_fp.write(serialize(en2x, docid=item["docid"], type="science_en2roman"))
            elif args.science_roman2en:
                x2en = generate_other_translation(item, args.lang, roman=True, x2en=True)
                output_fp.write(serialize(x2en, docid=item["docid"], type="science_roman2en"))
            elif args.science_native or args.science_roman:
                text = generate_pure_text(item, args.lang, roman=args.science_roman)
                type = 'science_native' if args.science_native else 'science_romain'
                output_fp.write(serialize(text, docid=item["docid"], type=type))
            elif args.science_native2roman:
                native2roman = generate_other_romanization(item, args.lang, roman2native=False)
                output_fp.write(serialize(native2roman, docid=item["docid"], type="science_native2roman"))
            elif args.science_roman2native:
                roman2native = generate_other_romanization(item, args.lang, roman2native=True)
                output_fp.write(serialize(roman2native, docid=item["docid"], type="science_roman2native"))
            else:
                raise ValueError
                
        output_fp.close()
        sys.exit(0)
        
    # The below generation logic is only related to medical instructions besides those requires English translation
    input_type, output_type = decide_text_type(
        native=args.medical_native,
        roman=args.medical_roman, 
        halfroman=args.medical_halfroman,
        halfroman_reverse=args.medical_halfroman_reverse, 
        noisy=args.noisy_instruction)

    for item in tqdm(load_jsonl_iteratively(full_items_fn)):
        docid = item['docid']
        random.seed(docid)

        # `en` data is not further processed, need to be structured as items here
        if args.lang == "en":
            _item = {"docid": docid, "raw": item}
            item = _item

        lang_keyword = "en" if args.lang == "en_jstage" else args.lang
        item = process_keywords(item)
        summ = generate_summarization(item, lang_keyword, input_type, output_type)
        output_fp.write(serialize(summ, docid=docid, type="summarization"))
        
        if len(item["raw"]['sentences']) >= 2:
            complete = generate_textcomplete(item, lang_keyword, input_type, output_type)
            if complete is not None:
                output_fp.write(serialize(complete, docid=docid, type="text-completion"))

        # if 'conclu' in item['raw']:
        #     conclu = generate_conclusion(item, lang_keyword, input_type, output_type)
        #     output_fp.write(serialize(conclu, docid=docid, type="conclusion"))
        
        if 'qa' in item['raw']:
            qas = generate_qa(item, lang_keyword, input_type, output_type, with_context=False)
            for qa in qas:
                output_fp.write(serialize(qa, docid=docid, type="qa"))
        
        if len(item['raw']['keywords']) >=3:
            text2key = generate_text2key(item, lang_keyword, input_type, output_type)
            output_fp.write(serialize(text2key, docid=docid, type="text2key"))
        
            key2text = generate_key2text(item, lang_keyword, input_type, output_type)
            output_fp.write(serialize(key2text, docid=docid, type="key2text"))
            
        if 'causal' in item['raw']:
            for instruction in generate_causal(item, lang_keyword, input_type, output_type):
                output_fp.write(serialize(instruction, docid=docid, type="causal"))
            
        if 'gmrc' in item['raw']:
            for instruction in generate_gmrc(item, lang_keyword, input_type, output_type):
                output_fp.write(serialize(instruction, docid=docid, type="gmrc"))
    
    output_fp.close()