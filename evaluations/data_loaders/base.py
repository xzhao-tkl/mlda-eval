from pdb import set_trace
import random
import datasets
import os
import ujson as json
from typing import List, Dict
from datasets import load_dataset

from tasks.mcqa import MCQASample
from tasks.ner import NERSample
from tasks.mt import MTSample
from tasks.nli import NLISample
from tool_utils import main_print

try:
    from config_file import DATA_ROOT_DIR
except:
    print("DATA_ROOT_DIR is not defined, so loading local data is not supported.")


BENCHMARK_NAME = "Coldog2333/JMedBench"


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

def load_mcqa_samples(dataset_name: str) -> Dict[str, List[MCQASample]]:

    mcqa_samples = {"train": [], "test": [], "validation": []}

    # try to load from huggingface firstly
    try:
        main_print(f"Try loading {dataset_name} dataset from Hugging Face...")
        if dataset_name.startswith("nii"):
            nii_evals_root = "/home/xzhao/workspace/med-eval/dataset/nii-evals"
            if dataset_name.startswith("nii_en5_mono_prompt"):
                filename = nii_evals_root + "/en5-mono-prompts.jsonl"
            elif dataset_name.startswith("nii_en5_bi_prompt"):
                filename = nii_evals_root + "/en5-bi-prompts.jsonl"
            elif dataset_name.startswith("nii_en5_tri_prompt"):
                filename = nii_evals_root + "/en5-tri-prompts.jsonl"
            elif dataset_name.startswith("nii_ja5_mono_prompt"):
                filename = nii_evals_root + "/ja5-mono-prompts.jsonl"
            elif dataset_name.startswith("nii_ja5_bi_prompt"):
                filename = nii_evals_root + "/ja5-bi-prompts.jsonl"
            elif dataset_name.startswith("nii_ja5_tri_prompt"):
                filename = nii_evals_root + "/ja5-tri-prompts.jsonl"
            else:
                raise ValueError(f"Unknown dataset name: {dataset_name}")
            
            if dataset_name.endswith(".cross"):
                qlang = dataset_name.split(".")[0].split("-")[-1]
                olang = 'en' if qlang == 'ja' else 'ja'
            else:
                qlang = dataset_name.split("-")[-1]
                olang = qlang
            
            items = load_jsonl_iteratively(filename)
            for item in items:
                sample_id = item['en']["docid"]
                answer_idx=item['en']["answer"]
                en_ques = item['en']["question"]
                en_opts = item['en']["options"]
                ja_ques = item['ja']['ja']["question"]
                ja_opts = item['ja']['ja']["options"]

                question = en_ques if qlang == "en" else ja_ques
                options = en_opts if olang == 'en' else ja_opts
                
                mcqa_samples["test"].append(
                    MCQASample(
                        sample_id=sample_id,
                        question=question,
                        options=options,
                        answer_idx=answer_idx,
                        n_options=4,
                        metadata=None,
                        dataset=dataset_name
                    )
                )
        elif dataset_name.startswith("adaxeval"):
            adaxeval_root = "/data/xzhao/experiments/med-eval/dataset/adaxeval"
            dataset_name = dataset_name.split("-")[-1]
            assert dataset_name.startswith("en_") or dataset_name.startswith("ja_"), \
                "Dataset name must start with 'en_' or 'ja_' for adaxeval datasets."
            assert dataset_name.endswith("knowledge_memorization") or \
                   dataset_name.endswith("knowledge_generalization") or \
                   dataset_name.endswith("knowledge_reasoning"), \
                "Dataset name must end with 'knowledge_memorization', 'knowledge_generalization', or 'knowledge_reasoning' for adaxeval datasets."
            
            dataset_file = os.path.join(adaxeval_root, dataset_name + ".jsonl")
            print(f"Loading dataset from {dataset_file}")
            assert os.path.exists(dataset_file), "File not found: " + os.path.join(adaxeval_root, dataset_name + ".jsonl")
            items = load_jsonl_iteratively(dataset_file)
            for item in items:
                sample_id = item['id']
                question = item['question']
                options = item['options']
                assert len(options) == 4, "Options must have exactly 4 items."
                answer_idx = item['answer_idx']
                n_options = len(options)
                metadata = item.get('metadata', None)
                try:
                    sample = MCQASample(
                        sample_id=sample_id,
                        question=question,
                        options=options,
                        answer_idx=answer_idx,
                        n_options=n_options,
                        metadata=metadata,
                        dataset=dataset_name
                    )
                except Exception as e:
                    # print(f"Error loading sample {sample_id} from {dataset_name}: {e}")
                    continue
                mcqa_samples["test"].append(sample)
        else:
            dataset = load_dataset(BENCHMARK_NAME, dataset_name)
            for split in ["train", "test", "validation"]:
                if split in dataset.keys():
                    for sample in dataset[split]:
                        mcqa_samples[split].append(
                            MCQASample(
                                sample_id=sample["sample_id"],
                                question=sample["question"],
                                options=sample["options"],
                                answer_idx=sample["answer_idx"],
                                n_options=sample["n_options"],
                                metadata=sample["metadata"],
                                dataset=sample["dataset"]
                            )
                        )
    except Exception as e:
        print(e)

        main_print(f"Loading {dataset_name} dataset from local file...")

        for split in ["train", "test", "validation"]:
            data_filename = os.path.join(DATA_ROOT_DIR, f"{dataset_name}/{split}.jsonl")
            if os.path.exists(data_filename):

                with open(data_filename, "r", encoding='utf-8') as f:
                    for line in f:
                        sample = json.loads(line)
                        mcqa_samples[split].append(
                            MCQASample(
                                sample_id=sample["sample_id"],
                                question=sample["question"],
                                options=sample["options"],
                                answer_idx=sample["answer_idx"],
                                n_options=sample["n_options"],
                                metadata=sample["metadata"],
                                dataset=sample["dataset"]
                            )
                        )

            else:
                main_print(f"File {data_filename} does not exist.")

    main_print(f"Loaded {len(mcqa_samples['test'])} samples for testing set.")
    main_print(f"Loaded {len(mcqa_samples['train'])} samples for training set.")
    main_print(f"Loaded {len(mcqa_samples['validation'])} samples for validation set.")

    return mcqa_samples


def load_nli_samples(dataset_name: str) -> Dict[str, List[NLISample]]:

    nli_samples = {"train": [], "test": []}

    # try to load from huggingface firstly
    try:
        main_print(f"Try loading {dataset_name} dataset from Hugging Face...")
        dataset = load_dataset(BENCHMARK_NAME, dataset_name)
        for split in ["train", "test"]:
            if split in dataset.keys():
                for sample in dataset[split]:
                    nli_samples[split].append(
                        NLISample(
                            sample_id=sample["sample_id"],
                            premise=sample["premise"],
                            hypothesis=sample["hypothesis"],
                            label=sample["label"],
                            n_label=sample["n_label"],
                            metadata=sample["metadata"]
                        )
                    )

    except:

        main_print(f"Loading {dataset_name} dataset from local file...")

        for split in ["train", "test"]:
            data_filename = os.path.join(DATA_ROOT_DIR, f"{dataset_name}/{split}.jsonl")
            if os.path.exists(data_filename):
                with open(data_filename, "r", encoding='utf-8') as f:
                    for line in f:
                        sample = json.loads(line)
                        nli_samples[split].append(
                            NLISample(
                                sample_id=sample["sample_id"],
                                premise=sample["premise"],
                                hypothesis=sample["hypothesis"],
                                label=sample["label"],
                                n_label=sample["n_label"],
                                metadata=sample["metadata"]
                            )
                        )

    main_print(f"Loaded {len(nli_samples['test'])} samples for testing set.")
    main_print(f"Loaded {len(nli_samples['train'])} samples for training set.")

    return nli_samples


def load_ner(dataset_name: str) -> Dict[str, List[NERSample]]:
    ner_samples = {"train": [], "test": [], "validation": []}

    # try to load from huggingface firstly
    try:
        main_print(f"Try loading {dataset_name} dataset from Hugging Face...")
        dataset = load_dataset(BENCHMARK_NAME, dataset_name)
        for split in ["train", "validation", "test"]:
            if split in dataset.keys():
                for sample in dataset[split]:
                    ner_samples[split].append(
                        NERSample(
                            text=sample["text"],
                            labels=sample["labels"],
                            entity_type=sample["entity_type"]
                        )
                    )

    except Exception as e:
        print(e)

        main_print(f"Loading {dataset_name} dataset from local file...")

        for split in ["train", "test"]:
            data_filename = os.path.join(DATA_ROOT_DIR, f"{dataset_name}/{split}.jsonl")
            if os.path.exists(data_filename):
                with open(data_filename, "r", encoding='utf-8') as f:
                    for line in f:
                        sample = json.loads(line)
                        ner_samples[split].append(
                            NERSample(
                                text=sample["text"],
                                labels=sample["labels"],
                                entity_type=sample["entity_type"]
                            )
                        )

    main_print(f"Loaded {len(ner_samples['test'])} samples for testing set.")
    main_print(f"Loaded {len(ner_samples['train'])} samples for training set.")
    main_print(f"Loaded {len(ner_samples['validation'])} samples for validation set.")

    return ner_samples


def load_blurb(subset_name):
    """
    :param subset_name: Options: "bc2gm", "bc5chem", "bc5disease", "jnlpba", "ncbi_disease"
    :return:
    """
    dataset = load_dataset('bigbio/blurb', subset_name)
    ner_samples = {"train": [], "test": [], "validation": []}
    for split in dataset.keys():
        for sample in dataset[split]:
            text = " ".join(sample["tokens"])

            # collect labels
            labels = []
            cache = []
            assert len(sample["ner_tags"]) == len(sample["tokens"])
            for tag, token in zip(sample["ner_tags"], sample["tokens"]):
                if tag == 0:
                    if cache:
                        labels.append(" ".join(cache))
                        cache = []
                elif tag == 1:
                    if cache:
                        labels.append(" ".join(cache))
                        cache = []
                    cache.append(token)
                else:
                    assert len(cache) > 0
                    cache.append(token)

            if cache:
                labels.append(" ".join(cache))
                cache = []

            if len(labels) == 0:
                labels = ["none"]

            ner_samples[split].append(
                NERSample(
                    text=text,
                    labels=labels,
                    entity_type=sample["type"]
                )
            )

    main_print(f"Loaded {len(ner_samples['test'])} samples for testing set.")
    main_print(f"Loaded {len(ner_samples['train'])} samples for training set.")
    main_print(f"Loaded {len(ner_samples['validation'])} samples for validation set.")

    return ner_samples




def load_ejmmt(
    dataset_name="ejmmt",
    source_lang="english",
    target_lang="japanese",
) -> Dict[str, List[MTSample]]:

    mt_samples = {"train": [], "test": [], "validation": []}

    # try to load from huggingface firstly
    try:
        main_print(f"Try loading {dataset_name} dataset from Hugging Face...")
        dataset = load_dataset(BENCHMARK_NAME, dataset_name)
        for split in ["train", "test", "validation"]:
            if split in dataset.keys():
                for sample in dataset[split]:
                    mt_samples[split].append(
                        MTSample(
                            source_text=sample[source_lang],
                            target_text=sample[target_lang],
                            source_language=source_lang,
                            target_language=target_lang
                        )
                    )

    except:

        main_print(f"Loading {dataset_name} dataset from local file...")

        for split in ["train", "test", "validation"]:
            data_filename = os.path.join(DATA_ROOT_DIR, f"{dataset_name}/{split}.jsonl")
            if os.path.exists(data_filename):
                with open(data_filename, "r", encoding='utf-8') as f:
                    for line in f:
                        sample = json.loads(line)
                        mt_samples[split].append(
                            MTSample(
                                source_text=sample[source_lang],
                                target_text=sample[target_lang],
                                source_language=source_lang,
                                target_language=target_lang
                            )
                        )

            else:
                main_print(f"File {data_filename} does not exist.")

    main_print(f"Loaded {len(mt_samples['test'])} samples for testing set.")

    return mt_samples


VALID_DATASET_NAMES = ["medmcqa", "usmleqa", "medqa", "mmlu", "mmlu_medical"]

if __name__ == "__main__":
    for dataset_name in VALID_DATASET_NAMES:
        mcqa_samples = load_mcqa_samples(dataset_name)
        # print(mcqa_samples)
