# MLDA-EVAL

This project focuses on **medical domain adaptation and evaluation** for cross-lingual medical knowledge understanding between English and Japanese. 
The codebase consists of two main components: **data generation** and **training settings**.

## Data Generation

The data generation component creates both training and evaluation datasets for medical domain language models<cite />.

### Training Data Generation

The training data pipeline (`datasets/generate_train/`) creates instruction datasets from medical abstracts through multiple stages, including regex-based instruction creation, LLM-based QA generation, romanization, and instruction formatting.

### Evaluation Data Generation

The evaluation component (`datasets/generate_eval/`) creates benchmarks for assessing cross-lingual medical knowledge understanding and distractor quality in multiple-choice questions, based on the proposal of AdaXEval.

## Training Settings

The training settings component manages model training configurations and execution. 

### Configuration System

Training configurations are defined in YAML files that specify dataset collections, model parameters, parallelism settings, and hyperparameters [4](#1-3) . 

### Training Pipeline

The training pipeline consists of three stages executed via shell scripts [6](#1-5) :
1. Dataset generation from configuration
2. Tokenization 
3. Distributed training using Megatron-LM

The actual training is performed using Megatron-LM with distributed optimization across multiple GPUs.

## Notes

The project uses a flexible configuration system where experiments can inherit from a default configuration and override specific parameters. 
Training supports various dataset combinations including monolingual medical corpora, cross-lingual transfer tasks, and instruction tuning datasets.
