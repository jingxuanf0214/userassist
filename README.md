# User-Assistant Bias in LLMs

Paper accepted in ACL 2026 findings.

[![Datasets](https://img.shields.io/badge/Datasets%20%26%20Models-HuggingFace-F59E0B?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/UserAssist/UserAssist)

[![arXiv](https://img.shields.io/badge/arXiv-2508.15815-b31b1b.svg)](https://arxiv.org/abs/2508.15815)

- load the UserAssist benchmark from Hugging Face
- evaluate generation and target log-probability
- run LoRA SFT with LLaMA-Factory
- run LoRA DPO with LLaMA-Factory


## Install

```bash
pip install -r requirements.txt
pip install -e .
```

Training also requires `llamafactory-cli` to be installed in the environment.

## Repository Layout

```text
src/userassist/data.py         Dataset loading and prompt construction
src/userassist/evaluate.py     Evaluation CLI for generation + log-prob
scripts/train_sft.py           LLaMA-Factory SFT wrapper
scripts/train_dpo.py           LLaMA-Factory DPO wrapper
```

## Evaluation

Evaluate a base model on one UserAssist subset:

```bash
python -m userassist.evaluate \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset symbol_value \
  --chat-template llama \
  --output-dir outputs/eval
```

Evaluate a LoRA adapter on top of a base model:

```bash
python -m userassist.evaluate \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --adapter-path path/to/adapter \
  --dataset object_color \
  --chat-template llama \
  --output-dir outputs/eval
```

Outputs are written as JSONL files for generation results and log-probability results.

## SFT

The SFT script follows the original LLaMA-Factory-based workflow. It writes a minimal `dataset_info.json` plus a training YAML under `OUTPUT_DIR/_llamafactory/`, then optionally launches `llamafactory-cli train`.

The input dataset can be either:

- `messages`: a chat-style conversation ending with the assistant response
- or `instruction`, `input`, `output`

Example:

```bash
python scripts/train_sft.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-path examples/sft_example.json \
  --output-dir outputs/sft \
  --run
```

## DPO

The DPO script also uses the original LLaMA-Factory setup pattern and generates a local dataset registry plus YAML config before training.

The input dataset must contain:

- `messages`: prompt turns before the chosen/rejected answer
- `chosen`
- `rejected`

Example:

```bash
python scripts/train_dpo.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-path examples/dpo_example.json \
  --output-dir outputs/dpo \
  --run
```

## UserAssist Dataset

The evaluation dataloader pulls the public benchmark from:

- `UserAssist/UserAssist` on Hugging Face

Supported subsets in this repo:

- `symbol_value`
- `object_color`
- `realistic_conversation_philosophy`
- `realistic_symbol_value`
