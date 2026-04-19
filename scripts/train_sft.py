from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def _infer_template(model_name: str) -> str:
    model_name = model_name.lower()
    if "qwen" in model_name:
        return "qwen"
    if "llama" in model_name:
        return "llama3"
    raise ValueError("Could not infer a LLaMA-Factory template. Please pass --template explicitly.")


def _load_first_record(dataset_path: Path) -> dict:
    text = dataset_path.read_text(encoding="utf-8").strip()
    if dataset_path.suffix == ".jsonl":
        return json.loads(text.splitlines()[0])
    data = json.loads(text)
    if not isinstance(data, list) or not data:
        raise ValueError("Expected a non-empty JSON array or JSONL file.")
    return data[0]


def _build_dataset_info(record: dict, dataset_filename: str) -> dict:
    if "messages" in record:
        return {
            "file_name": dataset_filename,
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
            },
        }

    if "output" in record:
        return {
            "file_name": dataset_filename,
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        }

    raise ValueError("Unsupported SFT dataset format. Expected either `messages` or `instruction`/`input`/`output`.")


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UserAssist SFT via LLaMA-Factory.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-path", required=True, help="Local JSON or JSONL file.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--template", default=None, help="LLaMA-Factory template name, e.g. llama3 or qwen.")
    parser.add_argument("--dataset-name", default="userassist_sft")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--cutoff-len", type=int, default=2048)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--run", action="store_true", help="Launch llamafactory-cli train after writing config.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    run_dir = output_dir / "_llamafactory"
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    copied_dataset_path = data_dir / dataset_path.name
    shutil.copy2(dataset_path, copied_dataset_path)

    first_record = _load_first_record(dataset_path)
    dataset_info = {args.dataset_name: _build_dataset_info(first_record, copied_dataset_path.name)}
    (data_dir / "dataset_info.json").write_text(json.dumps(dataset_info, indent=2) + "\n", encoding="utf-8")

    template = args.template or _infer_template(args.model)
    yaml_path = run_dir / "train_sft.yaml"
    yaml_text = f"""
### model
model_name_or_path: {args.model}
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: {args.lora_rank}
lora_target: all

### dataset
dataset_dir: {data_dir}
dataset: {args.dataset_name}
template: {template}
cutoff_len: {args.cutoff_len}
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: {output_dir}
logging_steps: {args.logging_steps}
save_strategy: epoch
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none

### train
per_device_train_batch_size: {args.per_device_train_batch_size}
gradient_accumulation_steps: {args.gradient_accumulation_steps}
learning_rate: {args.learning_rate}
num_train_epochs: {args.num_train_epochs}
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null
"""
    _write_yaml(yaml_path, yaml_text)

    print(f"Wrote dataset registry: {data_dir / 'dataset_info.json'}")
    print(f"Wrote training config: {yaml_path}")

    if args.run:
        if shutil.which("llamafactory-cli") is None:
            raise SystemExit("`llamafactory-cli` was not found in PATH. Install LLaMA-Factory in this environment first.")
        subprocess.run(["llamafactory-cli", "train", str(yaml_path)], check=True)
    else:
        print("Dry run only. Re-run with --run to start training.")


if __name__ == "__main__":
    main()
