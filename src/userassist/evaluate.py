from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from peft import PeftModel
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from userassist.data import CHAT_TEMPLATES, build_generation_examples, build_logprob_examples


def _batched(items: List[Dict], batch_size: int) -> Iterable[List[Dict]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model(model_name: str, adapter_path: Optional[str] = None):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    return model


def run_generation(
    model,
    tokenizer,
    examples: List[Dict],
    batch_size: int,
    max_new_tokens: int,
) -> List[Dict]:
    results: List[Dict] = []
    device = model.device

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }

    for batch in tqdm(_batched(examples, batch_size), total=(len(examples) + batch_size - 1) // batch_size, desc="generation"):
        prompts = [item["total_prompt"] for item in batch]
        inputs = tokenizer(
            prompts,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            add_special_tokens=False,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        padded_input_len = inputs["input_ids"].shape[1]
        for item, output_ids in zip(batch, outputs):
            generated_ids = output_ids[padded_input_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            row = dict(item)
            row["generated_text"] = generated_text
            results.append(row)

    return results


def run_logprob(
    model,
    tokenizer,
    examples: List[Dict],
    batch_size: int,
) -> List[Dict]:
    results: List[Dict] = []
    device = model.device

    for batch in tqdm(_batched(examples, batch_size), total=(len(examples) + batch_size - 1) // batch_size, desc="logprob"):
        prompts = [item["total_prompt"] for item in batch]
        targets = [item["log_prob_target"] for item in batch]

        prompt_inputs = tokenizer(
            prompts,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            add_special_tokens=False,
        ).to(device)
        target_inputs = tokenizer(
            targets,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            add_special_tokens=False,
        ).to(device)

        with torch.no_grad():
            outputs = model(**prompt_inputs)

        log_probs = F.log_softmax(outputs.logits, dim=-1)

        for batch_index, (item, input_ids, target_ids) in enumerate(
            zip(batch, prompt_inputs["input_ids"], target_inputs["input_ids"])
        ):
            valid_target_ids = target_ids[target_ids != tokenizer.pad_token_id]
            target_len = len(valid_target_ids)
            sequence_target_ids = input_ids[-target_len:]
            token_log_probs = log_probs[batch_index, -target_len - 1 : -1, :].gather(
                1, sequence_target_ids.unsqueeze(-1)
            ).squeeze(-1)

            row = dict(item)
            row["avg_log_prob"] = token_log_probs.mean().item()
            results.append(row)

    return results


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate models on UserAssist.")
    parser.add_argument("--model", required=True, help="Base model name or local path.")
    parser.add_argument("--adapter-path", default=None, help="Optional LoRA adapter path.")
    parser.add_argument("--dataset", required=True, help="UserAssist subset name.")
    parser.add_argument("--chat-template", default="llama", choices=sorted(CHAT_TEMPLATES))
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size-generation", type=int, default=16)
    parser.add_argument("--batch-size-logprob", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-logprob", action="store_true")
    args = parser.parse_args()

    tokenizer = _load_tokenizer(args.model)
    model = _load_model(args.model, adapter_path=args.adapter_path)

    output_dir = Path(args.output_dir)
    model_tag = Path(args.adapter_path).name if args.adapter_path else args.model.split("/")[-1]
    prefix = f"{args.dataset}_{model_tag}"

    if not args.skip_generation:
        generation_examples = build_generation_examples(
            tokenizer=tokenizer,
            dataset_name=args.dataset,
            chat_template=args.chat_template,
            split=args.split,
        )
        generation_rows = run_generation(
            model=model,
            tokenizer=tokenizer,
            examples=generation_examples,
            batch_size=args.batch_size_generation,
            max_new_tokens=args.max_new_tokens,
        )
        _write_jsonl(output_dir / f"{prefix}_generation.jsonl", generation_rows)

    if not args.skip_logprob:
        logprob_examples = build_logprob_examples(
            tokenizer=tokenizer,
            dataset_name=args.dataset,
            chat_template=args.chat_template,
            split=args.split,
        )
        logprob_rows = run_logprob(
            model=model,
            tokenizer=tokenizer,
            examples=logprob_examples,
            batch_size=args.batch_size_logprob,
        )
        _write_jsonl(output_dir / f"{prefix}_logprob.jsonl", logprob_rows)


if __name__ == "__main__":
    main()
