from __future__ import annotations

from typing import Dict, List

from datasets import load_dataset

DATASET_NAMES = {
    "symbol_value",
    "object_color",
    "realistic_conversation_philosophy",
    "realistic_symbol_value",
}

CHAT_TEMPLATES = {
    "qwen",
    "llama",
    "deepseek",
    "qwq",
    "skywork",
    "qwen_base",
    "llama_base",
}


def load_userassist_dataset(
    dataset_name: str,
    split: str = "test",
    repo_id: str = "UserAssist/UserAssist",
) -> List[Dict]:
    if dataset_name not in DATASET_NAMES:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")
    dataset = load_dataset(repo_id, dataset_name, split=split)
    return dataset.to_list()


def _normalize_target(prefix: str, target: str) -> str:
    target = str(target)
    if not target.startswith(" ") and not prefix.endswith(" "):
        return " " + target
    return target


def _assistant_continuation(prefix: str, target: str, chat_template: str) -> str:
    target = _normalize_target(prefix, target)

    if chat_template in {"qwen", "qwq", "qwen_base"}:
        return f"<|im_start|>assistant\n{prefix}{target}"
    if chat_template in {"llama", "llama_base"}:
        return f"<|start_header_id|>assistant<|end_header_id|>\n\n{prefix}{target}"
    if chat_template in {"deepseek", "skywork"}:
        return f"<｜Assistant｜><think>\n\n</think>{prefix}{target}"
    raise ValueError(f"Unsupported chat_template: {chat_template}")


def build_generation_examples(
    tokenizer,
    dataset_name: str,
    chat_template: str,
    split: str = "test",
) -> List[Dict]:
    if chat_template not in CHAT_TEMPLATES:
        raise ValueError(f"Unsupported chat_template: {chat_template}")

    rows = load_userassist_dataset(dataset_name=dataset_name, split=split)
    examples = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            row["message"],
            add_generation_prompt=True,
            tokenize=False,
        )
        if chat_template in {"qwen_base", "llama_base"}:
            prompt = prompt + row["prefix"]

        item = dict(row)
        item["total_prompt"] = prompt
        examples.append(item)
    return examples


def build_logprob_examples(
    tokenizer,
    dataset_name: str,
    chat_template: str,
    split: str = "test",
) -> List[Dict]:
    if chat_template not in CHAT_TEMPLATES:
        raise ValueError(f"Unsupported chat_template: {chat_template}")

    rows = load_userassist_dataset(dataset_name=dataset_name, split=split)
    examples = []

    for row in rows:
        prompt = tokenizer.apply_chat_template(row["message"], tokenize=False)
        prefix = row["prefix"]

        for target_type, raw_target in (
            ("user", row["user_value"]),
            ("assistant", row["assistant_value"]),
        ):
            target = _normalize_target(prefix, str(raw_target))
            full_prompt = prompt + _assistant_continuation(prefix, target, chat_template)
            item = dict(row)
            item["target_type"] = target_type
            item["log_prob_target"] = target
            item["total_prompt"] = full_prompt
            examples.append(item)

    return examples
