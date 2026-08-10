#!/usr/bin/env python3
"""Preference-tune production Qwen VoiceInk using approved cleanup pairs.

The policy starts from the merged production V1 model. A fresh LoRA adapter is
optimized with DPO while the same model with adapters disabled acts as the
reference policy. Use --check-only for a data/config preflight; it never loads
the model or starts training.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = ROOT / "datasets" / "qwen35-2b-voiceink-dpo" / "train.jsonl"
DEFAULT_EVAL = ROOT / "datasets" / "qwen35-2b-voiceink-dpo" / "eval.jsonl"
DEFAULT_BASE_MODEL = ROOT / "models" / "Qwen3.5-2B-voiceink"
DEFAULT_LORA_DIR = ROOT / "training" / "qwen35-2b-voiceink-dpo-v1" / "lora"
DEFAULT_OUTPUT_DIR = ROOT / "training" / "qwen35-2b-voiceink-dpo-v1" / "outputs"
DEFAULT_GGUF_BASE = ROOT / "models" / "Qwen3.5-2B-VoiceInk-dpo-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO-tune Qwen VoiceInk")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--max-seq-length", type=int, default=16384)
    parser.add_argument("--max-prompt-length", type=int, default=12288)
    parser.add_argument("--max-completion-length", type=int, default=4096)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")

    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--beta", type=float, default=0.10)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--resume-from-checkpoint", type=Path)

    parser.add_argument("--lora-dir", type=Path, default=DEFAULT_LORA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gguf-base", type=Path, default=DEFAULT_GGUF_BASE)
    parser.add_argument("--export-gguf", nargs="*", default=None)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def has_text(content) -> bool:
    return (
        isinstance(content, list)
        and any(
            isinstance(block, dict)
            and block.get("type") == "text"
            and str(block.get("text", "")).strip()
            for block in content
        )
    )


def validate_preferences(records: list[dict], path: Path) -> None:
    for index, record in enumerate(records, 1):
        required = {"request_id", "prompt", "chosen", "rejected"}
        if not required <= set(record):
            raise ValueError(f"{path}:{index} is missing preference fields")
        if [message.get("role") for message in record["prompt"]] != ["system", "user"]:
            raise ValueError(f"{path}:{index} has invalid prompt roles")
        if len(record["chosen"]) != 1 or record["chosen"][0].get("role") != "assistant":
            raise ValueError(f"{path}:{index} has invalid chosen completion")
        if len(record["rejected"]) != 1 or record["rejected"][0].get("role") != "assistant":
            raise ValueError(f"{path}:{index} has invalid rejected completion")
        messages = record["prompt"] + record["chosen"] + record["rejected"]
        if not all(has_text(message.get("content")) for message in messages):
            raise ValueError(f"{path}:{index} requires non-empty typed text blocks")
        if record["chosen"][0]["content"] == record["rejected"][0]["content"]:
            raise ValueError(f"{path}:{index} has identical chosen/rejected text")


def print_preflight(args: argparse.Namespace, train: list[dict],
                    evaluation: list[dict]) -> None:
    precision = "4-bit" if args.load_in_4bit else (
        "8-bit" if args.load_in_8bit else "BF16"
    )
    effective_batch = args.batch_size * args.grad_accum
    quantizations = args.export_gguf if args.export_gguf is not None else []
    print("Qwen VoiceInk DPO preflight")
    print(f"  Base/reference policy: {args.base_model}")
    print(f"  Train: {args.train} ({len(train)}, sha256={sha256_file(args.train)})")
    print(f"  Eval: {args.eval} ({len(evaluation)}, sha256={sha256_file(args.eval)})")
    print(
        f"  Lengths: total={args.max_seq_length}, prompt={args.max_prompt_length}, "
        f"completion={args.max_completion_length}"
    )
    print(f"  Fresh LoRA: r={args.r}, alpha={args.lora_alpha}")
    print(f"  Effective batch size: {effective_batch}")
    print(f"  Epochs: {args.epochs}; max steps: {args.max_steps}")
    print(
        f"  DPO: beta={args.beta}, lr={args.lr}, "
        f"warmup ratio={args.warmup_ratio}"
    )
    print(
        f"  Eval/checkpoint every {args.eval_steps}/{args.save_steps} steps; "
        "best eval-loss checkpoint exported"
    )
    print(f"  Load precision: {precision}")
    print(f"  Adapter output: {args.lora_dir}")
    print(f"  Checkpoints: {args.output_dir}")
    if args.export_gguf is None:
        print("  GGUF export: disabled")
    else:
        methods = quantizations or ["q4_k_m"]
        print(f"  GGUF export: {args.gguf_base}_gguf ({', '.join(methods)})")


def main() -> None:
    args = parse_args()
    required = [args.train, args.eval, args.base_model / "config.json"]
    missing = [path for path in required if not path.is_file()]
    if missing:
        print(f"Missing required files: {missing}", file=sys.stderr)
        sys.exit(1)
    if args.load_in_4bit and args.load_in_8bit:
        print("Use only one of --load-in-4bit or --load-in-8bit", file=sys.stderr)
        sys.exit(2)
    if args.max_prompt_length + args.max_completion_length > args.max_seq_length:
        print("Prompt + completion limits exceed --max-seq-length", file=sys.stderr)
        sys.exit(2)
    if args.save_steps % args.eval_steps != 0:
        print("--save-steps must be a multiple of --eval-steps", file=sys.stderr)
        sys.exit(2)

    train_records = load_jsonl(args.train)
    eval_records = load_jsonl(args.eval)
    validate_preferences(train_records, args.train)
    validate_preferences(eval_records, args.eval)
    train_ids = {record["request_id"] for record in train_records}
    eval_ids = {record["request_id"] for record in eval_records}
    if train_ids & eval_ids:
        raise ValueError("Preference train/eval overlap")

    print_preflight(args, train_records, eval_records)
    if args.check_only:
        print("Preflight complete; no model was loaded and no training was run.")
        return

    from datasets import Dataset
    from unsloth import FastVisionModel
    import torch
    from trl import DPOConfig, DPOTrainer

    print(f"Loading merged production policy: {args.base_model}")
    model, processor = FastVisionModel.from_pretrained(
        str(args.base_model),
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
    )
    FastVisionModel.for_training(model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        processing_class=processor,
        train_dataset=Dataset.from_list(train_records),
        eval_dataset=Dataset.from_list(eval_records),
        args=DPOConfig(
            output_dir=str(args.output_dir),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps if args.max_steps > 0 else -1,
            learning_rate=args.lr,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            weight_decay=0.01,
            beta=args.beta,
            loss_type="sigmoid",
            use_logits_to_keep=True,
            max_length=args.max_seq_length,
            max_prompt_length=args.max_prompt_length,
            max_completion_length=args.max_completion_length,
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            per_device_eval_batch_size=1,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            seed=3407,
            report_to="none",
        ),
    )

    gpu = torch.cuda.get_device_properties(0)
    total_gb = gpu.total_memory / 1024**3
    print(f"GPU: {gpu.name} — {total_gb:.1f} GB")
    print("Starting DPO training...")
    stats = trainer.train(
        resume_from_checkpoint=(
            str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
        )
    )
    print(f"Training complete in {stats.metrics['train_runtime'] / 60:.1f} minutes")
    print(f"Best eval loss: {trainer.state.best_metric}")
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")

    args.lora_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving DPO LoRA adapter to {args.lora_dir}")
    model.save_pretrained(str(args.lora_dir))
    processor.save_pretrained(str(args.lora_dir))

    if args.export_gguf is not None:
        methods = args.export_gguf or ["q4_k_m"]
        print(f"Exporting GGUF ({', '.join(methods)}) -> {args.gguf_base}_gguf")
        model.save_pretrained_gguf(
            str(args.gguf_base),
            processor,
            quantization_method=methods,
        )
        print(f"GGUF export complete: {args.gguf_base}_gguf")


if __name__ == "__main__":
    main()
