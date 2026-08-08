#!/usr/bin/env python3
"""Fine-tune LFM2.5 1.2B for VoiceInk transcription cleanup with Unsloth.

The script intentionally uses separate datasets and output directories from the
Qwen training path. Run with --check-only to validate inputs without loading a
model or starting a training job.
"""
import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = ROOT / "datasets" / "lfm25" / "train.jsonl"
DEFAULT_EVAL = ROOT / "datasets" / "lfm25" / "eval.jsonl"
DEFAULT_LORA_DIR = ROOT / "training" / "lfm25" / "lora"
DEFAULT_OUTPUT_DIR = ROOT / "training" / "lfm25" / "outputs"
DEFAULT_GGUF_BASE = ROOT / "models" / "LFM2.5-1.2B-VoiceInk"
DEFAULT_MODEL = "LiquidAI/LFM2.5-1.2B-Instruct"
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "out_proj", "in_proj",
    "w1", "w2", "w3",
]


@dataclass(frozen=True)
class TrainingProfile:
    name: str
    description: str
    base_model: str
    lora_dir: Path
    output_dir: Path
    gguf_base: Path
    r: int
    lora_alpha: int
    epochs: int
    learning_rate: float


LFM25_12B_PROFILE = TrainingProfile(
    name="LFM2.5 1.2B Instruct",
    description="Fine-tune LFM2.5 1.2B for VoiceInk with Unsloth.",
    base_model=DEFAULT_MODEL,
    lora_dir=DEFAULT_LORA_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    gguf_base=DEFAULT_GGUF_BASE,
    r=16,
    lora_alpha=16,
    epochs=1,
    learning_rate=2e-4,
)

LFM25_26B_BASE_PROFILE = TrainingProfile(
    name="LFM2.5 2.6B Base",
    description="Fine-tune LFM2.5 2.6B Base for VoiceInk with Unsloth.",
    base_model="LiquidAI/LFM2.5-2.6B-Base",
    lora_dir=ROOT / "training" / "lfm25-2.6b-base" / "lora",
    output_dir=ROOT / "training" / "lfm25-2.6b-base" / "outputs",
    gguf_base=ROOT / "models" / "LFM2.5-2.6B-VoiceInk",
    r=32,
    lora_alpha=32,
    epochs=2,
    learning_rate=1e-4,
)


def parse_args(
    profile: TrainingProfile = LFM25_12B_PROFILE,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=profile.description
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--base-model", default=profile.base_model)
    parser.add_argument("--max-seq-length", type=int, default=16384)

    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--load-in-4bit", action="store_true")
    precision.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--offload-optimizer", action="store_true")

    parser.add_argument("--r", type=int, default=profile.r, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=profile.lora_alpha)
    parser.add_argument("--epochs", type=int, default=profile.epochs)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=profile.learning_rate)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=-1)

    parser.add_argument("--lora-dir", type=Path, default=profile.lora_dir)
    parser.add_argument("--output-dir", type=Path, default=profile.output_dir)
    parser.add_argument("--gguf-base", type=Path, default=profile.gguf_base)
    parser.add_argument(
        "--export-gguf", nargs="*", default=None,
        help="Export after training; methods such as q4_k_m q8_0",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Validate datasets and print configuration without loading or training a model",
    )
    parser.set_defaults(profile_name=profile.name)
    return parser.parse_args(argv)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_conversations(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc

            messages = record.get("messages")
            if not isinstance(messages, list) or len(messages) != 3:
                raise ValueError(f"{path}:{line_number}: expected exactly 3 messages")
            expected_roles = ("system", "user", "assistant")
            roles = tuple(message.get("role") for message in messages)
            if roles != expected_roles:
                raise ValueError(
                    f"{path}:{line_number}: expected roles {expected_roles}, got {roles}"
                )
            for message in messages:
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(
                        f"{path}:{line_number}: LFM2.5 requires non-empty string content"
                    )
            records.append(record)
    if not records:
        raise ValueError(f"{path}: dataset is empty")
    return records


def print_preflight(args: argparse.Namespace, train_data: list[dict],
                    eval_data: list[dict] | None) -> None:
    print("LFM2.5 training preflight")
    print(f"  Profile: {args.profile_name}")
    print(f"  Base model: {args.base_model}")
    print(f"  Train: {args.train} ({len(train_data)} samples, sha256={file_sha256(args.train)})")
    if eval_data is not None:
        print(f"  Eval: {args.eval} ({len(eval_data)} samples, sha256={file_sha256(args.eval)})")
    else:
        print("  Eval: disabled")
    print(f"  Max sequence length: {args.max_seq_length}")
    print(f"  LoRA: r={args.r}, alpha={args.lora_alpha}")
    print(f"  LoRA targets: {', '.join(LORA_TARGET_MODULES)}")
    print(f"  Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"  Epochs: {args.epochs}; max steps: {args.max_steps}")
    print(f"  Load precision: {'4-bit' if args.load_in_4bit else '8-bit' if args.load_in_8bit else 'BF16'}")
    print(f"  Adapter output: {args.lora_dir}")
    print(f"  Checkpoints: {args.output_dir}")
    if args.export_gguf is not None:
        methods = args.export_gguf or ["q4_k_m"]
        print(f"  GGUF export: {args.gguf_base}_gguf ({', '.join(methods)})")
    else:
        print("  GGUF export: disabled")


def render_conversations(records: list[dict], tokenizer) -> list[dict]:
    return [
        {
            "text": tokenizer.apply_chat_template(
                record["messages"], tokenize=False, add_generation_prompt=False
            )
        }
        for record in records
    ]


def backup_existing_ggufs(directory: Path) -> None:
    if not directory.exists():
        return
    for gguf_file in sorted(directory.glob("*.gguf")):
        if ".v" in gguf_file.stem or "BF16" in gguf_file.name.upper():
            continue
        next_version = 1
        for version_file in directory.glob(f"{gguf_file.stem}.v*.gguf"):
            try:
                version = int(version_file.stem.rsplit(".v", 1)[1])
            except (ValueError, IndexError):
                continue
            next_version = max(next_version, version + 1)
        backup = gguf_file.with_name(f"{gguf_file.stem}.v{next_version}.gguf")
        print(f"Backing up {gguf_file.name} -> {backup.name}")
        shutil.copy2(gguf_file, backup)


def main(profile: TrainingProfile = LFM25_12B_PROFILE) -> None:
    args = parse_args(profile)
    if not args.train.is_file():
        print(f"Training data not found: {args.train}", file=sys.stderr)
        sys.exit(1)

    train_conversations = load_conversations(args.train)
    eval_conversations = None
    if args.eval and args.eval.is_file():
        eval_conversations = load_conversations(args.eval)
    print_preflight(args, train_conversations, eval_conversations)
    if args.check_only:
        print("Preflight complete; no model was loaded and no training was run.")
        return

    from unsloth import FastLanguageModel, is_bfloat16_supported
    from datasets import Dataset
    import torch
    from trl import SFTConfig, SFTTrainer
    from unsloth.chat_templates import train_on_responses_only

    print(f"\nLoading base model: {args.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.r,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    train_data = Dataset.from_list(render_conversations(train_conversations, tokenizer))
    eval_data = (
        Dataset.from_list(render_conversations(eval_conversations, tokenizer))
        if eval_conversations is not None else None
    )
    bf16 = is_bfloat16_supported()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=args.max_seq_length,
            packing=False,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps if args.max_steps > 0 else -1,
            learning_rate=args.lr,
            logging_steps=10,
            optim="adamw_8bit" if not args.offload_optimizer else "adamw_torch",
            optim_args="cpu_offload=True" if args.offload_optimizer else None,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=str(args.output_dir),
            report_to="none",
            eval_strategy="steps" if eval_data else "no",
            eval_steps=50,
            per_device_eval_batch_size=1,
            fp16=not bf16,
            bf16=bf16,
            fp16_full_eval=not bf16,
            bf16_full_eval=bf16,
        ),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    gpu = torch.cuda.get_device_properties(0)
    initial_memory = torch.cuda.max_memory_reserved() / 1024 ** 3
    print(f"\nGPU: {gpu.name} — {gpu.total_memory / 1024 ** 3:.1f} GB")
    print(f"Initial reserved VRAM: {initial_memory:.2f} GB")
    print("Starting training...\n")
    stats = trainer.train()
    peak_memory = torch.cuda.max_memory_reserved() / 1024 ** 3
    runtime = stats.metrics.get("train_runtime", 0)
    print(f"\nTraining complete in {runtime / 60:.1f} minutes")
    print(f"Peak reserved VRAM: {peak_memory:.2f} GB")

    args.lora_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving LoRA adapter to {args.lora_dir}")
    model.save_pretrained(str(args.lora_dir))
    tokenizer.save_pretrained(str(args.lora_dir))

    if args.export_gguf is not None:
        methods = args.export_gguf or ["q4_k_m"]
        final_directory = Path(f"{args.gguf_base}_gguf")
        backup_existing_ggufs(final_directory)
        print(f"Exporting GGUF ({', '.join(methods)}) -> {final_directory}")
        model.save_pretrained_gguf(
            str(args.gguf_base), tokenizer, quantization_method=methods
        )
        print(f"GGUF export complete: {final_directory}")


if __name__ == "__main__":
    main()
