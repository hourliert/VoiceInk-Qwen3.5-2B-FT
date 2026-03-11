#!/usr/bin/env python3
"""Fine-tune Qwen 3.5 2B for VoiceInk transcription cleanup using Unsloth.

Qwen 3.5 2B is a unified VLM, so we use FastVisionModel even for text-only
tasks. Loads training data (from prepare_dataset.py), applies LoRA, trains
with SFT, and saves the adapter + optionally exports to GGUF.

Directory conventions:
    training/lora/      — LoRA adapter checkpoints
    training/outputs/   — training logs and intermediate outputs
    models/             — final GGUF exports (for llama-server)

Usage:
    python3 src/training/finetune.py
    python3 src/training/finetune.py --export-gguf q4_k_m
    python3 src/training/finetune.py --epochs 3 --lr 2e-4 --r 16
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = ROOT / "datasets" / "train.jsonl"
DEFAULT_EVAL = ROOT / "datasets" / "eval.jsonl"
DEFAULT_LORA_DIR = ROOT / "training" / "lora"
DEFAULT_OUTPUT_DIR = ROOT / "training" / "outputs"
DEFAULT_GGUF_DIR = ROOT / "models"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Qwen 3.5 2B with Unsloth.")

    # Data
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval", type=Path, default=DEFAULT_EVAL)

    # Model
    p.add_argument("--base-model", default="unsloth/Qwen3.5-2B",
                   help="Base model to fine-tune")
    p.add_argument("--max-seq-length", type=int, default=16384,
                   help="Max sequence length for training")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="Load base model in 4bit (saves VRAM, slightly lower quality)")

    # LoRA
    p.add_argument("--r", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")

    # Training
    p.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    p.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    p.add_argument("--warmup-steps", type=int, default=10, help="Warmup steps")
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Max training steps (-1 = use epochs)")

    # Output
    p.add_argument("--lora-dir", type=Path, default=DEFAULT_LORA_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--export-gguf", nargs="*", default=None,
                   help="Export to GGUF after training. Quantization methods, e.g.: q4_k_m q8_0")

    return p.parse_args()


def load_dataset_jsonl(path: Path) -> list[dict]:
    """Load a JSONL dataset into a list of conversation dicts."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    args = parse_args()

    if not args.train.exists():
        print(f"Training data not found: {args.train}", file=sys.stderr)
        sys.exit(1)

    # ---- Load model (VLM — Qwen 3.5 is a unified vision-language model) ----
    from unsloth import FastVisionModel
    import torch

    print(f"Loading base model: {args.base_model}")
    model, tokenizer = FastVisionModel.from_pretrained(
        args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )

    # ---- Add LoRA adapters ----
    print(f"Adding LoRA adapters (r={args.r}, alpha={args.lora_alpha})")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,      # Text-only task
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

    # ---- Load dataset ----
    print(f"Loading training data from {args.train}")
    train_data = load_dataset_jsonl(args.train)
    print(f"  {len(train_data)} training samples")

    eval_data = None
    if args.eval.exists():
        eval_data = load_dataset_jsonl(args.eval)
        print(f"  {len(eval_data)} eval samples")

    # ---- Train ----
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    FastVisionModel.for_training(model)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps if args.max_steps > 0 else -1,
            learning_rate=args.lr,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=str(args.output_dir),
            report_to="none",
            # Required for vision finetuning:
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=args.max_seq_length,
        ),
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"\nGPU: {gpu_stats.name} — {max_memory} GB total, {start_gpu_memory} GB reserved")
    print("Starting training...\n")

    trainer_stats = trainer.train()

    # Report stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    print(f"\nTraining complete.")
    print(f"  Runtime: {trainer_stats.metrics['train_runtime']:.0f}s "
          f"({trainer_stats.metrics['train_runtime']/60:.1f}min)")
    print(f"  Peak VRAM: {used_memory} GB / {max_memory} GB "
          f"({round(used_memory/max_memory*100, 1)}%)")

    # ---- Save LoRA adapter ----
    args.lora_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving LoRA adapter to {args.lora_dir}")
    model.save_pretrained(str(args.lora_dir))
    tokenizer.save_pretrained(str(args.lora_dir))

    # ---- Export to GGUF ----
    if args.export_gguf is not None:
        quant_methods = args.export_gguf if args.export_gguf else ["q4_k_m"]
        # Unsloth appends "_gguf" to the path, so we use a base name
        # that produces the final directory we want.
        gguf_base = DEFAULT_GGUF_DIR / "Qwen3.5-2B-voiceink"
        gguf_final = Path(str(gguf_base) + "_gguf")
        print(f"\nExporting to GGUF ({', '.join(quant_methods)}) -> {gguf_final}")
        model.save_pretrained_gguf(
            str(gguf_base),
            tokenizer,
            quantization_method=quant_methods,
        )
        print(f"GGUF export complete: {gguf_final}")

    print("\nDone!")


if __name__ == "__main__":
    main()
