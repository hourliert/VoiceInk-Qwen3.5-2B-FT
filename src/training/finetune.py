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
import filecmp
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = ROOT / "datasets" / "train.jsonl"
DEFAULT_EVAL = ROOT / "datasets" / "eval.jsonl"
DEFAULT_LORA_DIR = ROOT / "training" / "lora"
DEFAULT_OUTPUT_DIR = ROOT / "training" / "outputs"
DEFAULT_GGUF_DIR = ROOT / "models"


def parse_args(argv=None) -> argparse.Namespace:
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
    p.add_argument("--load-in-8bit", action="store_true",
                   help="Load base model in 8bit (saves ~2GB VRAM, minor quality impact)")
    p.add_argument("--offload-optimizer", action="store_true",
                   help="Offload optimizer states to CPU (saves ~3GB VRAM, slower training)")

    # LoRA
    p.add_argument("--r", type=int, default=32, help="LoRA rank")
    p.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")

    # Training
    p.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    p.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    p.add_argument("--warmup-steps", type=int, default=10, help="Warmup steps")
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Max training steps (-1 = use epochs)")
    p.add_argument("--eval-steps", type=int, default=50,
                   help="Run loss-only evaluation every N training steps")
    p.add_argument("--save-steps", type=int, default=25,
                   help="Save a resumable checkpoint every N training steps")
    p.add_argument(
        "--resume-from-checkpoint", type=Path,
        help="Resume model, optimizer, scheduler, and trainer state from a checkpoint",
    )

    # Output
    p.add_argument("--lora-dir", type=Path, default=DEFAULT_LORA_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--export-gguf", nargs="*", default=None,
                   help="Export to GGUF after training. Quantization methods, e.g.: q4_k_m q8_0")
    p.add_argument(
        "--gguf-base", type=Path,
        default=DEFAULT_GGUF_DIR / "Qwen3.5-2B-voiceink",
        help="Base export path; Unsloth appends _gguf",
    )
    p.add_argument(
        "--check-only", action="store_true",
        help="Validate and print the training configuration without loading a model",
    )

    return p.parse_args(argv)


def snapshot_labeled_data() -> str | None:
    """Snapshot datasets/labeled.jsonl if it changed since the last snapshot.

    Returns the snapshot filename used (existing or new), or None if
    labeled.jsonl doesn't exist.
    """
    labeled = ROOT / "datasets" / "labeled.jsonl"
    if not labeled.exists():
        return None

    # Count samples
    n_samples = sum(1 for line in labeled.open() if line.strip())

    # Find most recent existing snapshot
    snapshots = sorted(ROOT.glob("datasets/labeled.*.jsonl"))
    if snapshots:
        latest = snapshots[-1]
        if filecmp.cmp(labeled, latest, shallow=False):
            print(f"Dataset snapshot unchanged: {latest.name}")
            return latest.name

    # Create new snapshot
    date_str = datetime.now().strftime("%Y%m%d")
    name = f"labeled.{date_str}.{n_samples}samples.jsonl"
    dest = ROOT / "datasets" / name
    shutil.copy2(labeled, dest)
    print(f"Dataset snapshot created: {name}")
    return name


def load_dataset_jsonl(path: Path) -> list[dict]:
    """Load a JSONL dataset into a list of conversation dicts."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_conversations(records: list[dict], path: Path) -> None:
    """Fail closed on malformed Qwen typed-content conversations."""
    expected_roles = ["system", "user", "assistant"]
    for index, record in enumerate(records):
        messages = record.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"{path}:{index + 1} has no messages list")
        if [message.get("role") for message in messages] != expected_roles:
            raise ValueError(f"{path}:{index + 1} has unexpected message roles")
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list) or not content:
                raise ValueError(
                    f"{path}:{index + 1} requires non-empty typed content blocks"
                )
            if not any(
                isinstance(block, dict) and block.get("type") == "text"
                and str(block.get("text", "")).strip()
                for block in content
            ):
                raise ValueError(f"{path}:{index + 1} has no non-empty text block")


def print_preflight(args: argparse.Namespace, train_data: list[dict],
                    eval_data: list[dict] | None) -> None:
    precision = "4-bit" if args.load_in_4bit else (
        "8-bit" if args.load_in_8bit else "BF16"
    )
    quant_methods = args.export_gguf if args.export_gguf is not None else []
    print("Qwen3.5 training preflight")
    print(f"  Base model: {args.base_model}")
    print(
        f"  Train: {args.train} ({len(train_data)} samples, "
        f"sha256={sha256_file(args.train)})"
    )
    if eval_data is not None:
        print(
            f"  Eval: {args.eval} ({len(eval_data)} samples, "
            f"sha256={sha256_file(args.eval)})"
        )
    else:
        print("  Eval: disabled")
    print(f"  Max sequence length: {args.max_seq_length}")
    print(f"  LoRA: r={args.r}, alpha={args.lora_alpha}")
    print(
        f"  Effective batch size: {args.batch_size * args.grad_accum} "
        f"({args.batch_size} x {args.grad_accum})"
    )
    print(f"  Epochs: {args.epochs}; max steps: {args.max_steps}")
    print(f"  Learning rate: {args.lr}; warmup steps: {args.warmup_steps}")
    print(
        f"  Evaluation: loss-only every {args.eval_steps} steps; "
        f"checkpoint every {args.save_steps} steps"
    )
    if args.resume_from_checkpoint:
        print(f"  Resume checkpoint: {args.resume_from_checkpoint}")
    print(f"  Load precision: {precision}")
    print(f"  Adapter output: {args.lora_dir}")
    print(f"  Checkpoints: {args.output_dir}")
    if args.export_gguf is None:
        print("  GGUF export: disabled")
    else:
        methods = quant_methods or ["q4_k_m"]
        print(f"  GGUF export: {args.gguf_base}_gguf ({', '.join(methods)})")


def compute_fused_eval_loss(trainer, model, inputs, torch_module):
    """Compute eval loss while explicitly keeping Unsloth logits disabled."""
    inputs = trainer._prepare_inputs(inputs)
    previous = os.environ.get("UNSLOTH_RETURN_LOGITS")
    os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
    try:
        with torch_module.no_grad():
            with trainer.compute_loss_context_manager():
                loss = trainer.compute_loss(model, inputs, return_outputs=False)
        return loss.mean().detach(), None, None
    finally:
        if previous is None:
            os.environ.pop("UNSLOTH_RETURN_LOGITS", None)
        else:
            os.environ["UNSLOTH_RETURN_LOGITS"] = previous


def main() -> None:
    args = parse_args()

    if not args.train.exists():
        print(f"Training data not found: {args.train}", file=sys.stderr)
        sys.exit(1)

    if args.load_in_4bit and args.load_in_8bit:
        print("Use only one of --load-in-4bit or --load-in-8bit", file=sys.stderr)
        sys.exit(2)

    print(f"Loading training data from {args.train}")
    train_data = load_dataset_jsonl(args.train)
    validate_conversations(train_data, args.train)

    eval_data = None
    if args.eval and args.eval.is_file():
        eval_data = load_dataset_jsonl(args.eval)
        validate_conversations(eval_data, args.eval)

    print_preflight(args, train_data, eval_data)
    if args.check_only:
        print("Preflight complete; no model was loaded and no training was run.")
        return

    # Snapshot labeled data for reproducibility
    snapshot_labeled_data()

    # ---- Load model (VLM — Qwen 3.5 is a unified vision-language model) ----
    from unsloth import FastVisionModel
    import torch

    print(f"Loading base model: {args.base_model}")
    model, tokenizer = FastVisionModel.from_pretrained(
        args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
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

    # ---- Train ----
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    FastVisionModel.for_training(model)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    class LossOnlyEvalSFTTrainer(SFTTrainer):
        """Evaluate fused cross-entropy without materializing vocabulary logits.

        Unsloth's generic prediction step forces UNSLOTH_RETURN_LOGITS=1 even
        when Transformers requests loss only. For Qwen3.5's 248k vocabulary,
        a long eval example can make that temporary tensor exceed GPU memory.
        """

        def prediction_step(
            self, model, inputs, prediction_loss_only, ignore_keys=None
        ):
            if not prediction_loss_only:
                return super().prediction_step(
                    model, inputs, prediction_loss_only, ignore_keys
                )
            return compute_fused_eval_loss(self, model, inputs, torch)

    trainer = LossOnlyEvalSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(
            model, tokenizer,
            train_on_responses_only=True,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        ),
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=SFTConfig(
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
            # Evaluation during training (detect overfitting)
            eval_strategy="steps" if eval_data else "no",
            eval_steps=args.eval_steps,
            prediction_loss_only=True,
            fp16_full_eval=not args.load_in_4bit,
            per_device_eval_batch_size=1,
            # Save before the first evaluation so interruptions are resumable.
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=2,
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

    resume_checkpoint = (
        str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    )
    trainer_stats = trainer.train(resume_from_checkpoint=resume_checkpoint)

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
        gguf_base = args.gguf_base
        gguf_final = Path(str(gguf_base) + "_gguf")

        # Back up existing GGUF files before overwriting (v1, v2, v3, ...)
        if gguf_final.exists():
            for gguf_file in sorted(gguf_final.glob("*.gguf")):
                if ".v" in gguf_file.stem or gguf_file.name.startswith("Qwen3.5-2B.BF16"):
                    continue
                # Find next version number
                existing_versions = sorted(gguf_final.glob(f"{gguf_file.stem}.v*.gguf"))
                next_v = 1
                for v_file in existing_versions:
                    try:
                        n = int(v_file.stem.rsplit(".v", 1)[1])
                        next_v = max(next_v, n + 1)
                    except (ValueError, IndexError):
                        pass
                backup = gguf_file.with_name(f"{gguf_file.stem}.v{next_v}.gguf")
                print(f"\nBacking up {gguf_file.name} -> {backup.name}")
                shutil.copy2(gguf_file, backup)

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
