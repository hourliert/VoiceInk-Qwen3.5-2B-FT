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

try:
    from .finetune import align_final_eval_schedule, compute_fused_eval_loss
except ImportError:
    from finetune import align_final_eval_schedule, compute_fused_eval_loss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.mlflow_tracking import (  # noqa: E402
    add_mlflow_args,
    dataset_metadata,
    output_reference,
    start_mlflow_run,
)

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
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=profile.learning_rate)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--load-best-model-at-end", action="store_true")
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Disable evaluation for a quick smoke test",
    )
    parser.add_argument("--resume-from-checkpoint", type=Path)

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
    add_mlflow_args(parser, default_experiment="voiceink-training")
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
                    eval_data: list[dict] | None,
                    planned_steps: int) -> None:
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
    print(
        f"  Effective batch size: {args.batch_size * args.grad_accum} "
        f"({args.batch_size} x {args.grad_accum})"
    )
    print(f"  Evaluation batch size: {args.eval_batch_size}")
    print(f"  Epochs: {args.epochs}; max steps: {args.max_steps}")
    print(f"  Planned optimizer steps: {planned_steps}")
    print(
        f"  Evaluation: loss-only every {args.eval_steps} steps; "
        f"checkpoint every {args.save_steps} steps"
    )
    if eval_data is not None:
        print(f"  Terminal evaluation: guaranteed at step {planned_steps}")
    print(
        "  Final adapter source: "
        + ("best eval-loss checkpoint" if args.load_best_model_at_end else "last step")
    )
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


def main(
    profile: TrainingProfile = LFM25_12B_PROFILE,
    argv: list[str] | None = None,
) -> None:
    args = parse_args(profile, argv)
    if not args.train.is_file():
        print(f"Training data not found: {args.train}", file=sys.stderr)
        sys.exit(1)
    if min(args.batch_size, args.grad_accum, args.eval_batch_size, args.epochs,
           args.eval_steps, args.save_steps) <= 0:
        print("Training, evaluation, and checkpoint intervals must be positive",
              file=sys.stderr)
        sys.exit(2)

    train_conversations = load_conversations(args.train)
    eval_conversations = None
    if not args.skip_eval and args.eval and args.eval.is_file():
        eval_conversations = load_conversations(args.eval)

    planned_steps = align_final_eval_schedule(
        args, len(train_conversations), bool(eval_conversations)
    )
    if args.load_best_model_at_end and not eval_conversations:
        print("--load-best-model-at-end requires a non-empty --eval dataset",
              file=sys.stderr)
        sys.exit(2)
    if args.load_best_model_at_end and args.save_steps % args.eval_steps != 0:
        print("--save-steps must be a multiple of --eval-steps",
              file=sys.stderr)
        sys.exit(2)

    print_preflight(
        args, train_conversations, eval_conversations, planned_steps
    )
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
    tracking = start_mlflow_run(
        args,
        run_name=args.gguf_base.name,
        run_kind="training.sft.lfm",
        params={
            "profile": profile.name,
            "base_model": args.base_model,
            "max_seq_length": args.max_seq_length,
            "precision": (
                "4bit" if args.load_in_4bit else
                "8bit" if args.load_in_8bit else "bf16"
            ),
            "lora_r": args.r,
            "lora_alpha": args.lora_alpha,
            "lora_targets": LORA_TARGET_MODULES,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "planned_optimizer_steps": planned_steps,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.lr,
            "warmup_steps": args.warmup_steps,
            "eval_steps": args.eval_steps,
            "save_steps": args.save_steps,
            "load_best_model_at_end": args.load_best_model_at_end,
            "output_dir": args.output_dir,
            "lora_dir": args.lora_dir,
            "gguf_base": args.gguf_base,
            "gguf_quantizations": args.export_gguf or [],
        },
        datasets=[
            dataset_metadata("train", args.train, len(train_conversations)),
            *(
                [dataset_metadata(
                    "training_regression_eval", args.eval, len(eval_conversations)
                )]
                if eval_conversations is not None else []
            ),
        ],
    )

    class LossOnlyEvalSFTTrainer(SFTTrainer):
        """Evaluate fused loss without retaining full-vocabulary logits."""

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
            report_to="mlflow" if tracking.enabled else "none",
            run_name=tracking.run_name,
            eval_strategy="steps" if eval_data else "no",
            eval_steps=args.eval_steps,
            prediction_loss_only=True,
            per_device_eval_batch_size=args.eval_batch_size,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=2,
            load_best_model_at_end=args.load_best_model_at_end,
            metric_for_best_model=(
                "eval_loss" if args.load_best_model_at_end else None
            ),
            greater_is_better=False,
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
    resume_checkpoint = (
        str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    )
    stats = trainer.train(resume_from_checkpoint=resume_checkpoint)
    tracking.log_metrics(stats.metrics)
    peak_memory = torch.cuda.max_memory_reserved() / 1024 ** 3
    runtime = stats.metrics.get("train_runtime", 0)
    print(f"\nTraining complete in {runtime / 60:.1f} minutes")
    print(f"Peak reserved VRAM: {peak_memory:.2f} GB")
    if args.load_best_model_at_end:
        print(f"Best eval loss: {trainer.state.best_metric}")
        print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
        tracking.log_metrics({"best_eval_loss": trainer.state.best_metric})

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

    tracking.log_dict(
        {
            "global_step": trainer.state.global_step,
            "epoch": trainer.state.epoch,
            "best_metric": trainer.state.best_metric,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "log_history": trainer.state.log_history,
        },
        "training/trainer_state.json",
    )
    tracking.log_dict(
        {
            "outputs": [
                output_reference("checkpoints", args.output_dir, private=False),
                output_reference("lora_adapter", args.lora_dir, private=False),
                *(
                    [output_reference("gguf_export", final_directory, private=False)]
                    if args.export_gguf is not None else []
                ),
            ]
        },
        "metadata/outputs.json",
    )
    tracking.log_external_model(
        name=args.gguf_base.name,
        model_type="voiceink-transcript-cleanup",
        params={
            "base_model": args.base_model,
            "profile": profile.name,
            "lora_dir": args.lora_dir,
            "gguf_dir": (
                final_directory if args.export_gguf is not None else "not-exported"
            ),
            "quantizations": args.export_gguf or [],
        },
        tags={"voiceink.training_family": "lfm-sft"},
        metrics={
            **stats.metrics,
            "best_eval_loss": trainer.state.best_metric,
        },
        dataset_role="training_regression_eval",
    )
    tracking.finish()


if __name__ == "__main__":
    main()
