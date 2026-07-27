"""Config-driven training loop for the <SEG> bridge.

Designed for interrupted-anywhere execution (Colab preemption): checkpoints
carry model/optimizer/step state and training resumes from the latest one.
Heavy components (backbone, SAM2, LoRA) are assembled from the YAML config so
the same entrypoint serves CPU smoke runs and GPU training.

Run:
    uv run python -m voxae.train.train --config voxae/train/configs/smoke_2b.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from voxae.model.qwen_backbone import (
    LoraSettings,
    add_seg_token,
    apply_lora,
    load_backbone,
)
from voxae.model.sam2_head import Sam2DecoderHead
from voxae.model.seg_bridge import VoxaeSegModel
from voxae.train.collate import VoxaeCollator, load_samples


def build_model(cfg: dict, device: torch.device):
    backbone, processor = load_backbone(
        cfg["backbone_id"],
        load_in_4bit=cfg.get("load_in_4bit", False),
        dtype=cfg.get("dtype", "bfloat16"),
    )
    seg_token_id = add_seg_token(processor.tokenizer, backbone)
    if cfg.get("lora", True):
        backbone = apply_lora(backbone, LoraSettings(**cfg.get("lora_settings", {})))

    sam_head = Sam2DecoderHead.from_pretrained(
        cfg["sam2_id"], train_mask_decoder=cfg.get("train_mask_decoder", False)
    )
    model = VoxaeSegModel(
        backbone,
        sam_head,
        seg_token_id=seg_token_id,
        llm_hidden=backbone.config.get_text_config().hidden_size,
        ce_weight=cfg.get("ce_weight", 1.0),
        bce_weight=cfg.get("bce_weight", 2.0),
        dice_weight=cfg.get("dice_weight", 0.5),
    )
    return model.to(device), processor


def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


def save_checkpoint(path: Path, model, optimizer, scheduler, step: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    trainable = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
    torch.save(
        {
            "trainable": trainable,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
        },
        path / "state.pt",
    )


def load_checkpoint(path: Path, model, optimizer, scheduler) -> int:
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    current = dict(model.named_parameters())
    for name, tensor in state["trainable"].items():
        if name in current:
            current[name].data.copy_(tensor)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return int(state["step"])


def train(config_path: Path, fresh: bool = False) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model, processor = build_model(cfg, device)
    samples = load_samples(Path(cfg["train_jsonl"]), split=cfg.get("split", "train"))
    if cfg.get("limit"):
        samples = samples[: cfg["limit"]]
    collator = VoxaeCollator(
        processor=processor,
        data_root=Path(cfg["data_root"]),
        sam_image_size=model.sam_head.input_image_size,
        max_length=cfg.get("max_length", 512),
    )
    loader = DataLoader(
        samples,
        batch_size=cfg.get("batch_size", 1),
        shuffle=True,
        collate_fn=collator,
        num_workers=cfg.get("num_workers", 0),
    )

    steps_total = cfg.get("max_steps", len(loader) * cfg.get("epochs", 1))
    accum = cfg.get("grad_accum", 1)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=cfg.get("lr", 3e-4))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_lambda(s, cfg.get("warmup_steps", 50), steps_total)
    )

    step = 0
    latest = out_dir / "latest"
    # Resume is keyed on step count alone, so a finished run silently exits when
    # relaunched. Changing the config is a different experiment, not a resume.
    if fresh and latest.exists():
        shutil.rmtree(latest)
        print(f"discarded checkpoint at {latest}")
    if (latest / "state.pt").exists():
        step = load_checkpoint(latest, model, optimizer, scheduler)
        print(f"resumed from step {step}")

    use_wandb = bool(cfg.get("wandb_project"))
    if use_wandb:
        import wandb

        wandb.init(project=cfg["wandb_project"], config=cfg, resume="allow")

    model.train()
    log_path = out_dir / "train_log.jsonl"
    t0 = time.time()
    while step < steps_total:
        for batch in loader:
            if step >= steps_total:
                break
            gt = batch.pop("gt_masks").to(device)
            sam_px = batch.pop("sam_pixel_values").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                sam_feats = model.sam_head.encode_images(sam_px)

            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                sam_features=sam_feats,
                gt_masks=gt,
                labels=batch.get("labels"),
                **{
                    k: v
                    for k, v in batch.items()
                    if k not in {"input_ids", "attention_mask", "labels"}
                },
            )
            (out["loss"] / accum).backward()

            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % cfg.get("log_every", 10) == 0:
                record = {
                    "step": step,
                    "elapsed_s": round(time.time() - t0, 1),
                    **{
                        k: round(float(v.detach()), 5)
                        for k, v in out.items()
                        if k.startswith("loss")
                    },
                }
                print(json.dumps(record))
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                if use_wandb:
                    import wandb

                    wandb.log(record, step=step)

            if step > 0 and step % cfg.get("save_every", 200) == 0:
                save_checkpoint(latest, model, optimizer, scheduler, step)
            step += 1

    save_checkpoint(latest, model, optimizer, scheduler, step)
    print(f"done at step {step}; checkpoint -> {latest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fresh", action="store_true", help="discard any existing checkpoint")
    args = parser.parse_args()
    train(args.config, fresh=args.fresh)
