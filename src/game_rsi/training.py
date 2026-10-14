"""Native weight updates; checkpoints never masquerade as another architecture."""
from __future__ import annotations

import copy
import dataclasses
import json
import math
import random
import re
import shutil
import time
from pathlib import Path

from .config import RunConfig, upstream_paths
from .storage import Journal, atomic_json, fingerprint


def target_entropy(gold):
    """Mean entropy of soft labels: the attainable cross-entropy floor."""
    gold = gold.float()
    return -(gold * gold.clamp_min(1e-30).log()).sum(-1).mean()


def load_samples(data_root: Path, game: str, limit: int, seed: int):
    train, dev = [], []
    for manifest in sorted((data_root / game).glob("*/records.jsonl")):
        for line in manifest.read_text().splitlines():
            row = json.loads(line)
            row["frame_path"] = str(manifest.parent / row["frame"])
            (dev if row["seed"] % 10 == 0 else train).append(row)
    rng = random.Random(seed)
    # Bounded replay from earlier rounds prevents training exclusively on the newest mistakes.
    if len(train) > limit:
        train = rng.sample(train, limit)
    if len(dev) > 128:
        dev = rng.sample(dev, 128)
    if not train or not dev:
        raise ValueError("Collection must include disjoint train and validation episodes")
    return train, dev


def train_candidate(cfg: RunConfig, parent: str, data_root: Path, out: Path, journal: Journal, round_index: int):
    upstream_paths()
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from safetensors.torch import save_file

    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    train, dev = load_samples(data_root, cfg.game, cfg.replay_limit, cfg.seed)
    device = torch.device(cfg.device)
    started = time.perf_counter()
    parent_meta = None
    visual_prep = None
    if cfg.backend == "playjev":
        from playjev.data import Collate
        from playjev.train_sft import slot_ids_for, slot_logits
        from transformers import AutoModelForImageTextToText, AutoProcessor
        processor = AutoProcessor.from_pretrained(parent)
        processor.tokenizer.padding_side = "left"
        model = AutoModelForImageTextToText.from_pretrained(parent, dtype=torch.float32).to(device)
        model.config.use_cache = False
        layer_ids = [int(m.group(1)) for n, _ in model.named_parameters()
                     if "visual" not in n and (m := re.search(r"\.layers\.(\d+)\.", n))]
        last = max(layer_ids)
        for name, p in model.named_parameters():
            match = re.search(r"\.layers\.(\d+)\.", name)
            p.requires_grad_("visual" not in name and "embed_tokens" not in name and "lm_head" not in name
                             and (cfg.tune_tower or bool(match and int(match.group(1)) >= last - 1)))
        if cfg.tune_tower:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        collator = Collate(processor)
        slots = torch.tensor(slot_ids_for(processor.tokenizer, len(train[0]["actions"])), device=device)

        def forward(rows, shuffle=False):
            examples = []
            for row in rows:
                order = list(range(len(row["actions"])))
                if shuffle:
                    rng.shuffle(order)
                options = [row["options"][i] for i in order]
                with Image.open(row["frame_path"]) as image:
                    frame = image.convert("RGB")
                examples.append({"image": frame, "options": options,
                                 "target": [row["teacher_probs"][i] for i in order],
                                 "teacher_pos": order.index(row["teacher_action"]), "game": cfg.game,
                                 "n_opts": len(options)})
            batch = collator(examples)
            return slot_logits(model, batch, slots, device), batch["targets"].to(device)
    else:
        from serve.release import load_release, resolve_ckpt
        from rsijev.contract import Question
        from rsijev.encode import collate, encode_question, unpermute_logits
        resolved = resolve_ckpt(parent)
        parent_meta = json.loads((resolved / "meta.json").read_text())
        if (parent_meta["spec"].get("arch_extra") or {}).get("aux_exits"):
            raise ValueError("RSI training currently supports single-exit checkpoints; multi-exit releases can be used for play only")
        model, tokenizer, encoder, meta = load_release(resolved, device=str(device), adaptive="off")
        mode = ("pixels" if meta.get("vision") else "state") if cfg.input_mode == "auto" else cfg.input_mode
        if mode == "pixels" and not meta.get("vision"):
            raise ValueError("Pixel training requires a vision-capable RSI checkpoint")
        if mode == "pixels":
            from rsijev.vision import ImagePrep, VisionConfig, encode_vision_question, vision_collate
            visual_prep = ImagePrep(meta.get("weights_source") or meta["base_model"],
                                    VisionConfig(image_token_budget=meta["vision"]["image_token_budget"]),
                                    revision=meta["vision"].get("revision"))
        model.cal_mode = "none"
        model.cfg.freeze_base = not cfg.tune_tower
        for name, p in model.named_parameters():
            p.requires_grad_(name.startswith("scorer") or
                             (cfg.tune_tower and name.startswith("tower") and "embed_tokens" not in name))

        def forward(rows, shuffle=False):
            examples = []
            enc = dataclasses.replace(encoder, option_order="shuffled" if shuffle else "canonical")
            for row in rows:
                options = row["options"]
                q = Question("action", "choice", "Which move should the player make next?",
                             tuple(o["name"] for o in options), {o["name"]: o["description"] for o in options})
                if visual_prep:
                    with Image.open(row["frame_path"]) as im:
                        examples.append(encode_vision_question(tokenizer, visual_prep, "<image>",
                                                               [im.convert("RGB")], q, enc, rng=rng))
                else:
                    examples.append(encode_question(tokenizer, row["state"], q, enc, rng=rng))
            k = len(rows[0]["actions"])
            batch = (vision_collate if visual_prep else collate)(tokenizer, examples, k, device=device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(**batch)
            logits = unpermute_logits(logits.float(), batch["option_perm"], batch["option_mask"])
            target = torch.tensor([row["teacher_probs"] for row in rows], device=device)
            return logits, target

    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters selected")
    # Keep small exact before/after probes, not a second copy of the full network.
    probes = {n: p.detach().flatten()[:2048].float().cpu().clone() for n, p in parameters}
    heads = [p for n, p in parameters if n.startswith("scorer")]
    tower = [p for n, p in parameters if not n.startswith("scorer")]
    groups = []
    if heads:
        groups.append({"params": heads, "lr": cfg.learning_rate})
    if tower:
        groups.append({"params": tower, "lr": cfg.learning_rate * (0.1 if heads else 1)})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    base_lrs = [g["lr"] for g in groups]
    journal.event("training_setup", train_samples=len(train), dev_samples=len(dev),
                  trainable_parameters=sum(p.numel() for _, p in parameters), parent=parent,
                  input_mode=mode if cfg.backend == "rsi" else "pixels")

    def loss_of(logits, gold):
        logp = F.log_softmax(logits.float(), -1)
        return -(gold * logp.masked_fill(gold == 0, 0)).sum(-1).mean()

    def validate():
        model.eval()
        total, entropy_total, correct = 0.0, 0.0, 0
        with torch.no_grad():
            for i in range(0, len(dev), cfg.batch_size):
                journal.check()
                rows = dev[i:i + cfg.batch_size]
                logits, gold = forward(rows)
                total += float(loss_of(logits, gold)) * len(rows)
                entropy_total += float(target_entropy(gold)) * len(rows)
                correct += int((logits.argmax(-1) == gold.argmax(-1)).sum())
        return {"val_loss": total / len(dev), "agreement": correct / len(dev),
                "val_target_entropy": entropy_total / len(dev),
                "val_kl": (total - entropy_total) / len(dev)}

    initial = validate()
    journal.event("metric", step=0, round=round_index, **initial)
    for step in range(1, cfg.train_steps + 1):
        journal.check()
        model.train()
        rows = [train[rng.randrange(len(train))] for _ in range(cfg.batch_size)]
        optimizer.zero_grad(set_to_none=True)
        logits, gold = forward(rows, shuffle=True)
        loss = loss_of(logits, gold)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss; update aborted")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in parameters], 1.0, error_if_nonfinite=True)
        factor = 0.1 + 0.9 * (1 + math.cos(math.pi * (step - 1) / cfg.train_steps)) / 2
        for group, lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = lr * factor
        optimizer.step()
        entropy = float(target_entropy(gold))
        values = {"step": step, "round": round_index, "loss": float(loss.detach()),
                  "target_entropy": entropy, "kl": float(loss.detach()) - entropy,
                  "grad_norm": float(norm), "lr": optimizer.param_groups[0]["lr"],
                  "elapsed_s": time.perf_counter() - started,
                  "gpu_gb": torch.cuda.memory_allocated() / 2**30 if device.type == "cuda" else 0}
        if step % max(10, cfg.train_steps // 4) == 0 or step == cfg.train_steps:
            values.update(validate())
        journal.event("metric", **values)
        journal.publish(progress=step / cfg.train_steps, train_step=step, train_steps=cfg.train_steps,
                        loss=values["loss"], kl=values["kl"], target_entropy=entropy,
                        gpu_gb=values["gpu_gb"], elapsed_s=values["elapsed_s"])
    final = validate()
    changes = {n: float((p.detach().flatten()[:2048].float().cpu() - probes[n]).norm()) for n, p in parameters}
    if not any(value > 0 for value in changes.values()):
        raise RuntimeError("No parameter probe changed; refusing an empty training result")
    out.mkdir(parents=True, exist_ok=True)
    journal.phase("saving", round=round_index)
    model.eval()
    with torch.no_grad():
        probe_logits, _ = forward(dev[:1])
        probe_logits = probe_logits.cpu()
    if cfg.backend == "playjev":
        # Save the exact fp32 weights used in training/evaluation for reload parity.
        model.save_pretrained(out, safe_serialization=True)
        processor.save_pretrained(out)
    else:
        local = (resolved / "config.json").exists()
        save_file({k: v.detach().float().cpu().contiguous() for k, v in model.tower.state_dict().items()
                   if local or "embed_tokens" not in k}, str(out / "tower.safetensors"))
        save_file({k: v.detach().float().cpu().contiguous() for k, v in model.scorer.state_dict().items()},
                  str(out / "scorer.safetensors"))
        if local:
            for f in resolved.iterdir():
                if f.is_file() and (f.name == "config.json" or f.name.startswith(
                        ("tokenizer", "vocab", "merges", "special_tokens", "preprocessor", "processor", "visual"))):
                    shutil.copy2(f, out / f.name)
        tokenizer.save_pretrained(out)
        saved = copy.deepcopy(parent_meta)
        saved.pop("calibration", None)
        saved.pop("adaptive", None)
        saved["spec"].update(steps=cfg.train_steps, seed=cfg.seed, freeze_base=not cfg.tune_tower,
                             option_order="shuffled")
        saved["game_rsi"] = {"parent": parent, "config": cfg.model_dump(), "input_mode": mode}
        atomic_json(out / "meta.json", saved)
    # Checkpoint reload must reproduce the stored decision before candidate evaluation.
    del optimizer
    import gc
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if cfg.backend == "rsi":
        model, _, _, _ = load_release(out, device=str(device), adaptive="off")
        model.cal_mode = "none"
    else:
        model = AutoModelForImageTextToText.from_pretrained(out, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        restored, _ = forward(dev[:1])
    if not torch.allclose(probe_logits, restored.cpu(), atol=1e-4, rtol=1e-4):
        raise RuntimeError("Checkpoint failed prediction reload parity")
    report = {"parent": parent, "checkpoint": str(out), "sha256": fingerprint(out),
              "initial": initial, "final": final, "reload_parity": True,
              "steps": cfg.train_steps, "train_samples": len(train), "dev_samples": len(dev),
              "parameter_probe_deltas": {n: v for n, v in changes.items() if v > 0},
              "trainable_parameters": sum(p.numel() for _, p in parameters),
              "calibration": "none", "seconds": time.perf_counter() - started}
    atomic_json(out / "training.json", report)
    journal.event("trained", **{k: v for k, v in report.items() if k != "parameter_probe_deltas"})
    del model, parameters, heads, tower, groups
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report
