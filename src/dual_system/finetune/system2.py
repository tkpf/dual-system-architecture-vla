"""Finetune SmolVLM with LoRA to judge whether a subtask is finished.

    python -m dual_system.finetune.system2 train --data <res> --res <res> [--steps 1500]    # resumes from the newest checkpoint
    python -m dual_system.finetune.system2 test --data <res> --res <res> --model lora|smolvlm|qwen2b --split test
    python -m dual_system.finetune.system2 export --res <res> --out <dir>    # adapter folder for QwenReasoner

The loss looks only at the yes and no tokens. LoRA goes on the text attention layers and the connector from vision; the vision
encoder stays frozen. The best step comes from balanced accuracy on validation episodes, and the decision threshold from the
same frames, favouring few false "finished": a false finish skips a subtask or ends the episode.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..eval.perception import MODELS, balanced_accuracy, episode_interval, parse_yes_no
from ..knowledge import StaticRetriever
from ..system2.reasoners import IMAGES_KWARGS, phase_done_prompt
from . import checkpoint

NAME = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
TARGET = r".*text_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)|.*connector\.modality_projection\.proj"
MAX_FALSE_DONE = 0.05
IMAGE_SIZE = 256


def load_rows(data: Path, split: str) -> list[dict]:
    rows = [json.loads(line) for line in open(data / "judge_labels.jsonl", encoding="utf-8")]
    return [r for r in rows if r["split"] == split]


def balanced_order(rows: list[dict], rng: np.random.Generator) -> list[int]:
    """Equal numbers of finished and unfinished rows, with the smaller class repeated, shuffled."""
    pos = [i for i, r in enumerate(rows) if r["done"]]
    neg = [i for i, r in enumerate(rows) if not r["done"]]
    n = max(len(pos), len(neg))
    if not pos or not neg:
        raise ValueError(f"one class is empty: {len(pos)} finished, {len(neg)} unfinished")
    order = np.concatenate([rng.choice(pos, n, replace=len(pos) < n), rng.choice(neg, n, replace=len(neg) < n)])
    rng.shuffle(order)
    return [int(i) for i in order]


def false_done_rate(rows: list[dict]) -> float:
    """Unfinished frames that the judge called finished."""
    neg = [r for r in rows if not r["truth"]]
    return sum(r["answer"] is True for r in neg) / len(neg) if neg else float("nan")


def choose_threshold(probs: list[float], truths: list[bool], max_false_done: float = MAX_FALSE_DONE) -> float:
    """The threshold with the best balanced accuracy among those within the false-done limit, or the strictest one tried."""
    best, best_score = 0.99, -1.0
    for t in (i / 100 for i in range(5, 100)):
        rows = [dict(truth=bool(x), answer=bool(p >= t)) for p, x in zip(probs, truths)]
        score = balanced_accuracy(rows)
        if false_done_rate(rows) <= max_false_done and score > best_score:
            best, best_score = t, score
    return best


def macro_by_step_kind(rows: list[dict]) -> dict:
    return {k: round(balanced_accuracy([r for r in rows if r["kind"] == k]), 3) for k in sorted({r["kind"] for r in rows})}


class Judge:
    """Prompting and batching shared by training and testing."""

    def __init__(self, data: Path):
        from transformers import AutoProcessor

        self.data = data
        self.processor = AutoProcessor.from_pretrained(NAME)
        self.processor.tokenizer.padding_side = "left"
        self.retriever = StaticRetriever("phase_atomic")
        tokens = [self.processor.tokenizer(w, add_special_tokens=False).input_ids for w in (" yes", " no")]
        assert all(len(t) == 1 for t in tokens), tokens
        self.yes, self.no = tokens[0][0], tokens[1][0]

    def images(self, frame: str) -> list:
        from PIL import Image

        return [Image.open(self.data / "frames" / f"{frame}_{cam}.jpg").convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)) for cam in ("agent", "wrist")]

    def batch(self, rows: list[dict]) -> dict:
        texts = []
        for r in rows:
            prompt = phase_done_prompt(r["step"], self.retriever.phase_definition(r["step"]))
            message = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": prompt}]}]
            texts.append(self.processor.apply_chat_template(message, add_generation_prompt=True))
        return self.processor(text=texts, images=[self.images(r["frame"]) for r in rows], return_tensors="pt", padding=True, images_kwargs=IMAGES_KWARGS).to("cuda")

    def logits(self, model, b: dict) -> torch.Tensor:
        """No and yes logits for the first answer token. Padding is on the left, so that is the last position."""
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(**b).logits[:, -1, :]
        return out[:, [self.no, self.yes]].float()


@torch.no_grad()
def probabilities(judge: Judge, model, rows: list[dict], batch_size: int = 8) -> list[float]:
    model.eval()
    out = []
    for i in range(0, len(rows), batch_size):
        out += torch.softmax(judge.logits(model, judge.batch(rows[i:i + batch_size])), dim=1)[:, 1].tolist()
    model.train()
    return out


def score(rows: list[dict], probs: list[float], threshold: float) -> list[dict]:
    return [dict(truth=bool(r["done"]), answer=bool(p >= threshold), question=r["kind"], episode=r["episode"], prob=round(p, 4), frame=r["frame"], step=r["step"], kind=r["kind"], mode=r["mode"], task=r["task"]) for r, p in zip(rows, probs)]


def build_lora():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(NAME, dtype=torch.float16).to("cuda")
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=TARGET, bias="none"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # float16 weights cannot take a gradient step on their own
    model.config.use_cache = False
    return model


def train(a) -> None:
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    from ..finetune.system1 import lr_factor

    data, res = Path(a.data), Path(a.res)
    root = res / "judge_checkpoints"
    res.mkdir(parents=True, exist_ok=True)
    judge = Judge(data)
    rows, val = load_rows(data, "train"), load_rows(data, "val")
    print("train rows", len(rows), "finished", sum(r["done"] for r in rows), "val rows", len(val), "finished", sum(r["done"] for r in val), flush=True)
    model = build_lora()
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: lr_factor(s, a.steps, 50))
    step, history = 0, []
    last = checkpoint.latest(root)
    if last:
        restored = checkpoint.load(last, optim, sched)
        set_peft_model_state_dict(model, restored["weights"])
        step, history = restored["state"]["step"], restored["state"]["extra"].get("history", [])
        print("resumed", last.name, flush=True)
    rng = np.random.default_rng(a.seed + step)
    order: list[int] = []
    metrics = open(res / "judge_metrics.jsonl", "a", encoding="utf-8")
    model.train()
    t0 = time.time()
    while step < a.steps:
        if len(order) < a.batch_size:
            order = balanced_order(rows, rng)
        part, order = [rows[i] for i in order[:a.batch_size]], order[a.batch_size:]
        loss = torch.nn.functional.cross_entropy(judge.logits(model, judge.batch(part)), torch.tensor([int(r["done"]) for r in part], device="cuda"))
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optim.step()
        sched.step()
        step += 1
        if step % a.log_every == 0:
            row = dict(step=step, loss=round(float(loss), 4), s_per_step=round((time.time() - t0) / a.log_every, 2), gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
            t0 = time.time()
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()
            print(json.dumps(row), flush=True)
        if step % a.save_every == 0 or step >= a.steps:
            probs = probabilities(judge, model, val)
            truths = [bool(r["done"]) for r in val]
            threshold = choose_threshold(probs, truths)
            scored = score(val, probs, threshold)
            row = dict(step=step, val_score=round(balanced_accuracy(scored), 4), false_done=round(false_done_rate(scored), 4), threshold=threshold, by_kind=macro_by_step_kind(scored))
            history.append(row)
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()
            print(json.dumps(row), flush=True)
            path = checkpoint.save(root, step, {k: v.cpu() for k, v in get_peft_model_state_dict(model).items()}, optim, sched, dict(history=history))
            best = max(history, key=lambda h: h["val_score"])
            (res / "JUDGE_BEST.json").write_text(json.dumps(best))
            checkpoint.prune(root, keep_last=2, keep={best["step"]})
            print("saved", path.name, "best step", best["step"], flush=True)
            t0 = time.time()
            if len(history) > a.patience and all(h["val_score"] <= best["val_score"] for h in history[-a.patience:]) and best not in history[-a.patience:]:
                print("stopped: no better validation score in", a.patience, "evaluations", flush=True)
                break


def load_best(res: Path):
    from peft import set_peft_model_state_dict

    best = json.loads((res / "JUDGE_BEST.json").read_text())
    model = build_lora()
    weights = torch.load(res / "judge_checkpoints" / f"step_{best['step']:07d}" / "weights.pt", map_location="cpu")
    set_peft_model_state_dict(model, weights)
    return model.eval(), best


def test_lora(judge: Judge, res: Path, rows: list[dict]) -> list[dict]:
    model, best = load_best(res)
    return score(rows, probabilities(judge, model, rows), best["threshold"])


def export(a) -> None:
    """The best step as an adapter folder for QwenReasoner, with its threshold next to it."""
    model, best = load_best(Path(a.res))
    model.save_pretrained(a.out)
    (Path(a.out) / "threshold.json").write_text(json.dumps(best))
    print("exported", a.out, best, flush=True)


def test_baseline(judge: Judge, rows: list[dict], key: str) -> list[dict]:
    """An untrained model, answering as it would in use: free generation, parsed as yes or no. Unusable counts as wrong."""
    from ..system2.reasoners import QwenReasoner

    reasoner = QwenReasoner(**MODELS[key], images_kwargs=IMAGES_KWARGS if key == "smolvlm" else None)
    out = []
    for r in rows:
        images = [np.asarray(i) for i in judge.images(r["frame"])]
        a = reasoner._generate(phase_done_prompt(r["step"], judge.retriever.phase_definition(r["step"])), images, 4, "phase_done_v1", parse_yes_no)
        out.append(dict(truth=bool(r["done"]), answer=a.parsed["yes"] if a.parsed else None, question=r["kind"], episode=r["episode"], raw=a.raw, frame=r["frame"], step=r["step"], kind=r["kind"], mode=r["mode"], task=r["task"]))
    return out


def test(a) -> None:
    data, res = Path(a.data), Path(a.res)
    judge = Judge(data)
    rows = load_rows(data, a.split)
    scored = test_lora(judge, res, rows) if a.model == "lora" else test_baseline(judge, rows, a.model)
    path = res / f"judge_{a.split}_{a.model}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in scored), encoding="utf-8")
    lo, hi = episode_interval(scored)
    print(json.dumps(dict(model=a.model, split=a.split, rows=len(scored), finished=sum(r["truth"] for r in scored), score=round(balanced_accuracy(scored), 3),
                          interval=[round(lo, 3), round(hi, 3)], false_done=round(false_done_rate(scored), 3), by_kind=macro_by_step_kind(scored),
                          unusable=sum(r["answer"] is None for r in scored))), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["train", "test", "export"])
    p.add_argument("--data")
    p.add_argument("--out")
    p.add_argument("--res", required=True)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--model", choices=["lora", "smolvlm", "qwen2b"], default="lora")
    p.add_argument("--split", default="test")
    a = p.parse_args()
    {"train": train, "test": test, "export": export}[a.command](a)
