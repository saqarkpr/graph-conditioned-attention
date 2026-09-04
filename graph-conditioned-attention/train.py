"""
Trains VanillaArcPredictor (plain Transformer, no graph conditioning) and
GraphToGraphTransformer (attention biased by an iteratively-refined graph,
starting from a naive linear-chain guess) on the same unlabeled dependency
arc prediction task, and compares them on held-out UD English-EWT sentences.

Metric: unlabeled attachment accuracy — for each real (non-root) token, is
the predicted head (argmax over the arc-logit column) the correct gold head?

Hypothesis to test, not a guaranteed result: conditioning attention on an
explicit (even if initially naive) graph, and refining that graph across
layers, should let the model route information more effectively than
uniform self-attention with no structural prior, especially on longer
sentences where the naive linear-chain initial guess is wrong for most
tokens but still gives the model *something* structural to correct.

Mirrors the other three projects' structure: `run_experiment(args)` is
factored out of `main()`, both trained models are checkpointed (not just
logged to CSV), and `--seed`/`--n_seeds` give a single reproducible run or
a mean±std study across seeds.
"""
import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import load_data, MAX_LEN
from model import GraphToGraphTransformer, VanillaArcPredictor


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate(batch):
    ids = torch.stack([b[0] for b in batch])
    pad_mask = torch.stack([b[1] for b in batch])
    gold_arcs = torch.stack([b[2] for b in batch])
    init_arcs = torch.stack([b[3] for b in batch])
    return ids, pad_mask, gold_arcs, init_arcs


def arc_loss(logits, gold_arcs, pad_mask):
    # for each dependent column, the correct head is a single row -> cross-entropy
    # over the "head" dimension, restricted to real (non-pad) dependent tokens.
    B, T, _ = logits.shape
    logits_t = logits.transpose(1, 2)  # (B, T_dep, T_head) so softmax is over heads per dependent
    gold_head_idx = gold_arcs.transpose(1, 2).argmax(dim=-1)  # (B, T_dep) -- 0 if no head (root/pad)
    has_head = gold_arcs.transpose(1, 2).sum(dim=-1) > 0  # (B, T_dep) real arcs only

    loss = F.cross_entropy(
        logits_t.reshape(-1, T), gold_head_idx.reshape(-1), reduction="none"
    ).view(B, T)
    mask = has_head & pad_mask
    return (loss * mask.float()).sum() / mask.float().sum().clamp(min=1)


@torch.no_grad()
def unlabeled_attachment_accuracy(logits, gold_arcs, pad_mask):
    gold_head_idx = gold_arcs.transpose(1, 2).argmax(dim=-1)
    has_head = gold_arcs.transpose(1, 2).sum(dim=-1) > 0
    pred_head_idx = logits.transpose(1, 2).argmax(dim=-1)
    mask = has_head & pad_mask
    correct = ((pred_head_idx == gold_head_idx) & mask).sum().item()
    total = mask.sum().item()
    return correct, total


def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, total_correct, total_toks = 0.0, 0, 0

    for ids, pad_mask, gold_arcs, init_arcs in loader:
        ids, pad_mask = ids.to(device), pad_mask.to(device)
        gold_arcs, init_arcs = gold_arcs.to(device), init_arcs.to(device)

        with torch.set_grad_enabled(is_train):
            logits, _ = model(ids, init_arcs, pad_mask)
            loss = arc_loss(logits, gold_arcs, pad_mask)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        correct, total = unlabeled_attachment_accuracy(logits.detach(), gold_arcs, pad_mask)
        total_loss += loss.item() * ids.size(0)
        total_correct += correct
        total_toks += total

    return total_loss / len(loader.dataset), total_correct / max(total_toks, 1)


def train_model(model, train_loader, dev_loader, device, epochs, lr, name, log_path):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_UAS", "dev_loss", "dev_UAS"])

    for ep in range(1, epochs + 1):
        train_loss, train_uas = run_epoch(model, train_loader, device, optimizer)
        dev_loss, dev_uas = run_epoch(model, dev_loader, device, optimizer=None)
        print(f"[{name}] epoch {ep:3d} | train_loss {train_loss:.4f} train_UAS {train_uas:.3f} "
              f"| dev_loss {dev_loss:.4f} dev_UAS {dev_uas:.3f}")
        history.append((ep, train_loss, train_uas, dev_loss, dev_uas))
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, train_loss, train_uas, dev_loss, dev_uas])
    return history


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=800)
    p.add_argument("--n_dev", type=int, default=200)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--out_dir", type=str, default="results")
    p.add_argument("--tag", type=str, default="run1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init_graph", choices=["chain", "random", "empty"], default="chain",
                    help="control: `random` and `empty` test whether the model reads the graph at all")
    p.add_argument("--n_seeds", type=int, default=1,
                    help="if > 1, repeats both models at seeds 42, 43, ... and reports mean +/- std dev UAS")
    return p


def run_experiment(args) -> dict:
    """Trains + evaluates both models once, at args.seed. Writes checkpoints,
    per-epoch CSV logs for each model, and a combined comparison.csv.
    Returns a summary dict."""
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()

    print(f"[{args.tag}] loading UD English-EWT subset...")
    train_ds, dev_ds, vocab = load_data(n_train=args.n_train, n_dev=args.n_dev, max_len=MAX_LEN,
                                         init_graph=getattr(args, "init_graph", "chain"))
    print(f"[{args.tag}] train: {len(train_ds)} sentences | dev: {len(dev_ds)} sentences | vocab: {vocab.vocab_size}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    print(f"\n=== [{args.tag}] training vanilla (no graph conditioning) baseline, seed={args.seed} ===")
    baseline = VanillaArcPredictor(vocab.vocab_size, args.d_model, args.n_heads, args.n_layers, MAX_LEN).to(device)
    base_history = train_model(baseline, train_loader, dev_loader, device, args.epochs, args.lr, "baseline",
                                os.path.join(run_dir, "loss_log_baseline.csv"))
    torch.save({"model_state": baseline.state_dict(), "kind": "baseline", "args": vars(args),
                "vocab_size": vocab.vocab_size}, os.path.join(run_dir, "model_baseline.pt"))

    print(f"\n=== [{args.tag}] training graph-to-graph transformer, seed={args.seed} ===")
    g2g = GraphToGraphTransformer(vocab.vocab_size, args.d_model, args.n_heads, args.n_layers, MAX_LEN).to(device)
    g2g_history = train_model(g2g, train_loader, dev_loader, device, args.epochs, args.lr, "g2g",
                               os.path.join(run_dir, "loss_log_g2g.csv"))
    torch.save({"model_state": g2g.state_dict(), "kind": "g2g", "args": vars(args),
                "vocab_size": vocab.vocab_size}, os.path.join(run_dir, "model_g2g.pt"))

    results_path = os.path.join(run_dir, "comparison.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "epoch", "train_loss", "train_UAS", "dev_loss", "dev_UAS"])
        for ep, tl, tu, dl, du in base_history:
            writer.writerow(["baseline", ep, tl, tu, dl, du])
        for ep, tl, tu, dl, du in g2g_history:
            writer.writerow(["g2g", ep, tl, tu, dl, du])

    n_params_base = sum(p_.numel() for p_ in baseline.parameters())
    n_params_g2g = sum(p_.numel() for p_ in g2g.parameters())
    summary = {
        "tag": args.tag, "seed": args.seed, "n_train": args.n_train, "n_dev": args.n_dev,
        "epochs": args.epochs, "n_params_baseline": n_params_base, "n_params_g2g": n_params_g2g,
        "final_baseline_dev_UAS": base_history[-1][4], "final_g2g_dev_UAS": g2g_history[-1][4],
        "training_time_s": time.time() - t0, "comparison_csv": results_path,
    }
    with open(os.path.join(run_dir, f"summary_{args.tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[{args.tag}] final dev UAS -- baseline: {base_history[-1][4]:.3f} | g2g: {g2g_history[-1][4]:.3f}")
    print(f"[{args.tag}] full per-epoch results written to {results_path}")
    return summary


def run_seed_study(args):
    seeds = [42 + i for i in range(args.n_seeds)]
    summaries = []
    for seed in seeds:
        run_args = argparse.Namespace(**{**vars(args), "seed": seed, "tag": f"seed{seed}"})
        summaries.append(run_experiment(run_args))

    base_uas = [s["final_baseline_dev_UAS"] for s in summaries]
    g2g_uas = [s["final_g2g_dev_UAS"] for s in summaries]
    b_mean, b_std = float(np.mean(base_uas)), float(np.std(base_uas))
    g_mean, g_std = float(np.mean(g2g_uas)), float(np.std(g2g_uas))

    agg_path = os.path.join(args.out_dir, "seed_study_summary.csv")
    with open(agg_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "baseline_dev_UAS", "g2g_dev_UAS"])
        for s in summaries:
            writer.writerow([s["seed"], s["final_baseline_dev_UAS"], s["final_g2g_dev_UAS"]])
        writer.writerow(["mean", b_mean, g_mean])
        writer.writerow(["std", b_std, g_std])

    print(f"\nfinal dev UAS across {len(seeds)} seeds {seeds}: "
          f"baseline {b_mean:.3f}+/-{b_std:.3f} | g2g {g_mean:.3f}+/-{g_std:.3f}")
    print(f"seed study written to {agg_path}")
    return summaries


def main():
    args = build_arg_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.n_seeds > 1:
        run_seed_study(args)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
