"""
Loads a trained baseline or graph-to-graph checkpoint and reports parameter
count plus a freshly-recomputed dev UAS. Mirrors evaluate.py in the other
three projects.

Usage:
    python evaluate.py --ckpt results/baseline/model_g2g.pt --n_dev 200
    python evaluate.py --summary results/baseline/summary_baseline.json   # no recompute
"""
import argparse
import json

import torch
from torch.utils.data import DataLoader

from data import load_data, MAX_LEN
from model import GraphToGraphTransformer, VanillaArcPredictor
from train import collate, run_epoch


def print_summary(d: dict):
    print("```")
    print(f"Model:        {d.get('kind', '?')}")
    if isinstance(d.get("n_params"), int):
        print(f"Parameters:   {d['n_params']:,}")
    print(f"Seed:         {d.get('seed', '?')}")
    print(f"Train sents:  {d.get('n_train', '?')}")
    print(f"Dev sents:    {d.get('n_dev', '?')}")
    print(f"Epochs:       {d.get('epochs', '?')}")
    if "dev_UAS" in d:
        print(f"Dev UAS:      {d['dev_UAS']:.3f}")
    print("```")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--summary", type=str, default=None, help="print a saved summary_*.json instead of recomputing")
    p.add_argument("--n_dev", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    args = p.parse_args()

    if args.summary:
        with open(args.summary) as f:
            s = json.load(f)
        for kind in ("baseline", "g2g"):
            print(f"\n--- {kind} (from summary) ---")
            print_summary({
                "kind": kind, "n_params": s.get(f"n_params_{kind}"), "seed": s["seed"],
                "n_train": s["n_train"], "n_dev": s["n_dev"], "epochs": s["epochs"],
                "dev_UAS": s[f"final_{kind}_dev_UAS"],
            })
        return

    if not args.ckpt:
        raise SystemExit("provide either --ckpt or --summary")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    kind = ckpt["kind"]
    margs = ckpt["args"]
    vocab_size = ckpt["vocab_size"]

    # IMPORTANT: vocab is built from the first n_train training sentences
    # (see data.py's load_data), so it must be rebuilt with the SAME n_train
    # the checkpoint was trained with, or token ids will desync from what
    # the checkpoint's embedding table actually learned. n_dev can differ
    # freely since dev sentences aren't used to build the vocab.
    _, dev_ds, _ = load_data(n_train=margs.get("n_train", 800), n_dev=args.n_dev, max_len=MAX_LEN)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    if kind == "baseline":
        model = VanillaArcPredictor(vocab_size, margs["d_model"], margs["n_heads"], margs["n_layers"], MAX_LEN).to(device)
    else:
        model = GraphToGraphTransformer(vocab_size, margs["d_model"], margs["n_heads"], margs["n_layers"], MAX_LEN).to(device)
    model.load_state_dict(ckpt["model_state"])

    n_params = sum(p_.numel() for p_ in model.parameters())
    _, dev_uas = run_epoch(model, dev_loader, device, optimizer=None)

    print_summary({
        "kind": kind, "n_params": n_params, "seed": margs.get("seed"),
        "n_train": margs.get("n_train"), "n_dev": args.n_dev, "epochs": margs.get("epochs"),
        "dev_UAS": dev_uas,
    })


if __name__ == "__main__":
    main()
