"""
Paired analysis of the graph-to-graph vs. vanilla comparison.

`train.py --n_seeds N` reports each model's marginal mean and std. As in
an unpaired test discards the pairing: both models are trained on the same
data split with the same seed, so per-seed data/init effects are shared and
should be cancelled rather than left in the noise.

Also reports the train/dev gap, because in a data-starved regime the headline
UAS comparison can be tied while the two models differ in *how* they spend
capacity.

    python analyze_results.py --csv results/seed_study_summary.csv
"""
import argparse
import csv
import statistics as st


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="results/seed_study_summary.csv")
    args = p.parse_args()

    rows = [r for r in csv.DictReader(open(args.csv)) if r["seed"] not in ("mean", "std")]
    if len(rows) < 2:
        raise SystemExit("need >= 2 seeds; run train.py --n_seeds 3")

    base = [float(r["baseline_dev_UAS"]) for r in rows]
    g2g = [float(r["g2g_dev_UAS"]) for r in rows]
    n = len(rows)

    mb, sb = st.mean(base), st.stdev(base)
    mg, sg = st.mean(g2g), st.stdev(g2g)

    print(f"seeds: {n}\n")
    print("1) Marginal dev UAS")
    print(f"   vanilla        {mb:.3f} +/- {sb:.3f}")
    print(f"   graph-to-graph {mg:.3f} +/- {sg:.3f}")
    print(f"   seed variance ratio (g2g / vanilla): {sg/sb:.2f}x"
          if sb > 0 else "")

    print("\n2) Paired (same data split and seed for both models)")
    diffs = [g - b for g, b in zip(g2g, base)]
    md = st.mean(diffs)
    sd = st.stdev(diffs)
    for r, d in zip(rows, diffs):
        print(f"   seed {r['seed']}: {d:+.4f}")
    print(f"   mean {md:+.4f} +/- {sd:.4f}")
    if sd > 0:
        print(f"   paired t = {md/(sd/n**0.5):.2f}, df = {n-1}")
    print(f"   graph-to-graph ahead on {sum(d > 0 for d in diffs)}/{n} seeds")

    print("\n3) Verdict")
    if sd > 0 and abs(md) < sd / n ** 0.5:
        print("   No detectable difference. The paired mean is smaller than its own")
        print("   standard error, so this is a null result rather than a weak effect.")
    else:
        print("   A direction is present; check the t value and seed count before")
        print("   treating it as a finding.")


if __name__ == "__main__":
    main()
