"""Figure for the initial-graph control experiment.

    python make_figures.py
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statistics as st

CONDS = ["chain", "random", "empty"]
LABEL = {"chain": "chain\n(informative-ish)", "random": "random\n(uninformative)", "empty": "empty\n(no graph)"}


def read(cond):
    rows = [r for r in csv.DictReader(open(f"results_{cond}/seed_study_summary.csv"))
            if r["seed"] not in ("mean", "std")]
    return ([float(r["baseline_dev_UAS"]) for r in rows],
            [float(r["g2g_dev_UAS"]) for r in rows])


def main():
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    x = range(len(CONDS))

    bm, bs, gm, gs, paired = [], [], [], [], []
    for c in CONDS:
        b, g = read(c)
        bm.append(st.mean(b)); bs.append(st.stdev(b))
        gm.append(st.mean(g)); gs.append(st.stdev(g))
        paired.append([gi - bi for gi, bi in zip(g, b)])

    w = 0.35
    axs[0].bar([i - w/2 for i in x], bm, w, yerr=bs, capsize=4, color="#888", label="vanilla (no graph input)")
    axs[0].bar([i + w/2 for i in x], gm, w, yerr=gs, capsize=4, color="#4c72b0", label="graph-to-graph")
    axs[0].set_xticks(list(x)); axs[0].set_xticklabels([LABEL[c] for c in CONDS], fontsize=9)
    axs[0].set_ylabel("dev UAS (mean ± std, 6 seeds)")
    axs[0].set_ylim(0.30, 0.44)
    axs[0].set_title("All three conditions overlap")
    axs[0].legend(fontsize=8); axs[0].grid(alpha=0.3, axis="y")

    for i, c in enumerate(CONDS):
        for d in paired[i]:
            axs[1].plot(i, d, "o", color="#c44e52", alpha=0.6)
        axs[1].plot(i, st.mean(paired[i]), "_", markersize=28, color="black")
    axs[1].axhline(0, ls="--", color="grey", lw=1)
    axs[1].set_xticks(list(x)); axs[1].set_xticklabels([LABEL[c] for c in CONDS], fontsize=9)
    axs[1].set_ylabel("paired (g2g − vanilla) per seed")
    axs[1].set_title("Every condition: g2g at or below baseline")
    axs[1].grid(alpha=0.3, axis="y")

    fig.suptitle("Initial-graph control, 6 seeds: graph conditioning does not beat the "
                 "baseline in any condition", fontsize=11)
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/fig_init_graph.png", dpi=150)
    print("saved results/fig_init_graph.png")


if __name__ == "__main__":
    main()
