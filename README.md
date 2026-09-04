# Graph-to-Graph Transformer — Relational Structure

This project constrains attention with an **explicit graph**, and — the part
that makes it more than a static bias term — lets the model *rewrite that
graph* at every layer, re-estimating it from its own representations as depth
increases.

**Central question:** is a Transformer better off being handed a wrong
structural hypothesis it can correct, than being handed no structure at all?

## Scope and honesty — read this first

A competitive graph-conditioned parser would use **labelled** relational
edges, sit inside a real parsing algorithm (transition-based or graph-based),
and train on a full treebank with a subword vocabulary.

This implementation simplifies on every one of those axes:

| A competitive version would use | This uses |
|---|---|
| Labelled edges with relation-type embeddings | A single dense unlabelled arc-probability matrix |
| Full transition-based / graph-based parser | Direct unlabelled arc prediction |
| Full treebank, subword vocabulary | ~800 sentences, word-level vocabulary |
| Competitive UAS/LAS | Proof-of-concept comparison against a matched baseline |

It is a mechanism study, not a parser. Anyone evaluating it as a parser will
find it uncompetitive, and correctly so.

## The mechanism

**Input:** a sentence plus a deliberately naive initial graph — every token's
head guessed as the immediately preceding token (a linear chain). This is wrong
for the large majority of real dependency arcs, which is the point: the model
must *correct* structure, not merely consume it.

**Each layer:**

1. Attention logits are biased by the current graph estimate:
   `att = QKᵀ/√d + λ · (G + Gᵀ)`, symmetrized so both head→dependent and
   dependent→head attention are informed. `λ` is learned.
2. After the attention and FFN sublayers, the layer **re-estimates** the graph
   from its own updated representations via a bilinear head:
   `G' = (W_h x)(W_d x)ᵀ / √d`.
3. `sigmoid(G')` becomes the bias for the next layer.

So the graph is not a fixed input feature. It is a hypothesis that gets revised
layer by layer, and the final layer's estimate is the prediction. This
iterative-refinement structure — rewriting the graph rather than only
conditioning on it — is what distinguishes this from a model that just adds a
static graph feature once at the input.

## Baseline

`VanillaArcPredictor` — a plain Transformer encoder that **never sees the graph
at all** and predicts arcs once from final token representations. Same
parameter budget, same objective, same optimizer, same data. The initial graph
is the only input it lacks, so the comparison isolates graph conditioning
rather than capacity.

Both are trained with cross-entropy over head choice per dependent token.

## Data — real, not synthetic

**Universal Dependencies English-EWT**, fetched directly from the
[UD GitHub organisation](https://github.com/UniversalDependencies/UD_English-EWT) —
real annotated linguistic data rather than a synthetic or toy corpus.

Metric: **unlabelled attachment score (UAS)** — the fraction of non-root tokens
assigned their correct head.

## Running it

```bash
pip install -r requirements.txt

python train.py --n_train 800 --n_dev 200 --epochs 15 --d_model 64 --n_layers 4 --tag baseline
python train.py --n_train 800 --n_dev 200 --epochs 15 --n_seeds 3   # with uncertainty

python evaluate.py --summary results/baseline/summary_baseline.json
python plot_results.py --comparison_csv results/baseline/comparison.csv --out results/uas.png
python plot_results.py --seed_study_csv results/seed_study_summary.csv --out results/seeds.png
```

Verified end-to-end on CPU at reduced scale. `--n_train 800` runs in minutes;
the full EWT train split has ~12k sentences.

### A vocabulary bug worth knowing about

The token vocabulary is built from the first `n_train` training sentences. An
early version of `evaluate.py` called `load_data(n_train=1, ...)` on the
reasoning that only the dev set was needed — which rebuilt the vocabulary from
one sentence, sent essentially every token to `<unk>`, and produced a
near-random UAS **with no error raised.**

It was caught by a check that should be standard: re-evaluating a checkpoint
must reproduce its training-time metric. After the fix (rebuild the vocabulary
using the checkpoint's own recorded `n_train`), `evaluate.py --ckpt` reproduces
the training-time UAS exactly — 0.026 vs 0.026 on the verification run.

This is a general class of failure worth naming: a silently wrong experiment
that runs cleanly to completion, with no exception and a plausible-looking
number at the end, is far more dangerous than a crash.

## Files

```
data.py          # downloads UD EWT; builds (tokens, gold_arcs, naive_init_arcs)
model.py         # GraphToGraphTransformer + VanillaArcPredictor
train.py         # run_experiment(); run_seed_study() for the multi-seed version
evaluate.py      # rebuilds the matching vocabulary from the checkpoint's n_train
plot_results.py    # per-epoch UAS/loss curves, or cross-seed error bars
analyze_results.py # paired statistical analysis of the seed study
make_figures.py    # regenerates the control figure above
```

## Results

Three conditions differing **only** in the initial graph handed to the
graph-to-graph model:

- `chain` — every token's head guessed as the previous token (informative-ish;
  on a sample sentence 2 of 28 arcs are correct)
- `random` — same arc count and one-head-per-dependent structure, heads drawn
  uniformly (graph-shaped, **uninformative**; 0 of 28 correct)
- `empty` — no graph at all

### A retraction first

This experiment was first run at **3 seeds** and written up here as a finding:
that `random` beat `chain` (+0.018, ahead on 3/3 seeds), implying the model
never reads the graph's structural content.

**Six seeds killed it.** Adding seeds 45–47 flipped the sign:

| condition | paired Δ, 3 seeds | paired Δ, 6 seeds | |
|---|---|---|---|
| `chain` | +0.000 | −0.007 | |
| `random` | **+0.018** (3/3 ahead) | **−0.004** (3/6 ahead) | **sign flip** |
| `empty` | −0.012 | −0.014 | |

The three-seed result was noise, and it was noise that happened to look like a
clean, publishable mechanism claim. It is left visible here rather than
silently overwritten, because it is a concrete demonstration of how easily a
three-seed effect can reverse under more scrutiny — and here it happened to a
conclusion that had already been written down as a result, not just a
tentative first look.

![initial graph control](results/fig_init_graph.png)

*All six bars overlap within their error bars. Right panel: per-seed paired
differences, black bars are means — every condition sits at or below zero, with
seeds on both sides. In the 3-seed version, `random` had all three points above
zero.*

### The 6-seed numbers

| initial graph | vanilla baseline | graph-to-graph | paired Δ | *t* (df=5) | seeds ahead |
|---|---|---|---|---|---|
| `chain` | 0.385 ± 0.015 | 0.379 ± 0.019 | −0.007 ± 0.026 | −0.61 | 3/6 |
| `random` | 0.388 ± 0.015 | 0.383 ± 0.017 | −0.004 ± 0.027 | −0.40 | 3/6 |
| `empty` | 0.385 ± 0.015 | 0.371 ± 0.020 | −0.014 ± 0.027 | −1.30 | 2/6 |

**Graph conditioning does not beat the vanilla baseline in any condition.** All
three point estimates are negative, all are well inside noise, and none reaches
even 4/6 seeds. At 800 sentences, the architecture buys nothing.

### The one effect that survives

Comparing conditions *within* seed, the only consistent signal is between
having a graph and having none:

| comparison | paired Δ | seeds ahead |
|---|---|---|
| **`chain` − `empty`** | **+0.008 ± 0.008** (*t* = 2.22) | **5/6** |
| `random` − `empty` | +0.013 ± 0.033 | 3/6 |
| `random` − `chain` | +0.005 ± 0.028 | 3/6 |

`chain` beats `empty` on 5 of 6 seeds with by far the tightest spread of any
comparison here. Supplying *some* graph tensor helps a little, consistently.

But `random` − `chain` is 3/6 and its spread is four times larger than the
`chain`−`empty` effect. So the earlier claim — that the graph's *content* is
irrelevant — is **not supported and not refuted**; the experiment lacks the
power to separate them. What it does show is that the content effect, if any,
is smaller than the effect of the graph tensor merely being present.

### Two caveats on these numbers

**The RNG fix did not take effect in this run.** `chain` and `empty` produce
bit-identical baselines across all 6 seeds, as they must. `random` does not
(0.380/0.358/0.389/0.397/0.403/0.400 vs 0.385/0.354/0.386/0.390/0.392/0.403) —
which is the signature of the global-RNG bug described below, meaning this run
used the pre-fix code. The `random` arm therefore remains slightly mismatched.
Given that its effect is now indistinguishable from zero, this is unlikely to
matter, but the arm should be re-run before the numbers are relied on.

**Overfitting is unchanged and severe.** Train UAS ~0.63 against dev UAS ~0.38
in every condition, with dev loss rising over the last third of training. The
models are memorising 800 sentences.

### A methodological flaw in the experiment design

The random graph was drawn from the **global** RNG during dataset construction,
which advanced the stream before model initialisation — so `random`-condition
models were initialised differently from the others' at the same `--seed`.
`data.py` now uses a dedicated `torch.Generator` seeded per example; verified
that all three modes leave the global RNG identical. The results above predate
that fix.

## Path to a publishable result

An earlier version of this section claimed the project had found a mechanism
result — that graph conditioning is an attention bias rather than a structural
prior. Six seeds withdrew the evidence for that. What is left is weaker and
more honest:

**At 800 sentences, graph-to-graph conditioning does not beat a matched vanilla
Transformer, and the only consistent effect is that having a graph tensor beats
having none (5/6 seeds, +0.008) — an effect small enough to be plausibly about
parameters rather than structure.**

That is not a paper. It is one point on a curve, and the curve is the paper:

> **At what data scale does an explicit structural prior start to pay off, and
> is the payoff about the structure or about the extra parameters?**

The `--init_graph {chain,random,empty}` control is the right instrument for the
second half of that question — it separates "structure" from "a graph-shaped
tensor" — but it needs to be run where the architecture actually works, not
where nothing works.

What is missing, in order:

1. **Scale until the architecture does something.** 800 → 2k → 5k → 12k (full
   EWT). Every conclusion here is from a regime where both models overfit hard
   and neither beats the other; testing a mechanism there is testing noise.
   Re-run the three-condition control **at each scale**.
2. **Seeds, and more of them than feels necessary.** Three seeds produced a
   clean sign-flipped illusion in this very project. Six is the minimum here;
   the `random`−`chain` spread (±0.028) implies ~25 seeds to resolve a 0.008
   effect at conventional power.
3. **Re-run `random` post-fix**, so the arm is properly matched.
4. **A graded initial graph.** Rather than chain/random/empty, take gold arcs
   and corrupt a controlled fraction (0% → 100%). That traces the whole
   structure-quality curve instead of sampling three arbitrary points on it,
   and it is the version that would actually answer the question.
5. **Labelled arcs and LAS**, to connect to the literature's metric.

Item 4 combined with item 1 is the contribution. Everything else is cost.

The realistic framing: this project currently contributes a **negative result
at small scale plus a validated control instrument**, and a demonstration that
its own first analysis was underpowered. That is honest and useful, and it is
not yet a paper.
# graph-conditioned-attention
