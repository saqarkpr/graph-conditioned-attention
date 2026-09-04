"""
Loads a subset of the Universal Dependencies English-EWT treebank and turns
it into an unlabeled dependency-arc prediction dataset: given a sentence and
an initial "guess" graph (a cheap linear-chain / adjacent-word heuristic),
predict for every ordered word pair whether a real dependency arc exists —
i.e. reconstruct the true dependency graph from a naive initial one. This is
a simplified stand-in for a "graph-to-graph" setting: refine an initial graph
into a better one using graph-conditioned attention, rather than predict a
parse from raw tokens with no graph prior.

Only a few hundred sentences are used by default — enough to demonstrate and
compare architectures on CPU, not to reach state-of-the-art parsing accuracy
(that would need the full treebank, subword handling, and far more compute).
"""
import os
import urllib.request
import torch

TRAIN_URL = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu"
DEV_URL = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-dev.conllu"

DATA_DIR = os.path.join(os.path.dirname(__file__), "ud_data")
MAX_LEN = 30  # keep sentences short so full dense arc matrices stay small


def _download(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"downloading {url} ...")
        urllib.request.urlretrieve(url, path)


def parse_conllu(path, max_sentences=None, max_len=MAX_LEN):
    """Yields (tokens: List[str], heads: List[int]) per sentence.
    heads[i] = index (1-based, 0 = root) of the head of token i."""
    sentences = []
    tokens, heads = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens and len(tokens) <= max_len:
                    sentences.append((tokens, heads))
                tokens, heads = [], []
                if max_sentences and len(sentences) >= max_sentences:
                    break
                continue
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if "-" in fields[0] or "." in fields[0]:
                continue  # skip multiword tokens / empty nodes
            tokens.append(fields[1].lower())
            heads.append(int(fields[6]))
    return sentences


class DepParsingVocab:
    def __init__(self, sentences, min_freq=1):
        from collections import Counter
        counts = Counter(tok for toks, _ in sentences for tok in toks)
        self.itos = ["<pad>", "<unk>"] + [w for w, c in counts.items() if c >= min_freq]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, tokens):
        return [self.stoi.get(t, 1) for t in tokens]

    @property
    def vocab_size(self):
        return len(self.itos)


def sentence_to_example(tokens, heads, vocab, max_len=MAX_LEN, init_graph="chain",
                         rand_seed=0):
    """Builds:
      ids: (max_len,) token ids, padded with 0
      pad_mask: (max_len,) bool
      gold_arcs: (max_len, max_len) float 0/1, gold_arcs[h, d] = 1 if h is head of d
      init_arcs: (max_len, max_len) float 0/1, a naive LINEAR-CHAIN initial graph
                 (token i's head guessed as token i-1) — this is the "initial graph"
                 that the graph-to-graph model refines into gold_arcs.
    """
    n = len(tokens)
    ids = torch.zeros(max_len, dtype=torch.long)
    ids[:n] = torch.tensor(vocab.encode(tokens), dtype=torch.long)
    pad_mask = torch.zeros(max_len, dtype=torch.bool)
    pad_mask[:n] = True

    gold_arcs = torch.zeros(max_len, max_len)
    for dep_idx, head in enumerate(heads):  # heads are 1-based, 0 = root
        if head == 0:
            continue  # root has no head token; skip (no arc into root)
        gold_arcs[head - 1, dep_idx] = 1.0

    # The initial graph handed to the model. Three modes exist so that
    # "does graph conditioning help?" can be separated from "does the model
    # even read the graph?" -- a model that scores identically on `chain`,
    # `random` and `empty` is ignoring the input entirely, which is a
    # different failure from having too little data to exploit it.
    init_arcs = torch.zeros(max_len, max_len)
    if init_graph == "chain":
        for dep_idx in range(1, n):
            init_arcs[dep_idx - 1, dep_idx] = 1.0     # previous word is the head
    elif init_graph == "random":
        # same arc COUNT as chain, same one-head-per-dependent structure,
        # but heads chosen uniformly at random -- isolates "structure" from
        # "a graph-shaped tensor of the right density".
        #
        # A DEDICATED generator is used rather than the global RNG. Drawing
        # from the global stream here would advance it during dataset
        # construction, so the model initialisation that follows would differ
        # between `random` and `chain`/`empty` runs at the same --seed, and the
        # conditions would no longer be matched. (The first version of this did
        # exactly that: chain and empty produced bit-identical baselines while
        # random did not.)
        gen = torch.Generator().manual_seed(rand_seed)
        for dep_idx in range(1, n):
            head = torch.randint(0, n, (1,), generator=gen).item()
            init_arcs[head, dep_idx] = 1.0
    elif init_graph == "empty":
        pass                                           # all-zero: no graph at all
    else:
        raise ValueError(f"unknown init_graph mode: {init_graph!r}")

    return ids, pad_mask, gold_arcs, init_arcs


class DepParsingDataset(torch.utils.data.Dataset):
    def __init__(self, sentences, vocab, max_len=MAX_LEN, init_graph="chain"):
        # rand_seed derived from the example index, so the random graph is
        # deterministic per sentence and independent of the global RNG
        self.examples = [sentence_to_example(t, h, vocab, max_len, init_graph, i)
                          for i, (t, h) in enumerate(sentences)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def load_data(n_train=800, n_dev=200, max_len=MAX_LEN, init_graph="chain"):
    train_path = os.path.join(DATA_DIR, "train.conllu")
    dev_path = os.path.join(DATA_DIR, "dev.conllu")
    _download(TRAIN_URL, train_path)
    _download(DEV_URL, dev_path)

    train_sents = parse_conllu(train_path, max_sentences=n_train, max_len=max_len)
    dev_sents = parse_conllu(dev_path, max_sentences=n_dev, max_len=max_len)

    vocab = DepParsingVocab(train_sents)
    train_ds = DepParsingDataset(train_sents, vocab, max_len, init_graph)
    dev_ds = DepParsingDataset(dev_sents, vocab, max_len, init_graph)
    return train_ds, dev_ds, vocab


if __name__ == "__main__":
    train_ds, dev_ds, vocab = load_data(n_train=50, n_dev=20)
    print(f"train sentences: {len(train_ds)} | dev: {len(dev_ds)} | vocab: {vocab.vocab_size}")
    ids, pad_mask, gold_arcs, init_arcs = train_ds[0]
    print("ids:", ids[:10])
    print("n real tokens:", pad_mask.sum().item())
    print("gold arcs (nonzero):", gold_arcs.nonzero().shape[0])
    print("init arcs (nonzero, should == n_real_tokens-1):", init_arcs.nonzero().shape[0])
