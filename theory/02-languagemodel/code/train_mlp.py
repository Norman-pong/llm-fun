"""T2 demo：bigram 基线 vs MLP，训练曲线、采样与诊断。

    uv run python theory/02-languagemodel/code/train_mlp.py
"""

from __future__ import annotations

from pathlib import Path

from languagemodel import (
    BLOCK,
    BigramModel,
    MLPModel,
    build_dataset,
    build_vocab,
    diagnostics,
    load_names,
    nll,
    sample_names,
    split,
    train_mlp,
)


def main() -> None:
    data = Path(__file__).resolve().parent.parent / "names.txt"
    words = load_names(data)
    stoi, itos = build_vocab(words)
    tr, dev, _ = split(words)
    print(f"词表大小: {len(itos)} | 训练 {len(tr)} / dev {len(dev)} 个名字")

    Xb, yb = build_dataset(tr, stoi, block=1)
    Xtr, ytr = build_dataset(tr, stoi, BLOCK)
    Xdev, ydev = build_dataset(dev, stoi, BLOCK)

    bigram = BigramModel(vocab_size=len(itos))
    bigram.fit(Xb, yb)
    print(f"\n[bigram 基线] train NLL {bigram.nll(Xb, yb):.4f} | dev NLL {bigram.nll(Xdev[:, -1:], ydev):.4f}")
    print("bigram 采样:", bigram.sample(8, itos, seed=1))

    model = MLPModel(vocab=len(itos))
    losses = train_mlp(model, Xtr, ytr, steps=3000, batch_size=128, lr=0.1)
    for i in range(0, len(losses), 500):
        chunk = losses[i : i + 500]
        print(f"steps {i:4d}-{i + len(chunk):4d}  mean loss {sum(chunk) / len(chunk):.4f}")
    print(f"[MLP] train NLL {nll(model, Xtr, ytr):.4f} | dev NLL {nll(model, Xdev, ydev):.4f}")
    print("MLP 采样:", sample_names(model, itos, n=8, seed=1))
    print("诊断:", {k: f"{v:.4f}" for k, v in diagnostics(model, Xtr).items()})


if __name__ == "__main__":
    main()
