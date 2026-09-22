"""T5 训练曲线可视化：把 runs/ 落盘的 history.json 画成 PNG。

一条命令看懂小语料过拟合：train 一路下降、val 在中途拐头回升——
「早停点即 val 最优点」这件事画出来比读 JSON 数字直观得多。

    uv run python theory/05-pretrain/code/plot.py                     # 默认 run
    uv run python theory/05-pretrain/code/plot.py --run s3000_d192_L6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无头环境：不弹窗，只落盘
import matplotlib.pyplot as plt

RUNS = Path(__file__).resolve().parents[3] / "runs" / "05-pretrain"


def default_run() -> Path | None:
    """默认 run：val 最优者（与 sample.py latest_run 一致，不被测试产物污染）。"""
    dirs = [d for d in RUNS.iterdir() if (d / "history.json").exists()]

    def best_val(d: Path) -> float:
        h = json.loads((d / "history.json").read_text(encoding="utf-8"))
        return min(v["loss"] for v in h["val"])

    return min(dirs, key=best_val) if dirs else None


def plot_run(run_dir: Path) -> Path:
    hist = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    cfg = hist["config"]
    steps = range(1, len(hist["train"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(steps, hist["train"], lw=0.6, alpha=0.4, label="train (per-step)")
    # train 按 val 评估窗口平滑，与 val 采样点对齐才可比
    w = cfg["steps"] // len(hist["val"])
    smooth = [
        sum(hist["train"][i * w : (i + 1) * w]) / w for i in range(len(hist["val"]))
    ]
    val_steps = [v["step"] for v in hist["val"]]
    ax1.plot(val_steps, smooth, lw=1.5, label=f"train ({w}-step mean)")
    ax1.plot(val_steps, [v["loss"] for v in hist["val"]], "o-", lw=1.5, label="val")
    best = min(hist["val"], key=lambda v: v["loss"])
    ax1.axvline(best["step"], color="gray", ls=":", lw=1)
    ax1.annotate(
        f"best val {best['loss']:.3f}\n@step {best['step']}",
        xy=(best["step"], best["loss"]),
        xytext=(10, 30),
        textcoords="offset points",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "lw": 0.8},
    )
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    ax1.set_title(f"{run_dir.name}  ({hist['n_params']:,} params)")
    ax1.legend()

    ax2.plot(steps, hist["lr"], lw=1.2)
    ax2.set_xlabel("step")
    ax2.set_ylabel("lr")
    ax2.set_title("LR schedule (warmup + cosine)")

    fig.tight_layout()
    target = run_dir / "history.png"
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target


def main() -> None:
    ap = argparse.ArgumentParser(description="把 history.json 画成训练曲线 PNG")
    ap.add_argument(
        "--run", default=None, help="runs/05-pretrain 下的目录名，默认取 val 最优"
    )
    args = ap.parse_args()

    run_dir = RUNS / args.run if args.run else default_run()
    if run_dir is None or not (run_dir / "history.json").exists():
        raise SystemExit(f"找不到 {run_dir / 'history.json'}，先跑一次训练")

    print(f"已保存 {plot_run(run_dir)}")


if __name__ == "__main__":
    main()
