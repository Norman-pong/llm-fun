"""T5 实践：加载训好的模型玩采样——预训练闭环的「用」环节。

train.py 训完把权重存在 runs/（gitignore），本脚本读回来交互：
可指定 prompt 续写、扫温度看模型分布的变化、报 checkpoint 元信息。

    uv run python theory/05-pretrain/code/sample.py                          # 默认采样
    uv run python theory/05-pretrain/code/sample.py --prompt "ROMEO:"        # 续写
    uv run python theory/05-pretrain/code/sample.py --temperatures 0.5 0.8 1.2
    uv run python theory/05-pretrain/code/sample.py --run s300_d64_L2        # 指定小模型

依赖：先跑过一次训练（train.py 或 --smoke），runs/ 下有对应产物。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "04-gpt" / "code"))

from gpt import GPTModel
from train import RUNS, load_corpus


def latest_run() -> Path | None:
    """默认加载目标：所有 run 里 val 最优者（而非最新落盘——测试产物会污染 mtime）。"""
    if not RUNS.exists():
        return None
    dirs = [d for d in RUNS.iterdir() if (d / "model.pt").exists()]

    def best_val(d: Path) -> float:
        h = json.loads((d / "history.json").read_text(encoding="utf-8"))
        return min(v["loss"] for v in h["val"])

    return min(dirs, key=best_val) if dirs else None


def load_model(run_dir: Path) -> tuple[GPTModel, dict]:
    """按 history.json 里的 config 重建模型；优先载 val 最优权重 model_best.pt
    （旧 run 无此文件时回落 model.pt，且明确提示加载的是哪个）。"""
    hist = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    cfg = hist["config"]
    chars, _, _ = load_corpus()
    model = GPTModel(
        vocab_size=len(chars),
        emb_dim=cfg["emb_dim"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
        block_size=cfg["block_size"],
    )
    weights = "model_best.pt" if (run_dir / "model_best.pt").exists() else "model.pt"
    model.load_state_dict(
        torch.load(run_dir / weights, weights_only=True, map_location="cpu")
    )
    model.eval()
    return model, hist


def _weight_tag(run_dir: Path) -> str:
    return "best" if (run_dir / "model_best.pt").exists() else "final"


def encode_prompt(prompt: str, chars: list[str], stoi: dict[str, int]) -> torch.Tensor:
    """词表外的字符直接报错——续写入口不做静默替换。"""
    bad = sorted(set(prompt) - set(chars))
    if bad:
        msg = "、".join(map(repr, bad))
        raise SystemExit(f"prompt 含词表外字符 {msg}（65 字符集见 input.txt 词表）")
    return torch.tensor([[stoi[c] for c in prompt]])


def main() -> None:
    ap = argparse.ArgumentParser(description="加载 T5 训好的模型采样/续写")
    ap.add_argument(
        "--run", default=None, help="runs/05-pretrain 下的目录名，默认取最新"
    )
    ap.add_argument("--prompt", default="\n", help="续写提示（默认换行，即自由生成）")
    ap.add_argument("--tokens", type=int, default=400, help="生成 token 数")
    ap.add_argument("--temperature", type=float, default=0.8, help="单温度采样")
    ap.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=None,
        help="多温度对比采样（覆盖 --temperature）",
    )
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--list", action="store_true", help="只列出可用 run 与元信息")
    args = ap.parse_args()

    run_dir = RUNS / args.run if args.run else latest_run()
    if args.list or run_dir is None or not (run_dir / "model.pt").exists():
        if not args.list:
            hint = "先跑 uv run python theory/05-pretrain/code/train.py --smoke"
            print(f"找不到 checkpoint：{run_dir}\n{hint}")
        for d in sorted(RUNS.glob("*/model.pt")) if RUNS.exists() else []:
            h = json.loads((d.parent / "history.json").read_text(encoding="utf-8"))
            best = min(h["val"], key=lambda v: v["loss"])
            print(
                f"  {d.parent.name}: {h['n_params']:,} 参数 | "
                f"steps {h['config']['steps']} | val 最优 {best['loss']:.4f}@{best['step']}"
            )
        raise SystemExit(1 if not args.list else 0)

    model, hist = load_model(run_dir)
    cfg = hist["config"]
    chars, stoi, _ = load_corpus()
    best = min(hist["val"], key=lambda v: v["loss"])
    print(
        f"[{run_dir.name}] {hist['n_params']:,} 参数 | val 最优 {best['loss']:.4f}"
        f"@{best['step']}（最终 {hist['val'][-1]['loss']:.4f}）"
    )
    print(
        f"配置 {cfg['n_layer']}L·{cfg['emb_dim']}d·{cfg['n_head']}H·ctx{cfg['block_size']}"
        f" | 权重 {_weight_tag(run_dir)}"
    )
    idx = encode_prompt(args.prompt, chars, stoi)
    print(f"\n[prompt] {args.prompt!r}")
    for t in args.temperatures or [args.temperature]:
        out = model.generate(
            idx, args.tokens, temperature=t, top_k=args.top_k, seed=args.seed
        )
        text = "".join(chars[i] for i in out[0].tolist())
        gen = text[len(args.prompt) :]  # 只看生成部分
        print(f"\n[temperature={t}]")
        print(gen if gen.strip() else text)


if __name__ == "__main__":
    main()
