"""01b · 生成 schema 一致的合成数据，用于在没有真实数据时做端到端冒烟验证。

为什么需要它：在拿到赛题数据之前，代码是**完全未经验证**的。合成数据有**已知的
生成过程**，因此可以在上面验证「打乱标签后 AUC 必须回到 0.5」这类断言确实成立 ——
这在真实数据上无法预先知道答案。

> ⚠️ 合成数据上的任何数字都不代表比赛结果，只能说明代码能跑、协议自洽。

用法::

    python scripts/01b_make_synthetic.py --out data/raw/synthetic --vehicles 120
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import get_config, setup_logging

from ie_safety.synthetic import SyntheticSpec, generate


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="生成合成赛题数据（schema 一致，内容为伪造）")
    ap.add_argument("--out", default=None, help="输出目录，默认 <data/raw>/synthetic")
    ap.add_argument("--vehicles", type=int, default=120)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    setup_logging()

    cfg = get_config()
    out = Path(args.out) if args.out else (cfg.resolve("raw", ensure=True) / "synthetic")

    spec = SyntheticSpec(n_vehicles=args.vehicles, n_days=args.days, seed=args.seed)
    paths = generate(spec, out)

    print("合成数据已生成：")
    for k, v in paths.items():
        print(f"  {k:11s} {v}")
    print(
        "\n⚠️  这是合成数据，任何在其上得到的指标都不代表比赛结果。\n"
        "   它的用途是验证代码链路与验证协议自洽。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
