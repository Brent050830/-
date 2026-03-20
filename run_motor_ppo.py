"""
run_motor_ppo.py - 训练入口
"""

import argparse
import os
import sys
import time

from motor_experiment import run_full_experiment, run_single_seed
from motor_env import load_efficiency_map


def main():
    parser = argparse.ArgumentParser(description="PPO 参考轨迹能量优化控制 - 训练与评估")
    parser.add_argument("--styles", nargs="+", default=["normal"],
                        choices=["eco", "normal", "sport"],
                        help="训练风格列表，默认只跑 normal")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="随机种子列表，默认只跑 seed 42")
    parser.add_argument("--preset", type=str, default="fast",
                        choices=["fast", "full"],
                        help="训练预设：fast 更快，full 保留完整训练量")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="输出结果目录")
    parser.add_argument("--eff-map", type=str, default="sys_eff_pivot.csv",
                        help="电机效率图 CSV 文件路径")
    parser.add_argument("--single", action="store_true",
                        help="只运行第一个 style 和第一个 seed")
    parser.add_argument("--quiet", action="store_true",
                        help="静默模式，减少输出")
    args = parser.parse_args()

    print("=" * 60)
    print("  PPO 参考轨迹能量优化控制系统")
    print("=" * 60)
    print(f"  风格:     {args.styles}")
    print(f"  Seeds:    {args.seeds}")
    print(f"  预设:     {args.preset}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  效率图:   {args.eff_map}")
    print(f"  模式:     {'单 seed' if args.single else '多 seed'}")
    print("=" * 60)

    if not os.path.exists(args.eff_map):
        print(f"[错误] 找不到效率图文件: {args.eff_map}")
        sys.exit(1)

    start_time = time.time()

    if args.single:
        eff_map = load_efficiency_map(args.eff_map)
        style = args.styles[0]
        seed = args.seeds[0]
        result = run_single_seed(
            style=style,
            seed=seed,
            eff_map=eff_map,
            output_dir=args.output_dir,
            preset=args.preset,
            verbose=not args.quiet,
        )
        print(f"\n单 seed 完成: style={style}  seed={seed}")
        print(f"  eval_saving_total = {result['eval_metrics'].get('saving_total_pct', 0):.2f}%")
        print(f"  eval_speed_mae    = {result['eval_metrics'].get('speed_mae', 0):.4f} m/s")
        print(f"  tracking_ok       = {result['eval_metrics'].get('tracking_ok', False)}")
    else:
        run_full_experiment(
            styles=args.styles,
            seeds=args.seeds,
            output_dir=args.output_dir,
            eff_map_path=args.eff_map,
            preset=args.preset,
            verbose=not args.quiet,
        )

    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.1f} 秒 ({elapsed / 60:.1f} 分钟)")
    print(f"结果已保存到: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
