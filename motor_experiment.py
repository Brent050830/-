"""
motor_experiment.py  ——  实验调度模块
=====================================
职责：
  - 构造训练道路 / 评估道路 / stress test 道路
  - 多 seed 训练
  - style-specific 配置
  - 选择 best seed
  - 导出图像、json、模型文件
"""

import os, json, time, copy
import numpy as np
from typing import Dict, List, Optional

from motor_env import (MotorEnv, generate_road, generate_reference_trajectory,
                       load_efficiency_map, STYLE_PROFILES)
from motor_agent import PPOAgent
from motor_training import (run_multistage_training, eval_episode,
                            load_best_checkpoint, LagrangianManager,
                            score_energy_metrics_for_selection)
from motor_plotting import plot_all

# ============================================================
#  道路集合生成
# ============================================================
def build_road_sets(eff_map: dict,
                    train_dist: float = 3000.0,
                    eval_dist: float = 2000.0,
                    stress_dist: float = 1500.0,
                    style: str = "normal",
                    seed_base: int = 0) -> dict:
    """
    构造训练 / 评估 / stress 三套道路和参考轨迹
    """
    train_road = generate_road(total_dist=train_dist, seed=seed_base + 100)
    eval_road  = generate_road(total_dist=eval_dist,  seed=seed_base + 200)
    stress_road = generate_road(total_dist=stress_dist, seed=seed_base + 300,
                                grade_sigma=0.06, curve_prob=0.25,
                                curve_speed_range=(5.0, 18.0))

    train_ref  = generate_reference_trajectory(train_road,  style=style, eff_map=eff_map)
    eval_ref   = generate_reference_trajectory(eval_road,   style=style, eff_map=eff_map)
    stress_ref = generate_reference_trajectory(stress_road, style=style, eff_map=eff_map)

    return {
        "train_road":  train_road,  "train_ref":  train_ref,
        "eval_road":   eval_road,   "eval_ref":   eval_ref,
        "stress_road": stress_road, "stress_ref": stress_ref,
    }


# ============================================================
#  风格专属训练配置
# ============================================================
STYLE_TRAIN_CFG = {
    "eco": {
        "track_episodes":       150,
        "energy_episodes":      300,
        "polish_episodes":      60,
        "energy_weight_final":  6.0,
        "residual_torque_final": 10.0,
        "lr_actor":             3e-4,
        "lr_critic":            3e-4,
    },
    "normal": {
        "track_episodes":       120,
        "energy_episodes":      250,
        "polish_episodes":      50,
        "energy_weight_final":  5.0,
        "residual_torque_final": 10.0,
        "lr_actor":             3e-4,
        "lr_critic":            3e-4,
    },
    "sport": {
        "track_episodes":       180,
        "energy_episodes":      350,
        "polish_episodes":      80,
        "energy_weight_final":  4.0,
        "residual_torque_final": 10.0,
        "lr_actor":             2e-4,
        "lr_critic":            2e-4,
    },
}

TRAIN_PRESETS = {
    "fast": {
        "road": {
            "train_dist": 1200.0,
            "eval_dist": 1000.0,
            "stress_dist": 800.0,
        },
        "agent": {
            "hidden_dims": [128, 128],
            "ppo_epochs": 4,
            "mini_batch_size": 256,
        },
        "eval_interval": {
            "track": 10,
            "energy": 10,
            "polish": 5,
        },
        "styles": {
            "eco": {
                "track_episodes":       40,
                "energy_episodes":      80,
                "polish_episodes":      20,
                "energy_weight_final":  6.0,
                "residual_torque_final": 10.0,
                "lr_actor":             3e-4,
                "lr_critic":            3e-4,
            },
            "normal": {
                "track_episodes":       35,
                "energy_episodes":      70,
                "polish_episodes":      15,
                "energy_weight_final":  5.0,
                "residual_torque_final": 10.0,
                "lr_actor":             3e-4,
                "lr_critic":            3e-4,
            },
            "sport": {
                "track_episodes":       50,
                "energy_episodes":      90,
                "polish_episodes":      20,
                "energy_weight_final":  4.0,
                "residual_torque_final": 10.0,
                "lr_actor":             2e-4,
                "lr_critic":            2e-4,
            },
        },
    },
    "full": {
        "road": {
            "train_dist": 3000.0,
            "eval_dist": 2000.0,
            "stress_dist": 1500.0,
        },
        "agent": {
            "hidden_dims": [256, 256],
            "ppo_epochs": 10,
            "mini_batch_size": 256,
        },
        "eval_interval": {
            "track": 20,
            "energy": 20,
            "polish": 10,
        },
        "styles": copy.deepcopy(STYLE_TRAIN_CFG),
    },
}


def resolve_style_train_cfg(style: str, preset: str = "fast") -> dict:
    preset_cfg = TRAIN_PRESETS[preset]
    cfg = copy.deepcopy(preset_cfg["styles"][style])
    cfg["road"] = copy.deepcopy(preset_cfg["road"])
    cfg["agent"] = copy.deepcopy(preset_cfg["agent"])
    cfg["eval_interval"] = copy.deepcopy(preset_cfg["eval_interval"])
    return cfg


# ============================================================
#  单个 seed 实验
# ============================================================
def run_single_seed(style: str,
                    seed: int,
                    eff_map: dict,
                    output_dir: str = "results",
                    preset: str = "fast",
                    verbose: bool = True) -> dict:
    """
    运行单个 seed 的完整多阶段训练 + 评估
    返回 {'history': ..., 'eval_metrics': ..., 'stress_metrics': ..., 'seed': ...}
    """
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)

    cfg = resolve_style_train_cfg(style, preset)
    run_dir = os.path.join(output_dir, style, f"seed_{seed}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    if verbose:
        print(f"\n{'#'*60}")
        print(f"  Experiment: style={style}  seed={seed}  preset={preset}")
        print(f"{'#'*60}")

    # 构建道路集
    road_sets = build_road_sets(
        eff_map,
        train_dist=cfg["road"]["train_dist"],
        eval_dist=cfg["road"]["eval_dist"],
        stress_dist=cfg["road"]["stress_dist"],
        style=style,
        seed_base=seed,
    )

    # 创建 Agent
    # 先临时创建 env 获取 obs_dim
    tmp_env = MotorEnv(road=road_sets["train_road"], ref=road_sets["train_ref"],
                       style=style, mode="track", eff_map=eff_map)
    obs_dim = tmp_env.OBS_DIM

    agent = PPOAgent(
        obs_dim=obs_dim,
        act_dim=tmp_env.ACT_DIM,
        hidden_dims=cfg["agent"]["hidden_dims"],
        lr_actor=cfg["lr_actor"],
        lr_critic=cfg["lr_critic"],
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        ppo_epochs=cfg["agent"]["ppo_epochs"],
        mini_batch_size=cfg["agent"]["mini_batch_size"],
    )

    # 多阶段训练
    history = run_multistage_training(
        agent=agent,
        train_road=road_sets["train_road"],
        train_ref=road_sets["train_ref"],
        eval_road=road_sets["eval_road"],
        eval_ref=road_sets["eval_ref"],
        eff_map=eff_map,
        style=style,
        track_episodes=cfg["track_episodes"],
        energy_episodes=cfg["energy_episodes"],
        polish_episodes=cfg["polish_episodes"],
        energy_weight_final=cfg["energy_weight_final"],
        residual_torque_final=cfg["residual_torque_final"],
        track_eval_interval=cfg["eval_interval"]["track"],
        energy_eval_interval=cfg["eval_interval"]["energy"],
        polish_eval_interval=cfg["eval_interval"]["polish"],
        checkpoint_dir=ckpt_dir,
        verbose=verbose,
    )

    # 加载 best energy model 进行最终评估
    if not load_best_checkpoint(agent, ckpt_dir, "polish", style):
        load_best_checkpoint(agent, ckpt_dir, "energy", style)

    # 评估
    eval_env = MotorEnv(road=road_sets["eval_road"], ref=road_sets["eval_ref"],
                        style=style, mode="energy", eff_map=eff_map,
                        energy_weight=cfg["energy_weight_final"],
                        residual_torque_scale=cfg["residual_torque_final"])
    eval_reward, eval_metrics, eval_infos = eval_episode(eval_env, agent)

    # Stress test
    stress_env = MotorEnv(road=road_sets["stress_road"], ref=road_sets["stress_ref"],
                          style=style, mode="energy", eff_map=eff_map,
                          energy_weight=cfg["energy_weight_final"],
                          residual_torque_scale=cfg["residual_torque_final"])
    stress_reward, stress_metrics, stress_infos = eval_episode(stress_env, agent)

    result = {
        "seed":            seed,
        "style":           style,
        "history":         history,
        "eval_metrics":    eval_metrics,
        "stress_metrics":  stress_metrics,
        "eval_infos":      eval_infos,
        "stress_infos":    stress_infos,
        "eval_road":       road_sets["eval_road"],
        "eval_ref":        road_sets["eval_ref"],
    }

    # 保存 metrics json
    metrics_path = os.path.join(run_dir, "metrics.json")
    saveable = {
        "seed":           seed,
        "style":          style,
        "eval_metrics":   eval_metrics,
        "stress_metrics": stress_metrics,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(saveable, f, indent=2, ensure_ascii=False, default=str)

    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_all(
        eval_infos=result["eval_infos"],
        eval_metrics=result["eval_metrics"],
        eval_road=result["eval_road"],
        eval_ref=result["eval_ref"],
        history=result["history"],
        style=style,
        save_dir=plot_dir,
        eff_map=eff_map
    )

    return result


# ============================================================
#  多 seed 实验 + best seed 选择
# ============================================================
def run_multi_seed_experiment(style: str,
                              seeds: List[int] = [42],
                              eff_map: Optional[dict] = None,
                              output_dir: str = "results",
                              preset: str = "fast",
                              verbose: bool = True) -> dict:
    """
    多 seed 实验：每个 seed 独立训练，选择 best seed
    """
    if eff_map is None:
        eff_map = load_efficiency_map("sys_eff_pivot.csv")

    results = []
    for seed in seeds:
        r = run_single_seed(style=style, seed=seed, eff_map=eff_map,
                            output_dir=output_dir, preset=preset, verbose=verbose)
        results.append(r)

    # 选择 best seed：以控制节能为主，同时要求基本可用的跟踪
    def score_fn(r):
        eval_score = score_energy_metrics_for_selection(r["eval_metrics"])
        stress_score = score_energy_metrics_for_selection(r["stress_metrics"])
        return 0.7 * eval_score + 0.3 * stress_score

    best_result = max(results, key=score_fn)
    best_seed = best_result["seed"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"  BEST SEED for {style}: {best_seed}")
        print(f"  eval_saving_iso={best_result['eval_metrics'].get('saving_isochronous_pct', 0):.2f}%"
              f"  stress_saving_iso={best_result['stress_metrics'].get('saving_isochronous_pct', 0):.2f}%")
        print(f"{'='*60}")

    return {
        "style":       style,
        "best_seed":   best_seed,
        "best_result": best_result,
        "all_results": results,
    }


# ============================================================
#  完整实验入口
# ============================================================
def run_full_experiment(styles: List[str] = ["normal"],
                        seeds: List[int] = [42],
                        output_dir: str = "results",
                        eff_map: Optional[dict] = None,
                        eff_map_path: str = "sys_eff_pivot.csv",
                        preset: str = "fast",
                        verbose: bool = True) -> dict:
    """
    完整实验：三种风格 × 多 seed 训练与评估
    生成图像 / json / 模型
    """
    if eff_map is None:
        eff_map = load_efficiency_map(eff_map_path)
    all_experiments = {}

    for style in styles:
        exp = run_multi_seed_experiment(
            style=style, seeds=seeds, eff_map=eff_map,
            output_dir=output_dir, preset=preset, verbose=verbose
        )
        all_experiments[style] = exp

        # 为 best seed 生成绘图
        best = exp["best_result"]
        plot_dir = os.path.join(output_dir, style, f"seed_{exp['best_seed']}", "plots")
        os.makedirs(plot_dir, exist_ok=True)

        plot_all(
            eval_infos=best["eval_infos"],
            eval_metrics=best["eval_metrics"],
            eval_road=best["eval_road"],
            eval_ref=best["eval_ref"],
            history=best["history"],
            style=style,
            save_dir=plot_dir,
            eff_map=eff_map,
        )

    # 汇总 json
    summary = {}
    for style, exp in all_experiments.items():
        best = exp["best_result"]
        summary[style] = {
            "best_seed":            exp["best_seed"],
            "eval_metrics":         best["eval_metrics"],
            "stress_metrics":       best["stress_metrics"],
            "saving_isochronous_pct":       best["eval_metrics"].get("saving_isochronous_pct", 0),
            "worst_style_saving_isochronous_pct": best["stress_metrics"].get("saving_isochronous_pct", 0),
        }

    # robust_saving_joint_pct: 所有风格中最差的
    all_iso = [v["saving_isochronous_pct"] for v in summary.values()]
    for style in summary:
        summary[style]["robust_saving_joint_pct"] = min(all_iso)

    summary_path = os.path.join(output_dir, "experiment_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    if verbose:
        print(f"\n{'='*60}")
        print("  EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        for style, info in summary.items():
            print(f"  {style:8s}  seed={info['best_seed']}  "
                  f"iso_saving={info['saving_isochronous_pct']:.2f}%  "
                  f"stress_iso={info['worst_style_saving_isochronous_pct']:.2f}%  "
                  f"robust_joint={info['robust_saving_joint_pct']:.2f}%")

    return all_experiments
