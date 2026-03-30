"""
motor_plotting.py  ——  绘图模块
=================================
职责：
  - 跟踪与能耗总图
  - 轨迹图
  - energy saving effect 图
  - saving trace 图
  - 10-step / rolling window saving 图
  突出：Pure Control Saving / Raw Saving / Low-speed Benefit
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

# 中文字体设置（优先微软雅黑，回退 SimHei）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
#  工具函数
# ============================================================
def _extract_series(infos: List[dict], key: str) -> np.ndarray:
    """从 step_infos 列表提取指定 key 序列"""
    vals = []
    for info in infos:
        if key in info:
            vals.append(info[key])
        elif "reward_info" in info and key in info["reward_info"]:
            vals.append(info["reward_info"][key])
        else:
            vals.append(0.0)
    return np.array(vals)


def _rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    """滑动窗口求和"""
    if len(arr) < window:
        return arr
    cs = np.cumsum(arr)
    cs = np.insert(cs, 0, 0)
    return cs[window:] - cs[:-window]


def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """简单滑动平均，长度与输入一致。"""
    if len(arr) == 0:
        return arr
    if window <= 1:
        return arr.copy()
    if window >= len(arr):
        return np.full(len(arr), float(np.mean(arr)))
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def _safe_saving_pct(ref_energy: np.ndarray, agent_energy: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """稳健地计算逐步节能率，参考能耗过小时置为 NaN，避免图像被极端值拉坏。"""
    n = min(len(ref_energy), len(agent_energy))
    out = np.full(n, np.nan, dtype=float)
    if n == 0:
        return out
    ref_arr = np.asarray(ref_energy[:n], dtype=float)
    agent_arr = np.asarray(agent_energy[:n], dtype=float)
    valid = np.abs(ref_arr) > eps
    out[valid] = (ref_arr[valid] - agent_arr[valid]) / (np.abs(ref_arr[valid]) + eps) * 100.0
    return out


# ============================================================
#  图 1：跟踪与能耗总图
# ============================================================
def plot_tracking_and_energy(infos: List[dict], metrics: dict, ref: dict,
                             style: str, save_dir: str):
    """
    绘制主图：速度跟踪、位移跟踪、累计节能率
    """
    agent_speed = _extract_series(infos, "speed")
    ref_speed = _extract_series(infos, "ref_speed")
    agent_dist = _extract_series(infos, "agent_dist")
    if len(agent_dist) == 0 or not np.any(agent_dist):
        agent_dist = np.arange(len(agent_speed))
    ref_dist = ref["ref_dist"][:len(agent_speed)]
    agent_energy = _extract_series(infos, "energy_step")
    ref_energy = ref["ref_energy_per_ds"][:len(agent_energy)]
    reward_basis = _extract_series(infos, "cmp_energy_step")
    if len(reward_basis) == 0 or not np.any(reward_basis):
        reward_basis = _extract_series(infos, "accounted_energy")
    ref_reward = ref.get("ref_cmp_energy_per_ds", ref_energy)[:len(agent_energy)]
    x_axis = ref_dist if len(ref_dist) == len(agent_speed) else np.arange(len(agent_speed))

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    # --- 速度跟踪 ---
    ax = axes[0]
    ax.plot(x_axis, ref_speed, color='royalblue', linestyle='--', alpha=0.85, linewidth=1.5, label='参考速度')
    ax.plot(x_axis, agent_speed, color='crimson', alpha=0.9, linewidth=1.6, label='智能体速度')
    ax.fill_between(x_axis, ref_speed, agent_speed, alpha=0.12, color='orange')
    ax.set_ylabel("速度 (m/s)")
    ax.set_title(
        f"[{style.upper()}] 主结果图  |  真实节能率={metrics.get('saving_total_pct', 0):.2f}%  "
        f"速度MAE={metrics.get('speed_mae', 0):.3f}  位移MAE={metrics.get('dist_mae', 0):.3f}"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- 位移跟踪 ---
    ax = axes[1]
    ax.plot(x_axis, ref_dist, color='royalblue', linestyle='--', alpha=0.85, linewidth=1.5, label='参考位移')
    ax.plot(x_axis, agent_dist[:len(x_axis)], color='darkorange', alpha=0.9, linewidth=1.6, label='智能体位移')
    ax.fill_between(x_axis, ref_dist, agent_dist[:len(x_axis)], alpha=0.10, color='goldenrod')
    ax.set_ylabel("位移 (m)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # --- 累计节能率 ---
    ax = axes[2]
    cum_ref = np.cumsum(ref_energy)
    cum_agent = np.cumsum(agent_energy)
    cum_ref_reward = np.cumsum(ref_reward)
    cum_agent_reward = np.cumsum(reward_basis)
    true_saving_pct = (cum_ref - cum_agent) / (np.abs(cum_ref) + 1e-8) * 100
    reward_saving_pct = (cum_ref_reward - cum_agent_reward) / (np.abs(cum_ref_reward) + 1e-8) * 100
    ax.plot(x_axis[:len(true_saving_pct)], true_saving_pct, color='forestgreen', linewidth=1.8, label='真实累计节能率')
    ax.plot(x_axis[:len(reward_saving_pct)], reward_saving_pct, color='navy', linewidth=1.6,
            linestyle='--', label='训练口径累计节能率')
    ax.fill_between(
        x_axis[:len(true_saving_pct)],
        0,
        true_saving_pct,
        where=true_saving_pct >= 0,
        color='green',
        alpha=0.12,
    )
    ax.fill_between(
        x_axis[:len(true_saving_pct)],
        0,
        true_saving_pct,
        where=true_saving_pct < 0,
        color='red',
        alpha=0.10,
    )
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel("累计节能率 (%)")
    ax.set_xlabel("距离 (m)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"tracking_energy_{style}.png"), dpi=150)
    plt.close(fig)


# ============================================================
#  图 2：轨迹图（扭矩 + regen/coast + SOC）
# ============================================================
def plot_trajectory(infos: List[dict], ref: dict, style: str, save_dir: str):
    """
    绘制轨迹图：扭矩对比、残差扭矩、加速度误差、SOC 变化
    """
    agent_torque  = _extract_series(infos, "motor_torque")
    ref_torque    = ref["ref_torque"][:len(agent_torque)]
    soc           = _extract_series(infos, "soc")
    residual_torque = _extract_series(infos, "residual_torque")
    accel = _extract_series(infos, "accel")
    ref_accel = _extract_series(infos, "ref_accel")
    steps = np.arange(len(agent_torque))

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # 扭矩
    ax = axes[0]
    ax.plot(steps, ref_torque, 'b--', alpha=0.6, label='参考扭矩')
    ax.plot(steps, agent_torque, 'r-', alpha=0.7, label='智能体扭矩')
    ax.set_ylabel("扭矩 (Nm)")
    ax.set_title(f"[{style.upper()}] 轨迹详细图")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 残差扭矩
    ax = axes[1]
    ax.bar(steps, residual_torque, color='purple', alpha=0.5, width=1.0, label='残差扭矩')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_ylabel("残差扭矩 (Nm)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # acceleration tracking
    ax = axes[2]
    ax.plot(steps, ref_accel, color='royalblue', linestyle='--', linewidth=1.4, label='参考加速度')
    ax.plot(steps, accel, color='darkgreen', linewidth=1.4, label='智能体加速度')
    ax.fill_between(steps, ref_accel, accel, color='mediumseagreen', alpha=0.12)
    ax.set_ylabel("加速度 (m/s²)")
    ax.set_title(
        f"加速度跟踪  |  mae={np.mean(np.abs(accel - ref_accel)) if len(accel) > 0 else 0.0:.3f}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # SOC
    ax = axes[3]
    ax.plot(steps, soc, 'b-', linewidth=1.5, label='SOC')
    ax.set_ylabel("SOC")
    ax.set_xlabel("距离步 (step)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"trajectory_{style}.png"), dpi=150)
    plt.close(fig)


# ============================================================
#  图 3：Energy Saving Effect 图
# ============================================================
def plot_energy_saving_effect(infos: List[dict], metrics: dict, ref: dict,
                              style: str, save_dir: str):
    """
    绘制节能效果图：区分效率损失口径与最终真实节能口径
    """
    agent_energy = _extract_series(infos, "energy_step")
    reward_basis = _extract_series(infos, "cmp_energy_step")
    if len(reward_basis) == 0 or not np.any(reward_basis):
        reward_basis = _extract_series(infos, "accounted_energy")
    ref_energy   = ref["ref_energy_per_ds"][:len(agent_energy)]
    ref_reward   = ref.get("ref_cmp_energy_per_ds", ref_energy)[:len(agent_energy)]
    agent_speed  = _extract_series(infos, "speed")
    steps = np.arange(len(agent_energy))

    raw_saving = ref_energy - agent_energy
    reward_saving = ref_reward - reward_basis
    speed_effect = raw_saving - reward_saving
    low_speed_mask = agent_speed < 10.0
    raw_saving_ma = _moving_average(raw_saving, 25)
    reward_saving_ma = _moving_average(reward_saving, 25)
    speed_effect_ma = _moving_average(speed_effect, 25)
    cum_raw = np.cumsum(raw_saving)
    cum_reward = np.cumsum(reward_saving)
    cum_speed_effect = np.cumsum(speed_effect)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # 累积真实节能 vs 效率损失口径节能
    ax = axes[0]
    ax.plot(steps, cum_raw, color='forestgreen', linewidth=1.8, label='真实累计节能')
    ax.plot(steps, cum_reward, color='royalblue', linewidth=1.8, label='效率损失口径累计节能')
    ax.plot(steps, cum_speed_effect, color='darkorange', linewidth=1.4, linestyle='--',
            label='速度相关能耗差累计')
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_ylabel("累计节能 (Wh)")
    ax.set_title(
        f"[{style.upper()}] Energy Saving Effect  |  "
        f"真实={metrics.get('saving_total_pct', 0):.2f}%  "
        f"效率损失口径={metrics.get('saving_cmp_total_pct', 0):.2f}%  "
        f"速度差={metrics.get('speed_energy_bias_pct', 0):.2f}%"
    )
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # 逐步节能主项
    ax = axes[1]
    ax.plot(steps, raw_saving, color='forestgreen', alpha=0.15, linewidth=0.8)
    ax.plot(steps, reward_saving, color='royalblue', alpha=0.15, linewidth=0.8)
    ax.plot(steps, raw_saving_ma, color='green', linewidth=1.6, label='真实逐步节能 (25-step MA)')
    ax.plot(steps, reward_saving_ma, color='blue', linewidth=1.6, label='效率损失口径节能 (25-step MA)')
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_ylabel("逐步节能 (Wh)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # 速度相关能耗差
    ax = axes[2]
    colors = np.where(speed_effect >= 0, 'darkorange', 'slateblue')
    ax.bar(steps, speed_effect, color=colors, alpha=0.45, width=1.0, label='速度相关能耗差')
    ax.plot(steps, speed_effect_ma, color='black', linewidth=1.4, label='25-step MA')
    ax.fill_between(
        steps,
        0.0,
        1.0,
        where=low_speed_mask,
        color='gray',
        alpha=0.08,
        transform=ax.get_xaxis_transform(),
        label='低速区',
    )
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_ylabel("真实-训练口径 (Wh)")
    ax.set_xlabel("距离步 (step)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"energy_saving_effect_{style}.png"), dpi=150)
    plt.close(fig)


# ============================================================
#  图 4：Saving Trace 图
# ============================================================
def plot_saving_trace(infos: List[dict], ref: dict, style: str, save_dir: str):
    """
    绘制逐步节能率变化（saving trace）
    """
    agent_energy = _extract_series(infos, "energy_step")
    ref_energy   = ref["ref_energy_per_ds"][:len(agent_energy)]
    steps = np.arange(len(agent_energy))

    # 逐步节能率
    rho = _safe_saving_pct(ref_energy, agent_energy)
    rho_ma = _moving_average(np.nan_to_num(rho, nan=0.0), 25)
    # 累积节能率
    cum_ref   = np.cumsum(ref_energy)
    cum_agent = np.cumsum(agent_energy)
    cum_saving_pct = (cum_ref - cum_agent) / (np.abs(cum_ref) + 1e-8) * 100

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax = axes[0]
    valid = np.isfinite(rho)
    ax.plot(steps[valid], rho[valid], color='teal', alpha=0.20, linewidth=0.8, label='逐步节能率 ρ_k (%)')
    ax.plot(steps, rho_ma, color='darkblue', linewidth=1.5, label='逐步节能率移动平均 (25-step)')
    ax.axhline(0, color='k', linewidth=0.8)
    ax.fill_between(steps[valid], 0, rho[valid], where=rho[valid] >= 0, color='green', alpha=0.12)
    ax.fill_between(steps[valid], 0, rho[valid], where=rho[valid] < 0, color='red', alpha=0.12)
    if np.any(valid):
        lo = float(np.percentile(rho[valid], 2))
        hi = float(np.percentile(rho[valid], 98))
        pad = max((hi - lo) * 0.15, 5.0)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_ylabel("节能率 (%)")
    ax.set_title(f"[{style.upper()}] Saving Trace")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(steps, cum_saving_pct, color='darkblue', linewidth=1.5, label='累积节能率 (%)')
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_ylabel("累积节能率 (%)")
    ax.set_xlabel("距离步 (step)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"saving_trace_{style}.png"), dpi=150)
    plt.close(fig)


# ============================================================
#  图 5：10-step / Rolling Window Saving 图
# ============================================================
def plot_rolling_window_saving(infos: List[dict], ref: dict, style: str,
                                save_dir: str, window_size: int = 10):
    """
    绘制 rolling window saving 图
    """
    agent_energy = _extract_series(infos, "energy_step")
    ref_energy   = ref["ref_energy_per_ds"][:len(agent_energy)]

    # rolling sum
    agent_roll = _rolling_sum(agent_energy, window_size)
    ref_roll   = _rolling_sum(ref_energy, window_size)
    roll_saving_pct = (ref_roll - agent_roll) / (np.abs(ref_roll) + 1e-8) * 100
    steps = np.arange(len(roll_saving_pct))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # 滑窗节能率
    ax = axes[0]
    colors = np.where(roll_saving_pct >= 0, 'green', 'red')
    ax.bar(steps, roll_saving_pct, color=colors, alpha=0.6, width=1.0)
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_ylabel(f"{window_size}-step 窗口节能率 (%)")
    ax.set_title(f"[{style.upper()}] Rolling Window Saving (window={window_size})")
    ax.grid(True, alpha=0.3)

    # 统计
    neg_ratio = float(np.sum(roll_saving_pct < 0)) / max(len(roll_saving_pct), 1) * 100
    worst     = float(np.min(roll_saving_pct)) if len(roll_saving_pct) > 0 else 0.0
    mean_sv   = float(np.mean(roll_saving_pct)) if len(roll_saving_pct) > 0 else 0.0

    ax.text(0.02, 0.05,
            f"mean={mean_sv:.1f}%  worst={worst:.1f}%  neg_ratio={neg_ratio:.1f}%",
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 滑窗能耗对比
    ax = axes[1]
    ax.plot(steps, ref_roll, 'b--', alpha=0.7, label=f'参考 {window_size}-step 能耗')
    ax.plot(steps, agent_roll, 'r-', alpha=0.7, label=f'智能体 {window_size}-step 能耗')
    ax.set_ylabel("窗口累积能耗 (Wh)")
    ax.set_xlabel("窗口起始 step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"rolling_window_saving_{style}.png"), dpi=150)
    plt.close(fig)


# ============================================================
#  训练曲线图（附加）
# ============================================================
def plot_training_curves(history: dict, style: str, save_dir: str):
    """绘制训练过程中的 reward 曲线与 lambda 变化"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # --- Reward curve ---
    ax = axes[0]
    all_rewards = []
    stage_boundaries = []
    for stage_name in ["track", "energy", "polish"]:
        if stage_name in history:
            rews = history[stage_name]["episode_rewards"]
            stage_boundaries.append((len(all_rewards), len(all_rewards) + len(rews), stage_name))
            all_rewards.extend(rews)

    if len(all_rewards) > 0:
        eps = np.arange(len(all_rewards))
        ax.plot(eps, all_rewards, color='navy', alpha=0.4, linewidth=0.5)
        # 滑动平均
        if len(all_rewards) > 20:
            kernel = np.ones(20) / 20
            smoothed = np.convolve(all_rewards, kernel, mode='valid')
            ax.plot(np.arange(len(smoothed)) + 10, smoothed, color='red', linewidth=1.5,
                    label='移动平均(20)')
        # 阶段分界
        for start, end, sn in stage_boundaries:
            ax.axvline(start, color='gray', linestyle=':', alpha=0.5)
            ax.text(start + 2, max(all_rewards) * 0.9, sn, fontsize=9, color='gray')

    ax.set_ylabel("Episode Reward")
    ax.set_title(f"[{style.upper()}] 训练曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Lambda 变化 ---
    ax = axes[1]
    if "energy" in history and len(history["energy"]["lambda_history"]) > 0:
        lam_hist = history["energy"]["lambda_history"]
        names = list(lam_hist[0].keys())
        for name in names:
            vals = [d[name] for d in lam_hist]
            ax.plot(vals, label=f"λ_{name}", alpha=0.8)
    ax.set_ylabel("Lagrangian λ")
    ax.set_xlabel("Episode (energy stage)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"training_curves_{style}.png"), dpi=150)
    plt.close(fig)


# ============================================================
#  总入口
# ============================================================
def plot_all(eval_infos: List[dict],
             eval_metrics: dict,
             eval_road: dict,
             eval_ref: dict,
             history: dict,
             style: str,
             save_dir: str,
             window_size: int = 10):
    """
    一键生成全部绘图
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"  生成绘图 [{style}] -> {save_dir}")

    plot_tracking_and_energy(eval_infos, eval_metrics, eval_ref, style, save_dir)
    plot_trajectory(eval_infos, eval_ref, style, save_dir)
    plot_energy_saving_effect(eval_infos, eval_metrics, eval_ref, style, save_dir)
    plot_saving_trace(eval_infos, eval_ref, style, save_dir)
    plot_rolling_window_saving(eval_infos, eval_ref, style, save_dir, window_size)
    plot_training_curves(history, style, save_dir)

    print(f"  绘图完成：共 6 张图已保存到 {save_dir}")
