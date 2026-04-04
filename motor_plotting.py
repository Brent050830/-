"""
motor_plotting.py  ——  绘图模块
=================================
职责：
  - 跟踪与能耗总图
  - energy saving effect 图
  - saving trace 图
  - 10-step / rolling window saving 图
  - 牵引/回收分诊断
  - 速度区间分诊断
  - 扭矩区间分诊断
  - 工作点命中热图
  突出：Pure Control Saving / Raw Saving / Low-speed Benefit
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

# 中文字体设置（优先微软雅黑，回退 SimHei）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

GEAR_RATIO = 9.0
WHEEL_RADIUS = 0.32


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


def _safe_total_saving_pct(ref_energy: np.ndarray, agent_energy: np.ndarray, eps: float = 1e-8) -> float:
    if len(ref_energy) == 0 or len(agent_energy) == 0:
        return 0.0
    ref_total = float(np.sum(ref_energy))
    agent_total = float(np.sum(agent_energy))
    return (ref_total - agent_total) / (abs(ref_total) + eps) * 100.0


def _bin_diagnostics(values: np.ndarray, ref_energy: np.ndarray, agent_energy: np.ndarray,
                     bins: np.ndarray, labels: List[str]) -> List[dict]:
    rows = []
    for idx in range(len(labels)):
        left = bins[idx]
        right = bins[idx + 1]
        mask = (values >= left) & (values < right)
        count = int(np.sum(mask))
        if count == 0:
            rows.append({"label": labels[idx], "count": 0, "saving_pct": 0.0})
            continue
        saving_pct = _safe_total_saving_pct(ref_energy[mask], agent_energy[mask])
        rows.append({"label": labels[idx], "count": count, "saving_pct": saving_pct})
    return rows


def _save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ============================================================
#  图 1：跟踪与能耗总图
# ============================================================
def plot_tracking_and_energy(infos: List[dict], metrics: dict, ref: dict,
                             style: str, save_dir: str):
    """
    绘制主图：速度跟踪、同一时间下的位移、累计节能率
    """
    agent_speed = _extract_series(infos, "speed")
    ref_speed = _extract_series(infos, "ref_speed")
    agent_dist = _extract_series(infos, "agent_dist")
    if len(agent_dist) == 0 or not np.any(agent_dist):
        agent_dist = np.arange(len(agent_speed))
    ref_dist = ref["ref_dist"][:len(agent_speed)]
    agent_time = _extract_series(infos, "total_time_agent")
    if len(agent_time) == 0 or not np.any(agent_time):
        agent_time = np.arange(len(agent_dist), dtype=float)
    ref_time = ref.get("ref_time", np.arange(len(ref_dist), dtype=float))[:len(agent_dist)]
    agent_energy = _extract_series(infos, "energy_step")
    ref_energy = ref["ref_energy_per_ds"][:len(agent_energy)]
    reward_basis = _extract_series(infos, "cmp_energy_step")
    if len(reward_basis) == 0 or not np.any(reward_basis):
        reward_basis = _extract_series(infos, "accounted_energy")
    ref_reward = ref.get("ref_cmp_energy_per_ds", ref_energy)[:len(agent_energy)]
    x_axis = ref_dist if len(ref_dist) == len(agent_speed) else np.arange(len(agent_speed))

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=False)

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
    ax.set_xlabel("距离 (m)")

    # --- 同一时间下的位移 ---
    ax = axes[1]
    ax.plot(ref_time, ref_dist[:len(ref_time)], color='royalblue', linestyle='--', alpha=0.85, linewidth=1.5, label='参考位移-时间')
    ax.plot(agent_time[:len(agent_dist)], agent_dist, color='darkorange', alpha=0.9, linewidth=1.6, label='智能体位移-时间')
    ax.set_ylabel("位移 (m)")
    ax.set_xlabel("时间 (s)")
    ref_end_time = float(ref_time[min(len(ref_time) - 1, len(ref_dist) - 1)]) if len(ref_time) > 0 else 0.0
    agent_end_time = float(agent_time[len(agent_dist) - 1]) if len(agent_time) > 0 else 0.0
    ax.set_title(f"同一时间下位移对比  |  参考到终点={ref_end_time:.2f}s  智能体到终点={agent_end_time:.2f}s")
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
#  图 2：Energy Saving Effect 图
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
#  图 3：Saving Trace 图
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
#  图 4：10-step / Rolling Window Saving 图
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
    if len(ref_roll) > 0:
        abs_ref_roll = np.abs(ref_roll)
        # 停走/回收窗口里参考能耗可能接近 0，直接算百分比会被极端放大
        denom_floor = max(float(np.percentile(abs_ref_roll, 15)) * 0.25, 0.5)
        valid = abs_ref_roll >= denom_floor
    else:
        denom_floor = 0.5
        valid = np.zeros_like(ref_roll, dtype=bool)

    roll_saving_pct = np.full(len(ref_roll), np.nan, dtype=float)
    roll_saving_pct[valid] = (
        (ref_roll[valid] - agent_roll[valid]) / (np.abs(ref_roll[valid]) + 1e-8) * 100.0
    )
    steps = np.arange(len(roll_saving_pct))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # 滑窗节能率
    ax = axes[0]
    valid_steps = steps[valid]
    valid_pct = roll_saving_pct[valid]
    colors = np.where(valid_pct >= 0, 'green', 'red')
    ax.bar(valid_steps, valid_pct, color=colors, alpha=0.6, width=1.0)
    invalid_count = int(np.sum(~valid))
    if invalid_count > 0:
        ax.scatter(
            steps[~valid],
            np.zeros(invalid_count),
            s=8,
            color='gray',
            alpha=0.45,
            label='低分母窗口(已跳过)',
        )
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_ylabel(f"{window_size}-step 窗口节能率 (%)")
    ax.set_title(f"[{style.upper()}] Rolling Window Saving (window={window_size})")
    ax.grid(True, alpha=0.3)
    if len(valid_pct) > 0:
        lo = float(np.percentile(valid_pct, 2))
        hi = float(np.percentile(valid_pct, 98))
        pad = max((hi - lo) * 0.15, 3.0)
        ax.set_ylim(lo - pad, hi + pad)

    # 统计
    neg_ratio = float(np.sum(valid_pct < 0)) / max(len(valid_pct), 1) * 100
    worst     = float(np.min(valid_pct)) if len(valid_pct) > 0 else 0.0
    mean_sv   = float(np.mean(valid_pct)) if len(valid_pct) > 0 else 0.0

    ax.text(0.02, 0.05,
            f"mean={mean_sv:.1f}%  worst={worst:.1f}%  neg_ratio={neg_ratio:.1f}%  skipped={invalid_count}",
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    if invalid_count > 0:
        ax.legend(loc='upper right')

    # 滑窗能耗对比
    ax = axes[1]
    ax.plot(steps, ref_roll, 'b--', alpha=0.7, label=f'参考 {window_size}-step 能耗')
    ax.plot(steps, agent_roll, 'r-', alpha=0.7, label=f'智能体 {window_size}-step 能耗')
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_ylabel("窗口累积能耗 (Wh)")
    ax.set_xlabel("窗口起始 step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"rolling_window_saving_{style}.png"), dpi=150)
    plt.close(fig)


# ============================================================
#  诊断 1：牵引/回收分段
# ============================================================
def plot_traction_regen_diagnostics(infos: List[dict], ref: dict, style: str, save_dir: str) -> dict:
    agent_torque = _extract_series(infos, "motor_torque")
    agent_energy = _extract_series(infos, "energy_step")
    agent_cmp = _extract_series(infos, "cmp_energy_step")
    ref_energy = ref["ref_energy_per_ds"][:len(agent_energy)]
    ref_cmp = ref.get("ref_cmp_energy_per_ds", ref_energy)[:len(agent_energy)]

    masks = {
        "traction": agent_torque >= 0.0,
        "regen": agent_torque < 0.0,
    }
    rows = []
    labels = []
    saving_real = []
    saving_reward = []
    counts = []
    for name, mask in masks.items():
        labels.append("牵引段" if name == "traction" else "回收段")
        counts.append(int(np.sum(mask)))
        rows.append({
            "segment": name,
            "count": int(np.sum(mask)),
            "saving_total_pct": _safe_total_saving_pct(ref_energy[mask], agent_energy[mask]),
            "saving_reward_pct": _safe_total_saving_pct(ref_cmp[mask], agent_cmp[mask]),
        })
        saving_real.append(rows[-1]["saving_total_pct"])
        saving_reward.append(rows[-1]["saving_reward_pct"])

    x = np.arange(len(labels))
    width = 0.35
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    ax = axes[0]
    ax.bar(x - width / 2, saving_real, width=width, color="forestgreen", alpha=0.75, label="真实节能率")
    ax.bar(x + width / 2, saving_reward, width=width, color="royalblue", alpha=0.75, label="训练口径节能率")
    ax.axhline(0, color="black", linewidth=0.8)
    y_min = min(min(saving_real), min(saving_reward), 0.0)
    y_max = max(max(saving_real), max(saving_reward), 0.0)
    pad = max((y_max - y_min) * 0.18, 1.0)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_ylabel("节能率 (%)")
    ax.set_title(f"[{style.upper()}] 诊断1：牵引/回收分段")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x, counts, color="slateblue", alpha=0.75, width=0.45)
    for idx, count in enumerate(counts):
        ax.text(x[idx], count + max(counts) * 0.03 + 1e-6, f"n={count}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("步数")
    ax.set_xticks(x, labels)
    ax.set_xlabel("分段类型")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"diagnostic_traction_regen_{style}.png"), dpi=150)
    plt.close(fig)
    return {"traction_regen": rows}


# ============================================================
#  诊断 2：速度区间
# ============================================================
def plot_speed_bin_diagnostics(infos: List[dict], ref: dict, style: str, save_dir: str) -> dict:
    ref_speed = ref["ref_speed"][:len(infos)]
    ref_energy = ref["ref_energy_per_ds"][:len(infos)]
    agent_energy = _extract_series(infos, "energy_step")
    bins = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 1e9])
    labels = ["0-10", "10-20", "20-30", "30-40", "40+"]
    rows = _bin_diagnostics(ref_speed, ref_energy, agent_energy, bins, labels)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    x = np.arange(len(labels))
    axes[0].bar(x, [r["saving_pct"] for r in rows], color=np.where(np.array([r["saving_pct"] for r in rows]) >= 0, "green", "red"), alpha=0.7)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("节能率 (%)")
    axes[0].set_title(f"[{style.upper()}] 诊断2：速度区间节能率")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(x, [r["count"] for r in rows], color="slateblue", alpha=0.7)
    axes[1].set_ylabel("步数")
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("参考速度区间 (m/s)")
    axes[1].grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"diagnostic_speed_bins_{style}.png"), dpi=150)
    plt.close(fig)
    return {"speed_bins": rows}


# ============================================================
#  诊断 3：扭矩区间
# ============================================================
def plot_torque_bin_diagnostics(infos: List[dict], ref: dict, style: str, save_dir: str) -> dict:
    ref_torque = np.abs(ref["ref_torque"][:len(infos)])
    ref_energy = ref["ref_energy_per_ds"][:len(infos)]
    agent_energy = _extract_series(infos, "energy_step")
    bins = np.array([0.0, 25.0, 50.0, 100.0, 150.0, 220.0, 1e9])
    labels = ["0-25", "25-50", "50-100", "100-150", "150-220", "220+"]
    rows = _bin_diagnostics(ref_torque, ref_energy, agent_energy, bins, labels)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    x = np.arange(len(labels))
    axes[0].bar(x, [r["saving_pct"] for r in rows], color=np.where(np.array([r["saving_pct"] for r in rows]) >= 0, "green", "red"), alpha=0.7)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("节能率 (%)")
    axes[0].set_title(f"[{style.upper()}] 诊断3：扭矩区间节能率")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(x, [r["count"] for r in rows], color="darkorange", alpha=0.7)
    axes[1].set_ylabel("步数")
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("参考扭矩绝对值区间 (Nm)")
    axes[1].grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"diagnostic_torque_bins_{style}.png"), dpi=150)
    plt.close(fig)
    return {"torque_bins": rows}


# ============================================================
#  诊断 4：路段分段
# ============================================================
def plot_route_segment_diagnostics(infos: List[dict], ref: dict, style: str, save_dir: str, num_segments: int = 10) -> dict:
    agent_energy = _extract_series(infos, "energy_step")
    ref_energy = ref["ref_energy_per_ds"][:len(agent_energy)]
    n = len(agent_energy)
    if n == 0:
        return {"route_segments": []}

    segment_len = max(n // num_segments, 1)
    rows = []
    labels = []
    real_vals = []
    counts = []
    for seg_idx in range(num_segments):
        start = seg_idx * segment_len
        end = n if seg_idx == num_segments - 1 else min((seg_idx + 1) * segment_len, n)
        if start >= n:
            break
        seg_ref = ref_energy[start:end]
        seg_agent = agent_energy[start:end]
        saving_pct = _safe_total_saving_pct(seg_ref, seg_agent)
        label = f"{start}-{end}"
        rows.append({
            "segment_index": seg_idx,
            "start_step": start,
            "end_step": end,
            "count": int(end - start),
            "saving_pct": saving_pct,
        })
        labels.append(label)
        real_vals.append(saving_pct)
        counts.append(int(end - start))

    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].bar(x, real_vals, color=np.where(np.array(real_vals) >= 0, "green", "red"), alpha=0.75)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("节能率 (%)")
    axes[0].set_title(f"[{style.upper()}] 诊断4：路段分段节能率")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(x, counts, color="teal", alpha=0.75)
    axes[1].set_ylabel("步数")
    axes[1].set_xticks(x, labels, rotation=45)
    axes[1].set_xlabel("路段 step 区间")
    axes[1].grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"diagnostic_route_segments_{style}.png"), dpi=150)
    plt.close(fig)
    return {"route_segments": rows}


# ============================================================
#  诊断 5：工作点命中热图
# ============================================================
def plot_operating_point_heatmap(infos: List[dict], ref: dict, style: str, save_dir: str, eff_map: Optional[dict] = None) -> dict:
    agent_torque = _extract_series(infos, "motor_torque")
    agent_rpm = _extract_series(infos, "motor_rpm")
    ref_speed = ref["ref_speed"][:len(infos)]
    ref_torque = ref["ref_torque"][:len(infos)]
    ref_rpm = ref_speed / WHEEL_RADIUS * GEAR_RATIO * 60.0 / (2.0 * np.pi)

    torque_min = float(min(np.min(agent_torque), np.min(ref_torque)))
    torque_max = float(max(np.max(agent_torque), np.max(ref_torque)))
    rpm_min = float(min(np.min(agent_rpm), np.min(ref_rpm)))
    rpm_max = float(max(np.max(agent_rpm), np.max(ref_rpm)))
    torque_bins = np.linspace(torque_min, torque_max, 41)
    rpm_bins = np.linspace(rpm_min, rpm_max, 41)

    fig, ax = plt.subplots(1, 1, figsize=(11, 7))
    ax.set_facecolor('#E0E0E0')  # 灰色背景代表电机无数据的非物理工作区域(图中空白缺口)
    
    # 绘制满底色的效率图
    heatmap_im = None
    if eff_map is not None:
        X, Y = np.meshgrid(eff_map["speeds"], eff_map["torques"])
        Z = eff_map["eff"] * 100.0
        levels = np.linspace(72, 96, 25)
        heatmap_im = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.9, zorder=0)
        cs = ax.contour(X, Y, Z, levels=[75, 80, 85, 90, 93, 95], colors='white', alpha=0.5, linewidths=0.8, zorder=1)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.0f%%')

    # 用散点路径图叠加真实轨迹点迹 (双轨合并在一张图上)
    ax.plot(ref_rpm, ref_torque, marker='s', color='#333333', alpha=0.45, markersize=3, linewidth=1.5, zorder=4, label='参考老司机轨迹 (Ref)')
    ax.plot(agent_rpm, agent_torque, marker='o', color='white', markeredgecolor='red', alpha=0.8, markersize=4, linewidth=1.5, zorder=5, label='智能体轨迹 (Agent)')
    
    # 起终点标识
    if len(ref_rpm) > 0:
        ax.plot(ref_rpm[0], ref_torque[0], marker='^', color='cyan', markersize=9, markeredgecolor='black', zorder=6, label='起点')
        ax.plot(ref_rpm[-1], ref_torque[-1], marker='*', color='gold', markersize=13, markeredgecolor='red', zorder=6, label='终点')

    ax.set_title(f"[{style.upper()}] 主控工作点 vs 参考工作点 偏移对比图")
    ax.set_xlabel("电机转速 (rpm)")
    ax.set_ylabel("输出扭矩 (Nm)")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    
    # 给效率底图加上正常排版的全局颜色条
    if heatmap_im is not None:
        cbar = fig.colorbar(heatmap_im, ax=ax, shrink=0.9, pad=0.03)
        cbar.set_label("电机效率 (%)")
    
    # 将视角动态框定在轨迹的附近，留出余量
    pad_rpm = max((rpm_max - rpm_min) * 0.15, 1000)
    pad_torque = max((torque_max - torque_min) * 0.15, 20)
    ax.set_xlim(max(0, rpm_min - pad_rpm), rpm_max + pad_rpm)
    ax.set_ylim(torque_min - pad_torque, torque_max + pad_torque)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"diagnostic_operating_points_{style}.png"), dpi=150)
    plt.close(fig)
    return {
        "operating_points": {
            "agent_unique_points": int(len(agent_torque)),
            "ref_unique_points": int(len(ref_torque)),
            "rpm_range": [rpm_min, rpm_max],
            "torque_range": [torque_min, torque_max],
        }
    }


def plot_actual_torque_delta(infos: List[dict], ref: dict, style: str, save_dir: str) -> None:
    """绘制参考/实际扭矩及残差、命令偏差、实际偏差。"""
    agent_torque = _extract_series(infos, "motor_torque")
    torque_cmd = _extract_series(infos, "motor_torque_cmd")
    residual_torque = _extract_series(infos, "residual_torque")
    ref_torque = _extract_series(infos, "ref_torque")
    x_axis = ref["ref_dist"][:len(agent_torque)]
    if len(x_axis) != len(agent_torque):
        x_axis = np.arange(len(agent_torque))

    cmd_delta = torque_cmd - ref_torque
    actual_delta = agent_torque - ref_torque

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax = axes[0]
    ax.plot(x_axis, ref_torque, color="royalblue", linestyle="--", linewidth=1.5, label="参考扭矩")
    ax.plot(x_axis, agent_torque, color="crimson", linewidth=1.6, label="智能体实际扭矩")
    ax.fill_between(x_axis, ref_torque, agent_torque, color="orange", alpha=0.10)
    ax.set_ylabel("扭矩 (Nm)")
    ax.set_title(f"[{style.upper()}] 参考/实际扭矩与偏差图")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(x_axis, residual_torque, color="purple", linewidth=1.2, label="残差扭矩")
    ax.plot(x_axis, cmd_delta, color="darkgreen", linewidth=1.3, label="命令偏差 = T_cmd - T_ref")
    ax.plot(x_axis, actual_delta, color="darkorange", linewidth=1.5, label="实际偏差 = T_agent - T_ref")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("偏差 (Nm)")
    ax.set_xlabel("距离 (m)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"actual_torque_delta_{style}.png"), dpi=150)
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
    handles, labels = ax.get_legend_handles_labels()
    if handles:
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
    handles, labels = ax.get_legend_handles_labels()
    if handles:
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
             window_size: int = 10,
             eff_map: Optional[dict] = None):
    """
    一键生成全部绘图
    """
    os.makedirs(save_dir, exist_ok=True)
    stale_trajectory = os.path.join(save_dir, f"trajectory_{style}.png")
    if os.path.exists(stale_trajectory):
        os.remove(stale_trajectory)

    print(f"  生成绘图 [{style}] -> {save_dir}")

    plot_tracking_and_energy(eval_infos, eval_metrics, eval_ref, style, save_dir)
    plot_actual_torque_delta(eval_infos, eval_ref, style, save_dir)
    plot_energy_saving_effect(eval_infos, eval_metrics, eval_ref, style, save_dir)
    plot_saving_trace(eval_infos, eval_ref, style, save_dir)
    plot_rolling_window_saving(eval_infos, eval_ref, style, save_dir, window_size)
    diagnostics = {}
    diagnostics.update(plot_traction_regen_diagnostics(eval_infos, eval_ref, style, save_dir))
    diagnostics.update(plot_speed_bin_diagnostics(eval_infos, eval_ref, style, save_dir))
    diagnostics.update(plot_torque_bin_diagnostics(eval_infos, eval_ref, style, save_dir))
    diagnostics.update(plot_route_segment_diagnostics(eval_infos, eval_ref, style, save_dir))
    diagnostics.update(plot_operating_point_heatmap(eval_infos, eval_ref, style, save_dir, eff_map=eff_map))
    _save_json(os.path.join(save_dir, f"diagnostics_{style}.json"), diagnostics)
    plot_training_curves(history, style, save_dir)

    print(f"  绘图完成：主图与分段/区间/工作点诊断已保存到 {save_dir}")
