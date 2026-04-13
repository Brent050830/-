"""
motor_env.py - 电动汽车电机环境与奖励逻辑模块

该模块实现了一个强化学习环境，用于训练电动汽车能耗优化控制策略。

主要功能：
1. 车辆纵向动力学模型：包含质量、轮胎、空气阻力等参数
2. 电机效率查询：基于扭矩和转速的二维插值查询
3. 能耗计算：包括电池功率、损耗能量等多种能耗指标
4. 路况生成：包含随机坡度、弯道限速等真实路况特征
5. 参考轨迹生成：三种驾驶风格（经济/正常/运动）的参考速度和扭矩轨迹
6. 强化学习环境（MotorEnv）：
    - 状态观测：包含速度、SOC、扭矩、预测、风格等30维特征
    - 动作空间：单维度残差扭矩调整 [-1, 1]
    - 奖励机制：支持跟踪模式和优化模式，综合考虑轨迹跟踪和能耗节省
    - 约束评估：速度、距离、平顺性、投影、窗口能耗五大约束
    - 性能评估：提供详细的能耗节省率、跟踪精度等多维评估指标

主要类：
- MotorEnv: 强化学习环境主类

主要函数：
- load_efficiency_map(): 加载电机效率表
- query_efficiency(): 查询特定工况下的电机效率
- compute_battery_power(): 计算电池功率
- generate_road(): 生成随机路况
- generate_reference_trajectory(): 生成参考轨迹
motor_env.py - Motor environment and reward logic.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 车辆的纵向动力学参数 
VEHICLE_MASS = 1800.0
WHEEL_RADIUS = 0.32
FRONTAL_AREA = 2.4
DRAG_COEFF = 0.30
ROLL_RESIST = 0.012
AIR_DENSITY = 1.225
GRAVITY = 9.81
GEAR_RATIO = 9.0
MOTOR_MAX_TORQUE = 320.0
MOTOR_MAX_SPEED = 13000.0
MOTOR_BASE_SPEED = 4000.0
MOTOR_MAX_POWER_W = MOTOR_MAX_TORQUE * MOTOR_BASE_SPEED * 2.0 * math.pi / 60.0
MOTOR_MAX_REGEN_TORQUE = MOTOR_MAX_TORQUE * 0.5
MOTOR_MAX_REGEN_POWER_W = MOTOR_MAX_POWER_W * 0.5
DRIVELINE_EFFICIENCY = 0.95
BATTERY_CAPACITY_WH = 60000.0
DELTA_M = 1.05
DT_DEFAULT = 0.1
DS_DEFAULT = 1.0
ENERGY_NORM_EPS = 1e-3
P_AUX_DEFAULT_W = 0.0
TORQUE_BLEND_DEFAULT = 0.28
REFERENCE_TORQUE_BLEND = TORQUE_BLEND_DEFAULT


STYLE_PROFILES = {
    "eco": {
        "speed_scale": 0.90, # 经济模式下的目标速度是限速的90%，鼓励更温和的驾驶
        "accel_limit": 1.0, # 加速限制较低，鼓励缓慢加速以节省能量
        "brake_limit": 1.2, # 制动限制较低，鼓励提前减速以利用动能回收
        "throttle_bias": 0.30, # 油门偏置较低，鼓励更早地松开油门以节省能量
        "energy_scale": 1.0, # 能耗缩放因子
        "speed_tol_ms": 0.20, # 速度容差 (m/s)，最大允许的误差
        "dist_tol_m": 0.30, # 距离容差 (m)
        "accel_tol_ms2": 0.25, # 加速度容差 (m/s^2)
        "terminal_time_weight": 4.0,
        "hard_violation_scale": 2.0,
        "terminal_track_fail_weight": 20.0,
        "one_hot": [1, 0, 0], # 驾驶风格的独热编码表示
    },
    "normal": {
        "speed_scale": 1.00, # 正常模式下的目标速度是限速的100%，平衡驾驶体验和能效
        "accel_limit": 1.8, # 加速限制中等，提供舒适的驾驶体验
        "brake_limit": 2.5, # 制动限制中等，确保安全性和舒适性
        "throttle_bias": 0.50, # 油门偏置中等，提供平衡的驾驶感受
        "energy_scale": 0.8, # 能耗缩放因子
        "speed_tol_ms": 0.25, # 速度容差 (m/s)，最大允许的误差
        "dist_tol_m": 0.40, # 距离容差 (m)
        "accel_tol_ms2": 0.35, # 加速度容差 (m/s^2)
        "terminal_time_weight": 5.0,
        "hard_violation_scale": 2.0,
        "terminal_track_fail_weight": 20.0,
        "one_hot": [0, 1, 0], # 驾驶风格的独热编码表示
    },
    "sport": {
        "speed_scale": 1.10,
        "accel_limit": 3.0,
        "brake_limit": 4.0,
        "throttle_bias": 0.70,
        "energy_scale": 0.5,
        "speed_tol_ms": 0.16,
        "dist_tol_m": 0.40,
        "accel_tol_ms2": 0.40,
        "terminal_time_weight": 5.0,
        "hard_violation_scale": 1.2,
        "terminal_track_fail_weight": 65.0,
        "one_hot": [0, 0, 1],
    },
}


def load_efficiency_map(csv_path: str = "sys_eff_pivot.csv") -> dict: # 加载电机效率表，CSV文件格式要求第一列为扭矩，第一行为转速，单元格为效率百分比
    df = pd.read_csv(csv_path, index_col=0)
    torques = df.index.values.astype(float) # 扭矩值数组
    speeds = df.columns.values.astype(float) # 转速值数组
    eff = df.values.astype(float) / 100.0
    return {"torques": torques, "speeds": speeds, "eff": eff}


def query_efficiency(eff_map: dict, torque: float, speed_rpm: float) -> float: # 基于扭矩和转速查询电机效率，使用二维插值方法，超出范围时进行边界处理，缺失数据使用默认效率0.5
    torques = eff_map["torques"]
    speeds = eff_map["speeds"]
    eff = eff_map["eff"]

    t = np.clip(torque, torques[0], torques[-1])
    s = np.clip(speed_rpm, speeds[0], speeds[-1])

    ti = np.searchsorted(torques, t) - 1
    si = np.searchsorted(speeds, s) - 1
    ti = np.clip(ti, 0, len(torques) - 2)
    si = np.clip(si, 0, len(speeds) - 2)

    dt = (t - torques[ti]) / (torques[ti + 1] - torques[ti] + 1e-12)
    ds = (s - speeds[si]) / (speeds[si + 1] - speeds[si] + 1e-12)
    dt = np.clip(dt, 0.0, 1.0)
    ds = np.clip(ds, 0.0, 1.0)

    e00 = eff[ti, si] if not np.isnan(eff[ti, si]) else 0.5
    e01 = eff[ti, si + 1] if not np.isnan(eff[ti, si + 1]) else 0.5
    e10 = eff[ti + 1, si] if not np.isnan(eff[ti + 1, si]) else 0.5
    e11 = eff[ti + 1, si + 1] if not np.isnan(eff[ti + 1, si + 1]) else 0.5
    eta = (
        (1 - dt) * (1 - ds) * e00
        + (1 - dt) * ds * e01
        + dt * (1 - ds) * e10
        + dt * ds * e11
    )
    return float(np.clip(eta, 0.01, 0.99))


def get_optimal_torque_delta(eff_map: Optional[dict], speed_rpm: float, current_torque: float) -> float:
    if eff_map is None:
        return 0.0
    torques = eff_map["torques"]
    speeds = eff_map["speeds"]
    eff = eff_map["eff"]
    
    s = np.clip(speed_rpm, speeds[0], speeds[-1])
    diffs = np.abs(speeds - s)
    closest_si = np.argmin(diffs)
    col_eff = np.nan_to_num(eff[:, closest_si], nan=0.0)
    
    mask = torques >= 0.0 if current_torque >= 0.0 else torques <= 0.0
    valid_effs = np.where(mask, col_eff, -1.0)
    best_idx = np.argmax(valid_effs)
    optimal_torque = torques[best_idx]
    
    return float(optimal_torque - current_torque)


def vehicle_speed_to_motor_speed_rpm(vehicle_speed_mps: float) -> float:
    wheel_omega_radps = vehicle_speed_mps / WHEEL_RADIUS
    motor_omega_radps = wheel_omega_radps * GEAR_RATIO
    return motor_omega_radps * 60.0 / (2.0 * math.pi)


def power_limited_torque(max_power_w: float, motor_speed_rpm: float) -> float:
    if motor_speed_rpm <= 1e-6:
        return float("inf")

    motor_omega_radps = motor_speed_rpm * 2.0 * math.pi / 60.0
    return max_power_w / motor_omega_radps


def limit_motor_torque(motor_torque_cmd: float, motor_speed_rpm: float) -> float:
    drive_torque_limit = MOTOR_MAX_TORQUE
    regen_torque_limit = MOTOR_MAX_REGEN_TORQUE

    if motor_speed_rpm > MOTOR_BASE_SPEED:
        drive_torque_limit = min(
            drive_torque_limit,
            power_limited_torque(MOTOR_MAX_POWER_W, motor_speed_rpm),
        )
        regen_torque_limit = min(
            regen_torque_limit,
            power_limited_torque(MOTOR_MAX_REGEN_POWER_W, motor_speed_rpm),
        )

    return float(np.clip(motor_torque_cmd, -regen_torque_limit, drive_torque_limit))


def driveline_efficiency_for_torque(motor_torque: float) -> float:
    if motor_torque >= 0.0:
        return DRIVELINE_EFFICIENCY
    return 1.0 / max(DRIVELINE_EFFICIENCY, 1e-6)


def calc_battery_power_w(
    motor_torque: float,
    motor_rpm: float,
    eta: float,
    p_aux_w: float = P_AUX_DEFAULT_W,
) -> float:
    p_motor = motor_torque * motor_rpm * 2.0 * math.pi / 60.0
    eta_clip = max(eta, 0.1)
    if p_motor >= 0.0:
        return p_motor / eta_clip + p_aux_w
    return p_motor * eta_clip + p_aux_w


def update_soc(
    soc: float,
    battery_power_w: float,
    dt: float,
    capacity_wh: float = BATTERY_CAPACITY_WH,
    soc_min: float = 0.0,
    soc_max: float = 1.0,
) -> float:
    energy_wh = battery_power_w * dt / 3600.0
    delta_soc = energy_wh / max(capacity_wh, 1e-8)
    next_soc = soc - delta_soc
    return float(np.clip(next_soc, soc_min, soc_max))


def compute_battery_power(motor_torque: float, motor_rpm: float, eta: float) -> float: # 计算电池功率，根据电机扭矩、转速和效率
    return calc_battery_power_w(motor_torque, motor_rpm, eta, p_aux_w=0.0)


def compute_loss_energy( # 计算损失能量，基于当前速度、加速度、坡度、效率和距离增量，适用于评估实际跟踪情况下的能量消耗，包括慢速损失
    speed: float,
    accel: float,
    grade: float,
    eta: float,
    ds: float,
) -> float:
    wheel_force = ( # 计算车轮受力，包含加速、坡度、滚阻和空气阻力等因素
        DELTA_M * VEHICLE_MASS * accel
        + VEHICLE_MASS * GRAVITY * ROLL_RESIST * math.cos(grade)
        + VEHICLE_MASS * GRAVITY * math.sin(grade)
        + 0.5 * AIR_DENSITY * DRAG_COEFF * FRONTAL_AREA * speed ** 2
    )
    eta_clip = max(eta, 0.1)
    if wheel_force >= 0.0:
        loss_coeff = max(0.0, 1.0 / eta_clip - 1.0)
    else:
        loss_coeff = max(0.0, 1.0 - eta_clip)
    return abs(wheel_force) * loss_coeff * ds / 3600.0 # 损失能量 = 受力 * 损失系数 * 距离增量，单位为Wh（也就是单位距离的能量损失）


def step_longitudinal_dynamics(
    speed_prev: float,
    motor_torque_cmd: float,
    grade: float,
    ds: float,
    eff_map: Optional[dict],
    soc_prev: float,
    p_aux_w: float = P_AUX_DEFAULT_W,
    drag_coeff_scale: float = 1.0,
    roll_resist_scale: float = 1.0,
) -> dict:
    motor_speed_now_rpm = vehicle_speed_to_motor_speed_rpm(speed_prev)
    f_roll = VEHICLE_MASS * GRAVITY * (ROLL_RESIST * roll_resist_scale) * math.cos(grade)
    f_grade = VEHICLE_MASS * GRAVITY * math.sin(grade)
    f_air = 0.5 * AIR_DENSITY * (DRAG_COEFF * drag_coeff_scale) * FRONTAL_AREA * speed_prev ** 2

    # 先用起点转速做一次粗限幅，预测区间均速，再用均速做最终恒功率限幅。
    motor_torque_pre = limit_motor_torque(motor_torque_cmd, motor_speed_now_rpm)
    driveline_eff_pre = driveline_efficiency_for_torque(motor_torque_pre)
    f_drive_pre = motor_torque_pre * GEAR_RATIO * driveline_eff_pre / WHEEL_RADIUS
    f_net_pre = f_drive_pre - f_roll - f_grade - f_air
    accel_pre = f_net_pre / (DELTA_M * VEHICLE_MASS)
    v_guess_sq = max(speed_prev ** 2 + 2.0 * accel_pre * ds, 0.25)
    v_guess = math.sqrt(v_guess_sq)
    v_avg_guess = max(0.5 * (speed_prev + v_guess), 0.5)

    motor_speed_limit_rpm = vehicle_speed_to_motor_speed_rpm(v_avg_guess)
    motor_torque = limit_motor_torque(motor_torque_cmd, motor_speed_limit_rpm)
    driveline_eff = driveline_efficiency_for_torque(motor_torque)
    f_drive = motor_torque * GEAR_RATIO * driveline_eff / WHEEL_RADIUS
    f_net = f_drive - f_roll - f_grade - f_air

    accel = f_net / (DELTA_M * VEHICLE_MASS)
    v_new_sq = max(speed_prev ** 2 + 2.0 * accel * ds, 0.25)
    v_new = math.sqrt(v_new_sq)
    v_avg = max(0.5 * (speed_prev + v_new), 0.5)
    dt_k = ds / v_avg

    motor_speed_avg_rpm = vehicle_speed_to_motor_speed_rpm(v_avg)
    motor_speed_next_rpm = vehicle_speed_to_motor_speed_rpm(v_new)
    eta = query_efficiency(eff_map, motor_torque, motor_speed_avg_rpm) if eff_map is not None else 0.85
    p_bat = calc_battery_power_w(motor_torque, motor_speed_avg_rpm, eta, p_aux_w=p_aux_w)
    energy_step = p_bat * dt_k / 3600.0
    soc_next = update_soc(soc_prev, p_bat, dt_k)
    cmp_energy_step = compute_loss_energy(v_avg, accel, grade, eta, ds)

    return {
        "motor_torque": motor_torque,
        "motor_speed_rpm": motor_speed_next_rpm,
        "motor_speed_avg_rpm": motor_speed_avg_rpm,
        "eta": eta,
        "battery_power_w": p_bat,
        "energy_step": energy_step,
        "cmp_energy_step": cmp_energy_step,
        "soc_next": soc_next,
        "speed_next": v_new,
        "speed_avg": v_avg,
        "dt": dt_k,
        "accel": accel,
        "f_roll": f_roll,
        "f_grade": f_grade,
        "f_air": f_air,
        "f_drive": f_drive,
        "f_net": f_net,
    }


def compute_saving_ratio( # 计算节省率，基于参考能量密度和实际能量密度，使用归一化差值方法，避免除零错误，当参考能量密度非常小时返回0
    ref_energy_density: float,
    agent_energy_density: float,
    eps: float = ENERGY_NORM_EPS, # 能量密度归一化中的小常数，避免除零错误
) -> float:
    if ref_energy_density <= eps:
        return 0.0
    return float((ref_energy_density - agent_energy_density) / (ref_energy_density + eps)) # 基于轮端受力的能量密度节省率计算方法，考虑了参考能量密度和实际能量密度之间的相对差异，提供了一个更稳定和有意义的节省率指标


def generate_road( # 生成随机路况，包含总距离、采样间隔、坡度特征、弯道特征等参数，返回包含路况信息的字典
    total_dist: float = 3000.0,
    ds: float = DS_DEFAULT,
    grade_sigma: float = 0.03,
    curve_prob: float = 0.15,
    curve_speed_range: Tuple[float, float] = (8.0, 25.0),
    urban_prob: float = 0.08,
    urban_speed_range: Tuple[float, float] = (10.0, 20.0),
    stop_prob: float = 0.03,
    stop_speed_range: Tuple[float, float] = (0.5, 1.2),
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    n = int(total_dist / ds)
    s = np.arange(n) * ds

    raw_grade = rng.randn(n) * grade_sigma
    kernel = np.ones(50) / 50
    grade = np.convolve(raw_grade, kernel, mode="same")

    speed_limit = np.full(n, 33.3)
    urban_mask = np.zeros(n, dtype=np.float32)
    stop_mask = np.zeros(n, dtype=np.float32)
    in_curve = False
    curve_speed = 33.3
    curve_remaining = 0
    for i in range(n):
        if not in_curve and rng.rand() < curve_prob * ds / 100.0:
            in_curve = True
            curve_speed = rng.uniform(*curve_speed_range)
            curve_remaining = rng.randint(30, 120)
        if in_curve:
            speed_limit[i] = curve_speed
            curve_remaining -= 1
            if curve_remaining <= 0:
                in_curve = False

    i = 40
    while i < n - 60:
        if rng.rand() < urban_prob * ds / 100.0:
            urban_len = rng.randint(80, 220)
            urban_speed = rng.uniform(*urban_speed_range)
            end = min(n, i + urban_len)
            speed_limit[i:end] = np.minimum(speed_limit[i:end], urban_speed)
            urban_mask[i:end] = 1.0
            i = end + rng.randint(30, 100)
        else:
            i += 1

    i = 80
    while i < n - 120:
        if rng.rand() < stop_prob * ds / 100.0:
            stop_len = rng.randint(25, 45)
            crawl_speed = rng.uniform(*stop_speed_range)
            exit_len = rng.randint(35, 90)
            exit_speed = rng.uniform(6.0, 14.0)

            stop_end = min(n, i + stop_len)
            exit_end = min(n, stop_end + exit_len)
            speed_limit[i:stop_end] = np.minimum(speed_limit[i:stop_end], crawl_speed)
            speed_limit[stop_end:exit_end] = np.minimum(speed_limit[stop_end:exit_end], exit_speed)
            stop_mask[i:stop_end] = 1.0
            urban_mask[stop_end:exit_end] = 1.0
            i = exit_end + rng.randint(80, 180)
        else:
            i += 1

    curvature = np.where(speed_limit < 30.0, 1.0 / (speed_limit + 1e-6), 0.0)
    return {
        "s": s,
        "grade": grade,
        "speed_limit": speed_limit,
        "curvature": curvature,
        "urban_mask": urban_mask,
        "stop_mask": stop_mask,
        "n": n,
        "ds": ds,
    }


def generate_reference_trajectory( # 生成参考轨迹，基于路况信息和驾驶风格参数，计算每个采样点的参考速度、扭矩、时间、能量等指标，返回包含参考轨迹信息的字典
    road: dict,
    style: str = "normal",
    eff_map: Optional[dict] = None,
) -> dict:
    n = road["n"]
    ds = road["ds"]
    sp = STYLE_PROFILES[style]

    ref_speed = np.zeros(n)
    ref_torque = np.zeros(n)
    ref_time = np.zeros(n)
    ref_energy = np.zeros(n)
    ref_cmp_energy = np.zeros(n)
    ref_accel = np.zeros(n)

    ref_speed[0] = road["speed_limit"][0] * sp["speed_scale"] * 0.8
    accel_lim = sp["accel_limit"]
    brake_lim = sp["brake_limit"]
    ref_soc = 1.0

    for k in range(1, n):
        local_target_v = road["speed_limit"][k] * sp["speed_scale"]
        v_prev = ref_speed[k - 1]
        lookahead_m = max(80.0, v_prev * 4.0, v_prev ** 2 / max(2.0 * brake_lim, 1e-6))
        lookahead_steps = min(n - k - 1, max(1, int(lookahead_m / max(ds, 1e-6))))
        target_v = local_target_v
        for h in range(1, lookahead_steps + 1):
            j = k + h
            future_limit = road["speed_limit"][j] * sp["speed_scale"]
            if future_limit >= target_v:
                continue
            dist_to_event = h * ds
            feasible_v = math.sqrt(max(future_limit ** 2 + 2.0 * brake_lim * dist_to_event, 0.0))
            target_v = min(target_v, feasible_v)

        accel_gain = 0.5 + 0.4 * sp["throttle_bias"]
        accel_desired = np.clip(accel_gain * (target_v - v_prev), -brake_lim, accel_lim)
        grade_k = road["grade"][k]
        f_roll = VEHICLE_MASS * GRAVITY * ROLL_RESIST * math.cos(grade_k)
        f_grade = VEHICLE_MASS * GRAVITY * math.sin(grade_k)
        f_air = 0.5 * AIR_DENSITY * DRAG_COEFF * FRONTAL_AREA * v_prev ** 2
        f_accel = DELTA_M * VEHICLE_MASS * accel_desired
        f_total = f_roll + f_grade + f_air + f_accel
        motor_torque_cmd_raw = f_total * WHEEL_RADIUS / (GEAR_RATIO * DRIVELINE_EFFICIENCY)
        # 参考驾驶员也做扭矩平滑，避免起步/前段出现不合理的突跳。
        motor_torque_cmd = (
            (1.0 - REFERENCE_TORQUE_BLEND) * ref_torque[k - 1]
            + REFERENCE_TORQUE_BLEND * motor_torque_cmd_raw
        )

        dyn = step_longitudinal_dynamics(
            speed_prev=v_prev,
            motor_torque_cmd=motor_torque_cmd,
            grade=grade_k,
            ds=ds,
            eff_map=eff_map,
            soc_prev=ref_soc,
            p_aux_w=P_AUX_DEFAULT_W,
        )

        ref_speed[k] = dyn["speed_next"]
        ref_accel[k] = dyn["accel"]
        ref_time[k] = ref_time[k - 1] + dyn["dt"]
        ref_torque[k] = dyn["motor_torque"]
        ref_energy[k] = dyn["energy_step"]
        ref_cmp_energy[k] = dyn["cmp_energy_step"]
        ref_soc = dyn["soc_next"]

    ref_dist = np.arange(n) * ds

    return {
        "ref_speed": ref_speed,
        "ref_dist": ref_dist,
        "ref_torque": ref_torque,
        "ref_time": ref_time,
        "ref_accel": ref_accel,
        "ref_energy_per_ds": ref_energy,
        "ref_cmp_energy_per_ds": ref_cmp_energy,
        "ref_total_energy": float(np.sum(ref_energy)),
        "ref_total_cmp_energy": float(np.sum(ref_cmp_energy)),
    }


class MotorEnv: # 电动汽车电机环境类，包含状态观测、动作空间、奖励计算等核心逻辑，支持不同驾驶风格和模式的训练
    OBS_DIM = 29 # 状态观测维度，维数+1，加入了“最优扭矩偏移（Optimal Torque Delta）”这个稳定的效率牵引特征
    ACT_DIM = 1 # 动作空间维度，表示电机扭矩的控制量
    CONSTRAINT_NAMES = ["speed", "distance", "smoothness", "projection", "window_energy"] # 约束名称列表，用于评估不同类型的约束违反程度

    def __init__( # 环境初始化函数，接受路况信息、参考轨迹、驾驶风格、模式、效率表等参数，设置环境的初始状态和参数
        self,
        road: dict,
        ref: dict,
        style: str = "normal",
        mode: str = "track",
        eff_map: Optional[dict] = None,
        energy_weight: float = 0.0,
        residual_torque_scale: float = 6.0,
        preview_steps: int = 5,
        window_size: int = 10,
        projection_horizon: int = 5,
        domain_randomization: Optional[dict] = None,
    ):
        self.road = road
        self.ref = ref
        self.style = style
        self.mode = mode
        self.sp = STYLE_PROFILES[style]
        self.eff_map = eff_map
        self.energy_weight = energy_weight
        self.preview_steps = preview_steps
        self.window_size = window_size
        self.projection_horizon = projection_horizon

        self.n = road["n"]
        self.ds = road["ds"]
        self.ref_loss_scale = max(
            float(np.mean(np.abs(self.ref.get("ref_cmp_energy_per_ds", np.zeros(self.n))))),
            ENERGY_NORM_EPS,
        )
        self.residual_torque_scale = residual_torque_scale
        self.speed_tol = self.sp["speed_tol_ms"]
        self.dist_tol = self.sp["dist_tol_m"]
        self.accel_tol = self.sp["accel_tol_ms2"]
        self.smooth_tol = 8.0
        self.window_energy_tol = 0.3
        self.torque_blend = TORQUE_BLEND_DEFAULT
        self.domain_randomization = domain_randomization or {}
        self.drag_coeff_scale = float(self.domain_randomization.get("drag_coeff_scale", 1.0))
        self.roll_resist_scale = float(self.domain_randomization.get("roll_resist_scale", 1.0))

        self.k = 0
        self.v = 0.0
        self.soc = 1.0
        self.total_energy_agent = 0.0
        self.total_cmp_energy_agent = 0.0
        self.total_time_agent = 0.0
        self.agent_dist = 0.0
        self.prev_torque = 0.0
        self.prev_action = np.zeros(self.ACT_DIM)
        self.energy_history: List[float] = []
        self.cmp_energy_history: List[float] = []
        self.speed_history: List[float] = []
        self.dist_history: List[float] = []
        self.torque_history: List[float] = []
        self.action_history: List[np.ndarray] = []
        self.time_history: List[float] = []
        self.cost_history = {name: [] for name in self.CONSTRAINT_NAMES}
        self.prev_loss_delta = 0.0
        self.prev_window_cmp_gap = 0.0
        self._compute_obs_dim()

    def _compute_obs_dim(self):
        self.OBS_DIM = len(self._build_obs())

    def reset(self) -> np.ndarray: # 环境重置函数，初始化环境状态，包括时间步、速度、SOC、能量统计等，并返回初始观测
        self.k = 0
        self.v = self.ref["ref_speed"][0] * (0.995 + 0.01 * np.random.rand())
        self.soc = 1.0
        self.total_energy_agent = 0.0
        self.total_cmp_energy_agent = 0.0
        self.total_time_agent = 0.0
        self.agent_dist = 0.0
        self.prev_torque = self.ref["ref_torque"][0]
        self.prev_action = np.zeros(self.ACT_DIM)
        self.energy_history = []
        self.cmp_energy_history = []
        self.speed_history = [self.v]
        self.dist_history = [0.0]
        self.torque_history = [self.prev_torque]
        self.action_history = []
        self.time_history = [0.0]
        self.cost_history = {name: [] for name in self.CONSTRAINT_NAMES}
        self.prev_loss_delta = 0.0
        self.prev_window_cmp_gap = 0.0
        return self._build_obs()

    def step(self, action_raw: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]: # 环境步进函数，根据输入动作更新环境状态，计算奖励和终止条件，并返回新的观测、奖励、终止标志和辅助信息
        action = np.clip(action_raw, -1.0, 1.0) # 动作裁剪，确保输入的残差扭矩调整在合理范围内，避免过大或过小的调整导致不稳定行为
        k = self.k

        residual_torque = action[0] * self.residual_torque_scale

        ref_v = self.ref["ref_speed"][k]
        ref_torque = self.ref["ref_torque"][k]
        # 直接由智能体来调节动作，不施加第二次外部物理延迟
        grade_k = self.road["grade"][k]
        ref_accel = self.ref["ref_accel"][k] if "ref_accel" in self.ref else 0.0
        motor_torque_cmd = ref_torque + residual_torque

        dyn = step_longitudinal_dynamics(
            speed_prev=self.v,
            motor_torque_cmd=motor_torque_cmd,
            grade=grade_k,
            ds=self.ds,
            eff_map=self.eff_map,
            soc_prev=self.soc,
            p_aux_w=P_AUX_DEFAULT_W,
            drag_coeff_scale=self.drag_coeff_scale,
            roll_resist_scale=self.roll_resist_scale,
        )
        motor_torque = dyn["motor_torque"]
        f_roll = dyn["f_roll"]
        f_grade = dyn["f_grade"]
        f_air = dyn["f_air"]
        f_motor = dyn["f_drive"]
        f_net = dyn["f_net"]
        accel = dyn["accel"]
        v_new = dyn["speed_next"]
        v_avg = dyn["speed_avg"]
        dt_k = dyn["dt"]
        motor_rpm = dyn["motor_speed_rpm"]
        eta = dyn["eta"]
        p_bat = dyn["battery_power_w"]
        energy_step = dyn["energy_step"]
        cmp_energy_step = dyn["cmp_energy_step"]

        ref_energy_k = self.ref["ref_energy_per_ds"][k]
        ref_cmp_energy_k = self.ref["ref_cmp_energy_per_ds"][k]
        self.total_energy_agent += energy_step
        self.total_cmp_energy_agent += cmp_energy_step
        self.total_time_agent += dt_k
        self.agent_dist += v_avg * dt_k
        self.soc = dyn["soc_next"]

        self.energy_history.append(energy_step)
        self.cmp_energy_history.append(cmp_energy_step)
        self.speed_history.append(v_new)
        self.dist_history.append(self.agent_dist)
        self.torque_history.append(motor_torque)
        self.action_history.append(action.copy())
        self.time_history.append(self.total_time_agent)

        speed_err = abs(v_new - ref_v)
        ref_dist_k = self.ref["ref_dist"][min(k + 1, self.n - 1)]
        ref_time_k = self.ref.get("ref_time", np.zeros(self.n))[min(k + 1, self.n - 1)]
        dist_err = abs(self.agent_dist - ref_dist_k)
        time_err_s = self.total_time_agent - ref_time_k
        torque_diff = abs(motor_torque - self.prev_torque)
        accel_err = abs(accel - ref_accel)

        cost_speed = max(0.0, speed_err / self.speed_tol - 1.0)
        cost_distance = max(0.0, dist_err / self.dist_tol - 1.0)
        cost_smooth = max(0.0, torque_diff / self.smooth_tol - 1.0)

        cost_projection = 0.0
        if k + self.projection_horizon < self.n: # 投影约束计算，基于当前加速度预测未来速度，并与未来参考速度进行比较，评估未来一段时间内的跟踪性能，鼓励在更长的时间范围内保持良好的跟踪
            for h in range(1, self.projection_horizon + 1): # 未来h步的投影计算，逐步预测未来速度，并与对应的参考速度进行比较，计算投影约束的违反程度
                future_ref_v = self.ref["ref_speed"][min(k + h, self.n - 1)]
                v_proj = v_new + accel * h * dt_k
                cost_projection += max(
                    0.0,
                    abs(v_proj - future_ref_v) / self.speed_tol - 1.5,
                ) / self.projection_horizon

        cost_window_energy = 0.0
        if len(self.cmp_energy_history) >= self.window_size:
            window_agent = float(sum(self.cmp_energy_history[-self.window_size:]))
            ref_window = self.ref["ref_cmp_energy_per_ds"][max(0, k - self.window_size + 1):k + 1]
            if len(ref_window) == self.window_size:
                ref_window_sum = float(sum(ref_window))
                if ref_window_sum > ENERGY_NORM_EPS * self.window_size:
                    ratio = window_agent / (ref_window_sum + ENERGY_NORM_EPS * self.window_size)
                    cost_window_energy = max(0.0, ratio - (1.0 + self.window_energy_tol))
                    self.prev_window_cmp_gap = (
                        window_agent - ref_window_sum
                    ) / (abs(ref_window_sum) + ENERGY_NORM_EPS * self.window_size)
                else:
                    self.prev_window_cmp_gap = 0.0
            else:
                self.prev_window_cmp_gap = 0.0
        else:
            self.prev_window_cmp_gap = 0.0

        costs = {
            "speed": cost_speed,
            "distance": cost_distance,
            "smoothness": cost_smooth,
            "projection": cost_projection,
            "window_energy": cost_window_energy,
        }
        for name in self.CONSTRAINT_NAMES:
            self.cost_history[name].append(costs[name])

        reward, reward_info = self._compute_reward( # 计算奖励函数，基于当前状态、参考轨迹、误差指标、能量消耗等多维因素，支持跟踪模式和优化模式的奖励计算逻辑，返回奖励值和详细的奖励信息字典
            v=v_new,
            ref_v=ref_v,
            speed_err=speed_err,
            dist_err=dist_err,
            accel=accel,
            ref_accel=ref_accel,
            accel_err=accel_err,
            torque=motor_torque,
            torque_diff=torque_diff,
            energy_step=energy_step,
            ref_energy_k=ref_energy_k,
            accounted_energy=cmp_energy_step,
            ref_cmp_energy_k=ref_cmp_energy_k,
            eta=eta,
            k=k,
            ref_torque=ref_torque,
            time_err_s=time_err_s,
            action_diff=abs(action[0] - self.prev_action[0]),
        )
        self.prev_loss_delta = reward_info.get("loss_delta", 0.0)
        self.prev_torque = motor_torque
        self.prev_action = action.copy()
        self.v = v_new
        self.k += 1

        violation_scale = float(self.sp.get("hard_violation_scale", 2.0))
        hard_violation = (
            speed_err > self.speed_tol * violation_scale
            or dist_err > self.dist_tol * violation_scale
        ) # 严重违反约束的判断条件，基于速度误差和距离误差是否超过风格化容差阈值，作为环境终止和奖励惩罚的重要依据
        done = (self.k >= self.n - 1) or (self.soc <= 0.01)
        terminal_reward = 0.0
        if done and self.k >= self.n - 2:
            terminal_reward = self._compute_terminal_reward()
            reward += terminal_reward
        elif hard_violation:
            reward -= 50.0

        info = { # 环境步进返回的辅助信息字典，包含当前时间步、速度、参考速度、误差指标、扭矩、效率、能量消耗、奖励信息等多维数据，供训练过程中的分析和调试使用
            "step": k,
            "speed": v_new,
            "ref_speed": ref_v,
            "speed_err": speed_err,
            "dist_err": dist_err,
            "time_err_s": time_err_s,
            "ref_torque": ref_torque,
            "motor_torque_cmd": motor_torque_cmd,
            "motor_torque": motor_torque,
            "motor_rpm": motor_rpm,
            "accel": accel,
            "ref_accel": ref_accel,
            "accel_err": accel_err,
            "eta": eta,
            "energy_step": energy_step,
            "cmp_energy_step": cmp_energy_step,
            "accounted_energy": cmp_energy_step,
            "ref_energy_k": ref_energy_k,
            "ref_cmp_energy_k": ref_cmp_energy_k,
            "soc": self.soc,
            "costs": costs,
            "reward_info": reward_info,
            "terminal_reward": terminal_reward,
            "total_energy_agent": self.total_energy_agent,
            "total_cmp_energy_agent": self.total_cmp_energy_agent,
            "total_time_agent": self.total_time_agent,
            "agent_dist": self.agent_dist,
            "residual_torque": residual_torque,
            "drag_coeff_scale": self.drag_coeff_scale,
            "roll_resist_scale": self.roll_resist_scale,
            "hard_violation": hard_violation,
        }
        return self._build_obs(), float(reward), done, info

    def _compute_reward( # 计算奖励函数，基于当前状态、参考轨迹、误差指标、能量消耗等多维因素，支持跟踪模式和优化模式的奖励计算逻辑，返回奖励值和详细的奖励信息字典
        self,
        v,
        ref_v,
        speed_err,
        dist_err,
        accel,
        ref_accel,
        accel_err,
        torque,
        torque_diff,
        energy_step,
        ref_energy_k,
        accounted_energy,
        ref_cmp_energy_k,
        eta,
        k,
        ref_torque,
        time_err_s,
        action_diff: float = 0.0,
    ) -> Tuple[float, dict]:
        track_penalty = ( # 跟踪误差的综合惩罚项，基于速度误差、距离误差和加速度误差的平方和，使用不同的权重进行组合，鼓励同时满足多个跟踪指标
            8.0 * (speed_err / self.speed_tol) ** 2
            + 4.0 * (dist_err / self.dist_tol) ** 2
            + 2.0 * (accel_err / self.accel_tol) ** 2
        )

        if self.mode == "track":
            smooth_penalty = -0.10 * (torque_diff / self.smooth_tol) ** 2
            reward = -track_penalty + smooth_penalty
            info = {
                "track_penalty": -track_penalty,
                "speed_penalty": -8.0 * (speed_err / self.speed_tol) ** 2,
                "dist_penalty": -4.0 * (dist_err / self.dist_tol) ** 2,
                "accel_penalty": -2.0 * (accel_err / self.accel_tol) ** 2,
                "smooth_penalty": smooth_penalty,
            }
            return float(reward), info

        tracking_ok = speed_err <= self.speed_tol and dist_err <= self.dist_tol and accel_err <= self.accel_tol
        w_c = self.energy_weight * self.sp["energy_scale"] * 200.0
        w_u = 0.08
        ref_loss_density = ref_cmp_energy_k / max(self.ds, 1e-8)
        agent_loss_density = accounted_energy / max(self.ds, 1e-8)
        loss_delta = agent_loss_density - ref_loss_density
        torque_delta_penalty = w_u * (torque_diff / max(self.residual_torque_scale, 1e-8)) ** 2
        
        # 引入动作平滑度惩罚，防止踩放油门（hunting）
        action_jerk_penalty = 2.0 * action_diff ** 2

        r_energy = -w_c * loss_delta - torque_delta_penalty - action_jerk_penalty

        beta = 0.05
        g_track = math.exp(-beta * track_penalty)
        reward = -track_penalty + g_track * r_energy

        info = { # 优化模式下的奖励信息字典，包含跟踪状态、跟踪惩罚、各项误差的惩罚等详细数据，供训练过程中的分析和调试使用
            "tracking_ok_step": 1.0 if tracking_ok else 0.0,
            "track_penalty": -track_penalty,
            "speed_penalty": -8.0 * (speed_err / self.speed_tol) ** 2,
            "dist_penalty": -4.0 * (dist_err / self.dist_tol) ** 2,
            "accel_penalty": -2.0 * (accel_err / self.accel_tol) ** 2,
            "loss_delta": loss_delta,
            "ref_loss_density": ref_loss_density,
            "agent_loss_density": agent_loss_density,
            "torque_delta_penalty": -torque_delta_penalty,
            "agent_eta": eta,
            "g_track": g_track,
            "r_energy_raw": r_energy,
        }
        return float(reward), info

    def _compute_terminal_reward(self) -> float: # 计算终止奖励，基于整个轨迹的跟踪性能和能量消耗情况，评估最终的跟踪效果和节能效果，提供一个综合性的终止奖励指标
        metrics = self.get_episode_metrics()
        if not metrics.get("tracking_ok", False):
            speed_term = (metrics.get("speed_mae", 0.0) / self.speed_tol) ** 2
            dist_term = (metrics.get("dist_mae", 0.0) / self.dist_tol) ** 2
            fail_weight = float(self.sp.get("terminal_track_fail_weight", 20.0))
            return float(-fail_weight * (speed_term + dist_term))

        ref_total = float(self.ref.get("ref_total_energy", 0.0))
        agent_total = float(self.total_energy_agent)
        total_dist = max(self.n * self.ds, 1e-8)
        reward_saving = compute_saving_ratio(ref_total / total_dist, agent_total / total_dist)

        ref_time_total = (
            self.ref["ref_time"][min(max(self.k - 1, 0), len(self.ref["ref_time"]) - 1)]
            if len(self.ref.get("ref_time", [])) > 0
            else self.total_time_agent
        )
        time_gap_ratio = (self.total_time_agent - ref_time_total) / (ref_time_total + 1e-8)
        time_penalty = float(self.sp.get("terminal_time_weight", 5.0)) * (time_gap_ratio ** 2)
        return float(10.0 * self.sp["energy_scale"] * reward_saving - time_penalty)
 
    def _build_obs(self) -> np.ndarray: # 构建状态观测函数，基于当前环境状态、参考轨迹、驾驶风格等信息，构建一个包含多维特征的状态观测向量，供模型输入使用
        k = self.k
        n = self.n

        ref_v = self.ref["ref_speed"][k]
        ref_dist_k = self.ref["ref_dist"][k]
        dist_err_m = self.agent_dist - ref_dist_k

        v_norm = self.v / 40.0
        soc_val = self.soc
        torque_norm = self.prev_torque / MOTOR_MAX_TORQUE

        ref_v_norm = ref_v / 40.0 # 参考速度归一化，基于一个合理的最大速度值进行归一化处理，确保输入特征在适当的范围内，促进模型的学习和稳定性
        ref_d_norm = self.ref["ref_dist"][k] / (self.n * self.ds)
        ref_t_norm = self.ref["ref_torque"][k] / MOTOR_MAX_TORQUE
        ref_loss_norm = self.ref["ref_cmp_energy_per_ds"][k] / self.ref_loss_scale

        ref_time_k = self.ref.get("ref_time", np.zeros(self.n))[k]
        time_err_s = self.total_time_agent - ref_time_k
        time_err_norm = time_err_s / 5.0  # 归一化：落后5秒记为1.0

        speed_err_norm = (self.v - ref_v) / self.speed_tol
        dist_err_norm = dist_err_m / self.dist_tol
        prev_loss_delta_norm = self.prev_loss_delta / self.ref_loss_scale
        prev_window_gap_norm = self.prev_window_cmp_gap

        hist_act = self.prev_action.tolist()

        preview = []
        for h in range(1, self.preview_steps + 1):
            idx = min(k + h, n - 1)
            preview.append(self.ref["ref_speed"][idx] / 40.0)
            preview.append(self.road["grade"][idx] * 10.0)

        style_oh = self.sp["one_hot"]
        
        agent_rpm_k = vehicle_speed_to_motor_speed_rpm(self.v)
        opt_t_delta = get_optimal_torque_delta(self.eff_map, agent_rpm_k, self.prev_torque)
        opt_t_delta_norm = opt_t_delta / MOTOR_MAX_TORQUE

        action_hints = [
            self.residual_torque_scale / MOTOR_MAX_TORQUE,
            opt_t_delta_norm,
        ]
        progress = k / max(n - 1, 1)

        obs = np.array(
            [
                v_norm,
                soc_val,
                torque_norm,
                ref_v_norm,
                ref_d_norm,
                ref_t_norm,
                ref_loss_norm,
                time_err_norm,
                speed_err_norm,
                dist_err_norm,
                prev_loss_delta_norm,
                prev_window_gap_norm,
                *hist_act,
                *preview,
                *style_oh,
                *action_hints,
                progress,
            ],
            dtype=np.float32,
        )
        return obs

    def get_episode_metrics(self) -> dict: # 获取episode指标函数，计算整个轨迹的跟踪性能和能量消耗指标，为奖励计算和性能评估提供数据支持，只是统计/评估函数，用于汇报与终端奖励计算
        n_valid = min(self.k, self.n)
        ref_speeds = self.ref["ref_speed"][:n_valid]
        agent_speeds = np.array(self.speed_history[:n_valid])
        ref_dist = self.ref["ref_dist"][:n_valid]
        agent_dist = np.array(self.dist_history[:n_valid])
        ref_energy = self.ref["ref_energy_per_ds"][:n_valid]
        agent_energy = np.array(self.energy_history[:n_valid])
        ref_cmp_energy = self.ref.get("ref_cmp_energy_per_ds", ref_energy)[:n_valid]
        agent_cmp_energy = np.array(self.cmp_energy_history[:n_valid])

        speed_mae = float(np.mean(np.abs(agent_speeds - ref_speeds[:len(agent_speeds)]))) if len(agent_speeds) > 0 else 0.0
        speed_rel_bias = (
            float(np.mean(agent_speeds - ref_speeds[:len(agent_speeds)]))
            / (np.mean(ref_speeds[:len(agent_speeds)]) + 1e-8)
            * 100
            if len(agent_speeds) > 0
            else 0.0
        )

        if len(agent_dist) > 0:
            dist_err_series = np.abs(agent_dist - ref_dist[:len(agent_dist)])
            dist_mae = float(np.mean(dist_err_series))
        else:
            dist_mae = 0.0

        ref_total = float(np.sum(ref_energy[:len(agent_energy)]))
        agent_total = float(np.sum(agent_energy))
        saving_total_pct = (ref_total - agent_total) / (abs(ref_total) + 1e-8) * 100 if len(agent_energy) > 0 else 0.0

        ref_epd = ref_total / max(n_valid * self.ds, 1.0)
        agent_epd = agent_total / max(n_valid * self.ds, 1.0)
        saving_epd_pct = (ref_epd - agent_epd) / (abs(ref_epd) + 1e-8) * 100 if n_valid > 0 else 0.0

        ref_cmp_total = float(np.sum(ref_cmp_energy[:len(agent_cmp_energy)]))
        agent_cmp_total = float(np.sum(agent_cmp_energy))
        saving_cmp_total_pct = compute_saving_ratio(
            ref_cmp_total / max(n_valid * self.ds, 1.0),
            agent_cmp_total / max(n_valid * self.ds, 1.0),
        ) * 100 if n_valid > 0 else 0.0

        ref_time_total = self.ref["ref_time"][min(max(n_valid - 1, 0), len(self.ref["ref_time"]) - 1)] if len(self.ref["ref_time"]) > 0 else self.total_time_agent
        time_ratio = self.total_time_agent / (ref_time_total + 1e-8) if ref_time_total > 0 else 1.0
        saving_isochronous_pct = saving_total_pct / max(time_ratio, 0.5)
        saving_net_epd_pct = saving_epd_pct
        saving_net_isochronous_pct = saving_isochronous_pct

        low_speed_mask = ref_speeds[:len(agent_energy)] < 10.0 if len(agent_energy) > 0 else np.array([])
        if len(agent_energy) > 0 and np.any(low_speed_mask):
            low_ref = float(np.sum(ref_energy[:len(agent_energy)][low_speed_mask]))
            low_agent = float(np.sum(agent_energy[low_speed_mask]))
            low_speed_benefit_pct = (low_ref - low_agent) / (abs(low_ref) + 1e-8) * 100
        else:
            low_speed_benefit_pct = 0.0

        half = len(agent_energy) // 2
        front_ref = float(np.sum(ref_energy[:half]))
        front_agent = float(np.sum(agent_energy[:half]))
        back_ref = float(np.sum(ref_energy[half:len(agent_energy)]))
        back_agent = float(np.sum(agent_energy[half:]))
        front_half_saving_pct = (front_ref - front_agent) / (abs(front_ref) + 1e-8) * 100 if half > 0 else 0.0
        back_half_saving_pct = (back_ref - back_agent) / (abs(back_ref) + 1e-8) * 100 if len(agent_energy) > half else 0.0
        front_back_saving_gap_pct = abs(front_half_saving_pct - back_half_saving_pct)

        ws = self.window_size
        worst_window_saving_pct = 100.0
        worst_window_reward_saving_pct = 100.0
        neg_window_count = 0
        neg_reward_window_count = 0
        total_windows = 0
        if len(agent_energy) >= ws:
            for i in range(len(agent_energy) - ws + 1):
                w_agent = float(np.sum(agent_energy[i:i + ws]))
                w_ref = float(np.sum(ref_energy[i:i + ws]))
                w_saving = compute_saving_ratio(
                    w_ref / max(ws * self.ds, 1e-8),
                    w_agent / max(ws * self.ds, 1e-8),
                ) * 100
                worst_window_saving_pct = min(worst_window_saving_pct, w_saving)
                if w_saving < 0:
                    neg_window_count += 1

                if i + ws <= len(agent_cmp_energy):
                    w_agent_reward = float(np.sum(agent_cmp_energy[i:i + ws]))
                    w_ref_reward = float(np.sum(ref_cmp_energy[i:i + ws]))
                    w_reward_saving = compute_saving_ratio(
                        w_ref_reward / max(ws * self.ds, 1e-8),
                        w_agent_reward / max(ws * self.ds, 1e-8),
                    ) * 100
                    worst_window_reward_saving_pct = min(
                        worst_window_reward_saving_pct,
                        w_reward_saving,
                    )
                    if w_reward_saving < 0:
                        neg_reward_window_count += 1
                total_windows += 1

        negative_window_ratio = neg_window_count / max(total_windows, 1)
        negative_window_reward_ratio = neg_reward_window_count / max(total_windows, 1)

        segment_len = max(n_valid // 10, 1)
        worst_seg_speed_mae = 0.0
        worst_seg_dist_mae = 0.0
        for i in range(0, max(len(agent_speeds) - segment_len, 1), segment_len):
            seg_speed = agent_speeds[i:i + segment_len]
            seg_ref_speed = ref_speeds[i:i + segment_len]
            if len(seg_speed) == 0:
                continue
            seg_mae = float(np.mean(np.abs(seg_speed - seg_ref_speed)))
            worst_seg_speed_mae = max(worst_seg_speed_mae, seg_mae)

            seg_agent_dist = agent_dist[i:i + segment_len]
            seg_ref_dist = ref_dist[i:i + segment_len]
            seg_dist = np.abs(seg_agent_dist - seg_ref_dist)
            worst_seg_dist_mae = max(worst_seg_dist_mae, float(np.mean(seg_dist)))

        tracking_ok = speed_mae < self.speed_tol and dist_mae < self.dist_tol
        smooth_ok = bool(float(np.mean(self.cost_history["smoothness"])) < 0.5) if len(self.cost_history["smoothness"]) > 0 else True
        bias_guard_ok = bool(abs(speed_rel_bias) < 5.0)
        dist_rel_bias_pct = (
            float(np.mean(agent_dist - ref_dist[:len(agent_dist)])) / (max(self.n * self.ds, 1.0)) * 100
            if len(agent_dist) > 0
            else 0.0
        )

        return { # episode指标字典，包含总节省百分比、每距离节省百分比、等时节省百分比、低速节省百分比、速度MAE、距离MAE、相对偏差百分比、前后半段节省百分比、最差窗口节省百分比、跟踪状态等多维指标，为训练过程中的分析和评估提供数据支持
            "saving_total_pct": saving_total_pct,
            "saving_epd_pct": saving_epd_pct,
            "saving_net_epd_pct": saving_net_epd_pct,
            "saving_isochronous_pct": saving_isochronous_pct,
            "saving_net_isochronous_pct": saving_net_isochronous_pct,
            "saving_cmp_total_pct": saving_cmp_total_pct,
            "speed_energy_bias_pct": saving_total_pct - saving_cmp_total_pct,
            "low_speed_benefit_pct": low_speed_benefit_pct,
            "speed_mae": speed_mae,
            "dist_mae": dist_mae,
            "speed_rel_bias_pct": speed_rel_bias,
            "dist_rel_bias_pct": dist_rel_bias_pct,
            "front_half_saving_pct": front_half_saving_pct,
            "back_half_saving_pct": back_half_saving_pct,
            "front_back_saving_gap_pct": front_back_saving_gap_pct,
            "worst_window_saving_pct": worst_window_saving_pct,
            "negative_window_ratio": negative_window_ratio,
            "worst_window_reward_saving_pct": worst_window_reward_saving_pct,
            "negative_window_reward_ratio": negative_window_reward_ratio,
            "worst_segment_speed_mae": worst_seg_speed_mae,
            "worst_segment_dist_mae": worst_seg_dist_mae,
            "tracking_ok": tracking_ok,
            "smooth_ok": smooth_ok,
            "bias_guard_ok": bias_guard_ok,
        }
