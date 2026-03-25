import math

from params import VehicleParams


def calc_battery_power_w(
    motor_torque_nm: float,
    motor_speed_rpm: float,
    p_aux_w: float,
    params: VehicleParams,
) -> float:
    """
    计算电池功率。

    约定：
    1. 电池功率 > 0 表示电池放电
    2. 电池功率 < 0 表示回收能量进入电池
    """
    motor_omega_radps = motor_speed_rpm * 2.0 * math.pi / 60.0
    motor_mech_power_w = motor_torque_nm * motor_omega_radps

    if motor_mech_power_w >= 0.0:
        battery_power_w = motor_mech_power_w / max(params.motor_drive_efficiency, 1e-6) + p_aux_w
    else:
        battery_power_w = motor_mech_power_w * params.motor_regen_efficiency + p_aux_w

    return battery_power_w


def update_soc(soc: float, battery_power_w: float, dt: float, params: VehicleParams) -> float:
    """根据电池功率更新 SOC。"""
    battery_energy_j = params.battery_capacity_kwh * 3_600_000.0
    delta_soc = battery_power_w * dt / battery_energy_j
    next_soc = soc - delta_soc
    return max(params.soc_min, min(params.soc_max, next_soc))
