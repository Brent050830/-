import math

from params import VehicleParams


def vehicle_speed_to_motor_speed_rpm(vehicle_speed_mps: float, params: VehicleParams) -> float:
    """根据车速计算电机转速。"""
    wheel_omega_radps = vehicle_speed_mps / params.wheel_radius_m
    motor_omega_radps = wheel_omega_radps * params.gear_ratio
    return motor_omega_radps * 60.0 / (2.0 * math.pi)


def power_limited_torque(max_power_w: float, motor_speed_rpm: float) -> float:
    """根据功率上限计算当前转速下允许的最大扭矩。"""
    if motor_speed_rpm <= 1e-6:
        return float("inf")

    motor_omega_radps = motor_speed_rpm * 2.0 * math.pi / 60.0
    return max_power_w / motor_omega_radps


def limit_motor_torque(t_cmd_nm: float, motor_speed_rpm: float, params: VehicleParams) -> float:
    """
    电机约束模型：
    1. 低速区采用恒扭矩限制
    2. 超过基速后采用恒功率限制
    3. 正扭矩为驱动，负扭矩为能量回收
    """
    drive_torque_limit = params.motor_max_torque_nm
    regen_torque_limit = params.motor_max_regen_torque_nm

    if motor_speed_rpm > params.motor_base_speed_rpm:
        drive_torque_limit = min(
            drive_torque_limit,
            power_limited_torque(params.motor_max_power_w, motor_speed_rpm),
        )
        regen_torque_limit = min(
            regen_torque_limit,
            power_limited_torque(params.motor_max_regen_power_w, motor_speed_rpm),
        )

    return max(-regen_torque_limit, min(t_cmd_nm, drive_torque_limit))
