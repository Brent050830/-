from dataclasses import dataclass


@dataclass
class VehicleParams:
    """车辆与动力系统参数。"""

    # 整车参数
    mass_kg: float = 1650.0
    wheel_radius_m: float = 0.31
    gear_ratio: float = 9.5
    driveline_efficiency: float = 0.95

    # 阻力参数
    rolling_resistance_coeff: float = 0.012
    air_density_kgpm3: float = 1.225
    drag_coefficient: float = 0.29
    frontal_area_m2: float = 2.2
    gravity_mps2: float = 9.81

    # 电池参数
    battery_capacity_kwh: float = 60.0
    soc_init: float = 0.85
    soc_min: float = 0.10
    soc_max: float = 0.95

    # 电机参数
    motor_max_torque_nm: float = 220.0
    motor_max_regen_torque_nm: float = 100.0
    motor_base_speed_rpm: float = 4000.0
    motor_max_power_w: float = 80_000.0
    motor_max_regen_power_w: float = 40_000.0
    motor_drive_efficiency: float = 0.92
    motor_regen_efficiency: float = 0.70


def default_vehicle_params() -> VehicleParams:
    """返回一组适合演示的默认参数。"""
    return VehicleParams()
