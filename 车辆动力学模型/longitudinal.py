import math
from dataclasses import dataclass

from battery import calc_battery_power_w, update_soc
from motor import limit_motor_torque, vehicle_speed_to_motor_speed_rpm
from params import VehicleParams


@dataclass
class VehicleState:
    """车辆当前状态。"""

    time_s: float = 0.0
    v_mps: float = 0.0
    a_mps2: float = 0.0
    soc: float = 0.85
    motor_torque_actual_nm: float = 0.0
    motor_speed_rpm: float = 0.0
    battery_power_w: float = 0.0
    cumulative_energy_kwh: float = 0.0


class LongitudinalVehicleModel:
    """纯电动车 1D 纵向动力学最小模型。"""

    def __init__(self, params: VehicleParams):
        self.params = params
        self.state = VehicleState(soc=params.soc_init)

    def reset(self) -> VehicleState:
        """将模型状态恢复到初始值。"""
        self.state = VehicleState(soc=self.params.soc_init)
        return self.state

    def step(self, dt: float, t_cmd_nm: float, grade: float, p_aux_w: float) -> dict:
        """
        单步更新车辆状态。

        参数说明：
        1. dt: 时间步长，单位 s
        2. t_cmd_nm: 电机需求扭矩，单位 Nm
        3. grade: 坡度，采用道路坡度比，例如 5% 坡写成 0.05
        4. p_aux_w: 附件功率，单位 W
        """
        s = self.state
        p = self.params

        # 先由当前车速计算当前电机转速，再进行扭矩限幅。
        motor_speed_now_rpm = vehicle_speed_to_motor_speed_rpm(s.v_mps, p)

        # 若 SOC 已触及边界，则禁止继续放电或继续回收。
        if s.soc <= p.soc_min and t_cmd_nm > 0.0:
            t_cmd_nm = 0.0
        if s.soc >= p.soc_max and t_cmd_nm < 0.0:
            t_cmd_nm = 0.0

        motor_torque_actual_nm = limit_motor_torque(t_cmd_nm, motor_speed_now_rpm, p)

        # 将坡度比转换为坡角，用于计算重力分量。
        theta_rad = math.atan(grade)

        # 滚动阻力：F_roll = C_rr * m * g * cos(theta)
        f_roll_n = p.rolling_resistance_coeff * p.mass_kg * p.gravity_mps2 * math.cos(theta_rad)

        # 空气阻力：F_aero = 0.5 * rho * Cd * A * v^2
        f_aero_n = 0.5 * p.air_density_kgpm3 * p.drag_coefficient * p.frontal_area_m2 * s.v_mps**2

        # 坡度阻力：F_grade = m * g * sin(theta)
        f_grade_n = p.mass_kg * p.gravity_mps2 * math.sin(theta_rad)

        # 驱动力：F_drive = T_motor * i_g * eta / r_w
        f_drive_n = (
            motor_torque_actual_nm * p.gear_ratio * p.driveline_efficiency / p.wheel_radius_m
        )

        # 合力与加速度：F_net = F_drive - F_roll - F_aero - F_grade
        f_net_n = f_drive_n - f_roll_n - f_aero_n - f_grade_n
        a_predict_mps2 = f_net_n / p.mass_kg

        # 车速积分：v_next = max(0, v + a * dt)
        v_next_mps = max(0.0, s.v_mps + a_predict_mps2 * dt)
        a_actual_mps2 = (v_next_mps - s.v_mps) / dt

        # 使用一步内平均车速近似电机平均转速，得到更平滑的电池功率。
        v_avg_mps = 0.5 * (s.v_mps + v_next_mps)
        motor_speed_avg_rpm = vehicle_speed_to_motor_speed_rpm(v_avg_mps, p)
        battery_power_w = calc_battery_power_w(
            motor_torque_actual_nm,
            motor_speed_avg_rpm,
            p_aux_w,
            p,
        )

        # 电池 SOC 更新。
        soc_next = update_soc(s.soc, battery_power_w, dt, p)

        # 这里按净电池能量累计，回收时该值会减小。
        cumulative_energy_kwh = s.cumulative_energy_kwh + battery_power_w * dt / 3_600_000.0

        motor_speed_next_rpm = vehicle_speed_to_motor_speed_rpm(v_next_mps, p)

        self.state = VehicleState(
            time_s=s.time_s + dt,
            v_mps=v_next_mps,
            a_mps2=a_actual_mps2,
            soc=soc_next,
            motor_torque_actual_nm=motor_torque_actual_nm,
            motor_speed_rpm=motor_speed_next_rpm,
            battery_power_w=battery_power_w,
            cumulative_energy_kwh=cumulative_energy_kwh,
        )

        return {
            "time_s": self.state.time_s,
            "v_mps": self.state.v_mps,
            "a_mps2": self.state.a_mps2,
            "soc": self.state.soc,
            "motor_torque_actual_nm": self.state.motor_torque_actual_nm,
            "motor_speed_rpm": self.state.motor_speed_rpm,
            "battery_power_w": self.state.battery_power_w,
            "cumulative_energy_kwh": self.state.cumulative_energy_kwh,
            "f_roll_n": f_roll_n,
            "f_aero_n": f_aero_n,
            "f_grade_n": f_grade_n,
            "f_drive_n": f_drive_n,
            "f_net_n": f_net_n,
        }
