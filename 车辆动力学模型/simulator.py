import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from longitudinal import LongitudinalVehicleModel


@dataclass
class SimulationStepInput:
    """单个仿真步的输入。"""

    phase_name: str
    dt_s: float
    t_cmd_nm: float
    grade: float
    p_aux_w: float


def run_simulation(
    model: LongitudinalVehicleModel,
    step_inputs: Iterable[SimulationStepInput],
) -> list[dict]:
    """按输入序列逐步运行仿真。"""
    records: list[dict] = []

    for step_input in step_inputs:
        result = model.step(
            dt=step_input.dt_s,
            t_cmd_nm=step_input.t_cmd_nm,
            grade=step_input.grade,
            p_aux_w=step_input.p_aux_w,
        )

        record = {
            "phase_name": step_input.phase_name,
            "dt_s": step_input.dt_s,
            "t_cmd_nm": step_input.t_cmd_nm,
            "grade": step_input.grade,
            "p_aux_w": step_input.p_aux_w,
            "time_s": result["time_s"],
            "v_mps": result["v_mps"],
            "v_kph": result["v_mps"] * 3.6,
            "a_mps2": result["a_mps2"],
            "soc": result["soc"],
            "motor_torque_actual_nm": result["motor_torque_actual_nm"],
            "motor_speed_rpm": result["motor_speed_rpm"],
            "battery_power_w": result["battery_power_w"],
            "cumulative_energy_kwh": result["cumulative_energy_kwh"],
            "f_roll_n": result["f_roll_n"],
            "f_aero_n": result["f_aero_n"],
            "f_grade_n": result["f_grade_n"],
            "f_drive_n": result["f_drive_n"],
            "f_net_n": result["f_net_n"],
        }
        records.append(record)

    return records


def save_results_to_csv(records: list[dict], csv_path: Path) -> None:
    """将仿真结果保存到 CSV 文件。"""
    if not records:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
