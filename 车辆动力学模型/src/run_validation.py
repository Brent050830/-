import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

# 验证脚本只需要保存 PNG，不需要弹出图窗，因此使用无界面后端更稳妥。
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 让脚本可以从 src 目录导入项目根目录下的模型文件。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from longitudinal import LongitudinalVehicleModel
from params import VehicleParams, default_vehicle_params


# 统一基础设置
DT_S = 0.1
INITIAL_SOC = 0.8
WIND_SPEED_MPS = 0.0
P_AUX_W = 500.0
RESULTS_DIR = PROJECT_ROOT / "results"

# 统一设置中文显示。若本机没有这些字体，matplotlib 会自动回退。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class PhaseConfig:
    """单个场景中的一个阶段配置。"""

    phase_name: str
    duration_s: float
    t_cmd_nm: float
    grade: float


@dataclass
class ScenarioConfig:
    """验证场景配置。"""

    scenario_id: str
    display_name: str
    description: str
    initial_speed_mps: float
    phases: list[PhaseConfig]


def mean(values: list[float]) -> float:
    """计算平均值，避免空列表时报错。"""
    if not values:
        return 0.0
    return statistics.fmean(values)


def compute_cruise_torque_nm(params: VehicleParams, speed_mps: float, grade: float) -> float:
    """
    根据当前模型中的阻力公式，估算匀速巡航所需的电机扭矩。

    说明：
    1. 当前车辆模型接口没有单独的风速输入
    2. 本次验证要求风速 = 0，因此直接按静风条件计算空气阻力即可
    """
    theta_rad = math.atan(grade)
    relative_air_speed_mps = speed_mps - WIND_SPEED_MPS

    f_roll_n = (
        params.rolling_resistance_coeff
        * params.mass_kg
        * params.gravity_mps2
        * math.cos(theta_rad)
    )
    f_aero_n = (
        0.5
        * params.air_density_kgpm3
        * params.drag_coefficient
        * params.frontal_area_m2
        * relative_air_speed_mps**2
    )
    f_grade_n = params.mass_kg * params.gravity_mps2 * math.sin(theta_rad)

    total_resistance_n = f_roll_n + f_aero_n + f_grade_n
    cruise_torque_nm = (
        total_resistance_n
        * params.wheel_radius_m
        / (params.gear_ratio * params.driveline_efficiency)
    )
    return cruise_torque_nm


def build_scenarios(params: VehicleParams) -> list[ScenarioConfig]:
    """构造 4 个验证场景。"""
    flat_cruise_speed_mps = 15.0
    uphill_cruise_speed_mps = 15.0
    regen_initial_speed_mps = 18.0

    flat_cruise_torque_nm = compute_cruise_torque_nm(params, flat_cruise_speed_mps, grade=0.0)
    uphill_cruise_torque_nm = compute_cruise_torque_nm(params, uphill_cruise_speed_mps, grade=0.05)
    regen_drive_torque_nm = compute_cruise_torque_nm(params, regen_initial_speed_mps, grade=0.0)

    return [
        ScenarioConfig(
            scenario_id="scenario_1_flat_accel",
            display_name="场景1：平路恒定正扭矩加速",
            description="从静止起步，在平路上施加恒定正扭矩，观察车速与 SOC 变化。",
            initial_speed_mps=0.0,
            phases=[
                PhaseConfig(
                    phase_name="正扭矩加速",
                    duration_s=12.0,
                    t_cmd_nm=120.0,
                    grade=0.0,
                )
            ],
        ),
        ScenarioConfig(
            scenario_id="scenario_2_flat_cruise",
            display_name="场景2：平路匀速巡航",
            description="在平路上以接近平衡阻力的扭矩巡航，观察车速稳定性。",
            initial_speed_mps=flat_cruise_speed_mps,
            phases=[
                PhaseConfig(
                    phase_name="平路巡航",
                    duration_s=20.0,
                    t_cmd_nm=flat_cruise_torque_nm,
                    grade=0.0,
                )
            ],
        ),
        ScenarioConfig(
            scenario_id="scenario_3_uphill_cruise",
            display_name="场景3：上坡巡航",
            description="在 5% 坡度下维持巡航，比较上坡与平路的功率和能耗差异。",
            initial_speed_mps=uphill_cruise_speed_mps,
            phases=[
                PhaseConfig(
                    phase_name="上坡巡航",
                    duration_s=20.0,
                    t_cmd_nm=uphill_cruise_torque_nm,
                    grade=0.05,
                )
            ],
        ),
        ScenarioConfig(
            scenario_id="scenario_4_regen_brake",
            display_name="场景4：减速 + 再生制动回收",
            description="先短暂驱动，再施加负扭矩进行再生制动，观察车速、SOC 和电池功率方向变化。",
            initial_speed_mps=regen_initial_speed_mps,
            phases=[
                PhaseConfig(
                    phase_name="驱动阶段",
                    duration_s=4.0,
                    t_cmd_nm=regen_drive_torque_nm,
                    grade=0.0,
                ),
                PhaseConfig(
                    phase_name="回收阶段",
                    duration_s=8.0,
                    t_cmd_nm=-60.0,
                    grade=0.0,
                ),
            ],
        ),
    ]


def create_model(initial_soc: float) -> LongitudinalVehicleModel:
    """创建一个新的模型实例，避免不同场景之间互相影响。"""
    params = default_vehicle_params()
    params.soc_init = initial_soc
    return LongitudinalVehicleModel(params)


def run_scenario(config: ScenarioConfig) -> dict:
    """运行单个场景，并返回记录与基础统计量。"""
    model = create_model(INITIAL_SOC)
    model.reset()

    # 通过直接设置状态，给场景一个明确的初始车速和初始 SOC。
    model.state.v_mps = config.initial_speed_mps
    model.state.soc = INITIAL_SOC

    records: list[dict] = []
    distance_m = 0.0
    previous_speed_mps = model.state.v_mps

    for phase in config.phases:
        step_count = int(round(phase.duration_s / DT_S))

        for _ in range(step_count):
            result = model.step(
                dt=DT_S,
                t_cmd_nm=phase.t_cmd_nm,
                grade=phase.grade,
                p_aux_w=P_AUX_W,
            )

            # 使用梯形积分近似累计行驶距离。
            distance_m += 0.5 * (previous_speed_mps + result["v_mps"]) * DT_S
            previous_speed_mps = result["v_mps"]

            records.append(
                {
                    "scenario_id": config.scenario_id,
                    "scenario_name": config.display_name,
                    "phase_name": phase.phase_name,
                    "dt_s": DT_S,
                    "t_cmd_nm": phase.t_cmd_nm,
                    "grade": phase.grade,
                    "wind_speed_mps": WIND_SPEED_MPS,
                    "p_aux_w": P_AUX_W,
                    "time_s": result["time_s"],
                    "v_mps": result["v_mps"],
                    "v_kph": result["v_mps"] * 3.6,
                    "a_mps2": result["a_mps2"],
                    "soc": result["soc"],
                    "motor_torque_actual_nm": result["motor_torque_actual_nm"],
                    "motor_speed_rpm": result["motor_speed_rpm"],
                    "battery_power_w": result["battery_power_w"],
                    "battery_power_kw": result["battery_power_w"] / 1000.0,
                    "cumulative_energy_kwh": result["cumulative_energy_kwh"],
                    "distance_m": distance_m,
                    "distance_km": distance_m / 1000.0,
                }
            )

    distance_km = distance_m / 1000.0
    energy_used_kwh = records[-1]["cumulative_energy_kwh"] if records else 0.0
    unit_energy_kwh_per_km = energy_used_kwh / distance_km if distance_km > 1e-9 else 0.0

    summary = {
        "initial_speed_kph": config.initial_speed_mps * 3.6,
        "final_speed_kph": records[-1]["v_kph"] if records else 0.0,
        "final_soc": records[-1]["soc"] if records else INITIAL_SOC,
        "avg_speed_kph": mean([record["v_kph"] for record in records]),
        "avg_abs_accel_mps2": mean([abs(record["a_mps2"]) for record in records]),
        "avg_battery_power_kw": mean([record["battery_power_kw"] for record in records]),
        "distance_km": distance_km,
        "energy_used_kwh": energy_used_kwh,
        "unit_energy_kwh_per_km": unit_energy_kwh_per_km,
        "speed_range_kph": (
            max(record["v_kph"] for record in records) - min(record["v_kph"] for record in records)
            if records
            else 0.0
        ),
    }

    return {"config": config, "records": records, "summary": summary}


def save_records_to_csv(records: list[dict], csv_path: Path) -> None:
    """保存单个场景的时序结果。"""
    if not records:
        return

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def plot_scenario(config: ScenarioConfig, records: list[dict], png_path: Path) -> None:
    """绘制并保存场景曲线图。"""
    time_s = [record["time_s"] for record in records]
    speed_kph = [record["v_kph"] for record in records]
    soc = [record["soc"] for record in records]
    battery_power_kw = [record["battery_power_kw"] for record in records]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(config.display_name, fontsize=14)

    axes[0].plot(time_s, speed_kph, color="#1f77b4", linewidth=2)
    axes[0].set_ylabel("车速 (km/h)")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(time_s, soc, color="#2ca02c", linewidth=2)
    axes[1].set_ylabel("SOC (-)")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    axes[2].plot(time_s, battery_power_kw, color="#d62728", linewidth=2)
    axes[2].set_ylabel("电池功率 (kW)")
    axes[2].set_xlabel("时间 (s)")
    axes[2].grid(True, linestyle="--", alpha=0.4)

    # 若一个场景含多个阶段，则用竖线标出阶段分界，便于阅读。
    boundary_time_s = 0.0
    for phase in config.phases[:-1]:
        boundary_time_s += phase.duration_s
        for axis in axes:
            axis.axvline(boundary_time_s, color="gray", linestyle=":", alpha=0.7)

    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def evaluate_scenario_1(result: dict) -> dict:
    """检查场景 1 是否满足预期。"""
    summary = result["summary"]
    speed_increase_kph = summary["final_speed_kph"] - summary["initial_speed_kph"]
    soc_drop = INITIAL_SOC - summary["final_soc"]

    passed = speed_increase_kph > 5.0 and 0.0 < soc_drop < 0.05
    reason = (
        f"车速增加 {speed_increase_kph:.2f} km/h，SOC 下降 {soc_drop:.4f}。"
        if passed
        else f"车速增加 {speed_increase_kph:.2f} km/h，SOC 下降 {soc_drop:.4f}，未同时满足上升与缓慢下降要求。"
    )

    result_lines = [
        f"初始车速：{summary['initial_speed_kph']:.2f} km/h",
        f"末端车速：{summary['final_speed_kph']:.2f} km/h",
        f"车速增量：{speed_increase_kph:.2f} km/h",
        f"末端 SOC：{summary['final_soc']:.4f}",
        f"SOC 下降量：{soc_drop:.4f}",
        f"平均电池功率：{summary['avg_battery_power_kw']:.2f} kW",
    ]

    return {"passed": passed, "reason": reason, "result_lines": result_lines}


def evaluate_scenario_2(result: dict) -> dict:
    """检查场景 2 是否满足预期。"""
    summary = result["summary"]
    passed = summary["speed_range_kph"] <= 2.0 and summary["avg_abs_accel_mps2"] <= 0.05
    reason = (
        f"车速波动范围 {summary['speed_range_kph']:.2f} km/h，平均绝对加速度 {summary['avg_abs_accel_mps2']:.4f} m/s^2。"
        if passed
        else f"车速波动范围 {summary['speed_range_kph']:.2f} km/h，平均绝对加速度 {summary['avg_abs_accel_mps2']:.4f} m/s^2，未达到基本稳定巡航要求。"
    )

    result_lines = [
        f"初始车速：{summary['initial_speed_kph']:.2f} km/h",
        f"平均车速：{summary['avg_speed_kph']:.2f} km/h",
        f"末端车速：{summary['final_speed_kph']:.2f} km/h",
        f"车速波动范围：{summary['speed_range_kph']:.2f} km/h",
        f"平均绝对加速度：{summary['avg_abs_accel_mps2']:.4f} m/s^2",
        f"平均电池功率：{summary['avg_battery_power_kw']:.2f} kW",
    ]

    return {"passed": passed, "reason": reason, "result_lines": result_lines}


def evaluate_scenario_3(result: dict, flat_cruise_result: dict) -> dict:
    """检查场景 3 是否满足预期。"""
    summary = result["summary"]
    flat_summary = flat_cruise_result["summary"]

    battery_power_higher = summary["avg_battery_power_kw"] > flat_summary["avg_battery_power_kw"]
    unit_energy_higher = summary["unit_energy_kwh_per_km"] > flat_summary["unit_energy_kwh_per_km"]
    passed = battery_power_higher and unit_energy_higher

    reason = (
        f"上坡平均电池功率 {summary['avg_battery_power_kw']:.2f} kW，高于平路巡航的 {flat_summary['avg_battery_power_kw']:.2f} kW；"
        f"上坡单位里程能耗 {summary['unit_energy_kwh_per_km']:.4f} kWh/km，高于平路巡航的 {flat_summary['unit_energy_kwh_per_km']:.4f} kWh/km。"
        if passed
        else f"上坡平均电池功率或单位里程能耗未同时高于平路巡航。"
    )

    result_lines = [
        f"平均车速：{summary['avg_speed_kph']:.2f} km/h",
        f"平均电池功率：{summary['avg_battery_power_kw']:.2f} kW",
        f"单位里程能耗：{summary['unit_energy_kwh_per_km']:.4f} kWh/km",
        f"平路巡航平均电池功率：{flat_summary['avg_battery_power_kw']:.2f} kW",
        f"平路巡航单位里程能耗：{flat_summary['unit_energy_kwh_per_km']:.4f} kWh/km",
        f"末端 SOC：{summary['final_soc']:.4f}",
    ]

    return {"passed": passed, "reason": reason, "result_lines": result_lines}


def evaluate_scenario_4(result: dict) -> dict:
    """检查场景 4 是否满足预期。"""
    records = result["records"]
    drive_records = [record for record in records if record["phase_name"] == "驱动阶段"]
    regen_records = [record for record in records if record["phase_name"] == "回收阶段"]

    drive_avg_battery_power_kw = mean([record["battery_power_kw"] for record in drive_records])
    regen_avg_battery_power_kw = mean([record["battery_power_kw"] for record in regen_records])

    drive_soc_rate = (drive_records[-1]["soc"] - INITIAL_SOC) / (len(drive_records) * DT_S)
    regen_soc_rate = (
        regen_records[-1]["soc"] - drive_records[-1]["soc"]
    ) / (len(regen_records) * DT_S)
    regen_speed_drop_kph = regen_records[0]["v_kph"] - regen_records[-1]["v_kph"]

    passed = (
        regen_speed_drop_kph > 5.0
        and drive_avg_battery_power_kw > 0.0
        and regen_avg_battery_power_kw < 0.0
        and regen_soc_rate > drive_soc_rate
    )

    reason = (
        f"回收阶段车速下降 {regen_speed_drop_kph:.2f} km/h，驱动阶段平均电池功率 {drive_avg_battery_power_kw:.2f} kW，"
        f"回收阶段平均电池功率 {regen_avg_battery_power_kw:.2f} kW，SOC 下降速度由 {drive_soc_rate:.6f}/s 变为 {regen_soc_rate:.6f}/s。"
        if passed
        else f"回收阶段的减速、电池功率方向或 SOC 变化速度未完全满足预期。"
    )

    result_lines = [
        f"驱动阶段平均电池功率：{drive_avg_battery_power_kw:.2f} kW",
        f"回收阶段平均电池功率：{regen_avg_battery_power_kw:.2f} kW",
        f"回收阶段车速降幅：{regen_speed_drop_kph:.2f} km/h",
        f"驱动阶段 SOC 变化率：{drive_soc_rate:.6f} /s",
        f"回收阶段 SOC 变化率：{regen_soc_rate:.6f} /s",
        f"末端 SOC：{result['summary']['final_soc']:.4f}",
    ]

    return {"passed": passed, "reason": reason, "result_lines": result_lines}


def build_input_lines(config: ScenarioConfig) -> list[str]:
    """整理单个场景的输入设定，便于写入 Markdown。"""
    lines = [
        f"初始车速：{config.initial_speed_mps * 3.6:.2f} km/h",
        f"时间步长 dt：{DT_S:.1f} s",
        f"初始 SOC：{INITIAL_SOC:.2f}",
        f"风速：{WIND_SPEED_MPS:.1f} m/s（当前模型无独立风速输入，本次按静风处理）",
        f"附件功率 P_aux：{P_AUX_W:.0f} W",
    ]

    for index, phase in enumerate(config.phases, start=1):
        lines.append(
            f"阶段 {index}：{phase.phase_name}，持续 {phase.duration_s:.1f} s，扭矩命令 {phase.t_cmd_nm:.2f} Nm，坡度 {phase.grade:.3f}"
        )

    return lines


def write_summary(validation_results: list[dict], summary_path: Path) -> None:
    """将验证结论写入 Markdown 文件。"""
    lines: list[str] = [
        "# 车辆动力学模型验证总结",
        "",
        "## 统一基础设置",
        f"- 时间步长 dt：{DT_S:.1f} s",
        f"- 初始 SOC：{INITIAL_SOC:.2f}",
        f"- 风速：{WIND_SPEED_MPS:.1f} m/s（当前模型接口无独立风速输入，本次按静风处理）",
        f"- 附件功率 P_aux：{P_AUX_W:.0f} W",
        "",
    ]

    for item in validation_results:
        config = item["config"]
        evaluation = item["evaluation"]
        csv_path = item["csv_path"]
        png_path = item["png_path"]

        lines.extend(
            [
                f"## {config.display_name}",
                "",
                "### 输入设定",
            ]
        )
        for input_line in item["input_lines"]:
            lines.append(f"- {input_line}")

        lines.extend(
            [
                f"- 结果 CSV：{csv_path.name}",
                f"- 结果 PNG：{png_path.name}",
                "",
                "### 关键结果",
            ]
        )
        for result_line in evaluation["result_lines"]:
            lines.append(f"- {result_line}")

        lines.extend(
            [
                "",
                "### 是否通过",
                f"- {'通过' if evaluation['passed'] else '未通过'}",
                f"- 判定说明：{evaluation['reason']}",
                "",
            ]
        )

    summary_path.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> None:
    """运行全部验证场景。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    base_params = default_vehicle_params()
    scenarios = build_scenarios(base_params)

    raw_results: dict[str, dict] = {}
    for scenario in scenarios:
        raw_results[scenario.scenario_id] = run_scenario(scenario)

    evaluated_results: list[dict] = []
    for scenario in scenarios:
        result = raw_results[scenario.scenario_id]
        records = result["records"]

        csv_path = RESULTS_DIR / f"{scenario.scenario_id}.csv"
        png_path = RESULTS_DIR / f"{scenario.scenario_id}.png"
        save_records_to_csv(records, csv_path)
        plot_scenario(scenario, records, png_path)

        if scenario.scenario_id == "scenario_1_flat_accel":
            evaluation = evaluate_scenario_1(result)
        elif scenario.scenario_id == "scenario_2_flat_cruise":
            evaluation = evaluate_scenario_2(result)
        elif scenario.scenario_id == "scenario_3_uphill_cruise":
            evaluation = evaluate_scenario_3(
                result,
                raw_results["scenario_2_flat_cruise"],
            )
        else:
            evaluation = evaluate_scenario_4(result)

        evaluated_results.append(
            {
                "config": scenario,
                "records": records,
                "summary": result["summary"],
                "input_lines": build_input_lines(scenario),
                "evaluation": evaluation,
                "csv_path": csv_path,
                "png_path": png_path,
            }
        )

    summary_path = RESULTS_DIR / "validation_summary.md"
    write_summary(evaluated_results, summary_path)

    print("验证完成。")
    for item in evaluated_results:
        status_text = "通过" if item["evaluation"]["passed"] else "未通过"
        print(f"- {item['config'].display_name}: {status_text}")
    print(f"汇总文件已保存到: {summary_path}")


if __name__ == "__main__":
    main()

