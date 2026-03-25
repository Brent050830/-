from pathlib import Path

from longitudinal import LongitudinalVehicleModel
from params import default_vehicle_params
from simulator import SimulationStepInput, run_simulation, save_results_to_csv


def build_demo_inputs() -> list[SimulationStepInput]:
    """构造一个简单的演示工况。"""
    dt_s = 0.1
    step_inputs: list[SimulationStepInput] = []

    def append_phase(
        phase_name: str,
        duration_s: float,
        t_cmd_nm: float,
        grade: float = 0.0,
        p_aux_w: float = 800.0,
    ) -> None:
        step_count = int(duration_s / dt_s)
        for _ in range(step_count):
            step_inputs.append(
                SimulationStepInput(
                    phase_name=phase_name,
                    dt_s=dt_s,
                    t_cmd_nm=t_cmd_nm,
                    grade=grade,
                    p_aux_w=p_aux_w,
                )
            )

    append_phase("平路加速", duration_s=10.0, t_cmd_nm=180.0)
    append_phase("平路匀速", duration_s=8.0, t_cmd_nm=12.0)
    append_phase("减速回收", duration_s=6.0, t_cmd_nm=-60.0)

    return step_inputs


def main() -> None:
    params = default_vehicle_params()
    model = LongitudinalVehicleModel(params)
    step_inputs = build_demo_inputs()

    records = run_simulation(model, step_inputs)

    project_dir = Path(__file__).resolve().parent
    csv_output_path = project_dir / "vehicle_demo_results.csv"
    plot_output_path = project_dir / "vehicle_demo_plots.png"

    save_results_to_csv(records, csv_output_path)

    final_record = records[-1]

    print("纯电动车纵向动力学演示完成。")
    print(f"CSV 文件已保存到: {csv_output_path}")
    print(f"最终车速: {final_record['v_kph']:.2f} km/h")
    print(f"最终加速度: {final_record['a_mps2']:.2f} m/s^2")
    print(f"最终 SOC: {final_record['soc']:.4f}")
    print(f"累计能耗: {final_record['cumulative_energy_kwh']:.4f} kWh")

    # 自动读取 CSV 并绘图，便于在 VS Code 中直接查看结果。
    try:
        from plot_vehicle_results import load_results, plot_results

        plot_data = load_results(csv_output_path)
        plot_results(plot_data, plot_output_path)
        print(f"图片文件已保存到: {plot_output_path}")
    except Exception as exc:
        print(f"自动绘图未完成，但仿真和 CSV 已成功生成: {exc}")


if __name__ == "__main__":
    main()
