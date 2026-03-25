import csv
from pathlib import Path

import matplotlib.pyplot as plt


# 统一设置中文显示。若本机没有这些字体，matplotlib 会自动回退到可用字体。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def load_results(csv_path: Path) -> dict[str, list[float]]:
    """从 CSV 文件读取仿真结果。"""
    data = {
        "time_s": [],
        "v_kph": [],
        "soc": [],
        "motor_torque_actual_nm": [],
        "battery_power_kw": [],
    }

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data["time_s"].append(float(row["time_s"]))
            data["v_kph"].append(float(row["v_kph"]))
            data["soc"].append(float(row["soc"]))
            data["motor_torque_actual_nm"].append(float(row["motor_torque_actual_nm"]))
            data["battery_power_kw"].append(float(row["battery_power_w"]) / 1000.0)

    return data


def plot_results(data: dict[str, list[float]], output_png_path: Path) -> None:
    """绘制车速、SOC、扭矩和电池功率曲线。"""
    time_s = data["time_s"]

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    fig.suptitle("纯电动车纵向动力学仿真结果", fontsize=14)

    # 子图 1：车速
    axes[0].plot(time_s, data["v_kph"], color="#1f77b4", linewidth=2)
    axes[0].set_ylabel("车速 (km/h)")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # 子图 2：SOC
    axes[1].plot(time_s, data["soc"], color="#2ca02c", linewidth=2)
    axes[1].set_ylabel("SOC (-)")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    # 子图 3：电机实际输出扭矩
    axes[2].plot(time_s, data["motor_torque_actual_nm"], color="#ff7f0e", linewidth=2)
    axes[2].set_ylabel("扭矩 (Nm)")
    axes[2].grid(True, linestyle="--", alpha=0.4)

    # 子图 4：电池功率
    axes[3].plot(time_s, data["battery_power_kw"], color="#d62728", linewidth=2)
    axes[3].set_ylabel("电池功率 (kW)")
    axes[3].set_xlabel("时间 (s)")
    axes[3].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.savefig(output_png_path, dpi=150)

    # 在可交互后端下显示图窗；在 Agg 等后端下只保存图片。
    backend_name = plt.get_backend().lower()
    if "agg" not in backend_name:
        plt.show()

    plt.close(fig)


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    csv_path = project_dir / "vehicle_demo_results.csv"
    output_png_path = project_dir / "vehicle_demo_plots.png"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"未找到仿真结果文件：{csv_path}。请先运行 run_vehicle_demo.py 生成 CSV。"
        )

    data = load_results(csv_path)
    plot_results(data, output_png_path)

    print("绘图完成。")
    print(f"图片文件已保存到: {output_png_path}")


if __name__ == "__main__":
    main()
