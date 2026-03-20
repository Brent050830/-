from pathlib import Path
import joblib
import pandas as pd


# 定义模型预测时必须使用的特征列
FEATURE_COLUMNS = [
    "speed_kmph",
    "accel_x",
    "accel_y",
    "brake_pressure",
    "steering_angle",
    "throttle",
    "lane_deviation",
    "phone_usage",
    "headway_distance",
    "reaction_time",
]


def get_project_root():
    # 以当前脚本所在目录为基准，向上定位到项目根目录
    return Path(__file__).resolve().parent.parent


def choose_data_path(project_root):
    # 优先读取老师演示用的小样本文件
    preferred_path = project_root / "data" / "raw" / "test_sample.csv"

    # 如果小样本文件不存在，则自动回退到完整数据集
    fallback_path = project_root / "data" / "raw" / "Driver_Behavior.csv"

    if preferred_path.exists():
        return preferred_path

    if fallback_path.exists():
        return fallback_path

    raise FileNotFoundError(
        "未找到可用于预测的数据文件，请检查以下路径是否存在：\n"
        f"1. {preferred_path}\n"
        f"2. {fallback_path}"
    )


def validate_columns(df, feature_columns):
    # 检查数据中是否包含模型预测所需的全部特征列
    missing_columns = [column for column in feature_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"数据缺少以下特征列，无法完成预测：{missing_columns}")


def main():
    # 调整 pandas 的显示效果，方便老师演示时在终端中完整查看结果
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1200)

    # 定位项目根目录、模型文件和结果文件路径
    project_root = get_project_root()
    model_path = project_root / "models" / "rf_model.joblib"
    output_path = project_root / "results" / "predictions_demo.csv"

    # 自动选择数据文件：优先 test_sample.csv，不存在则回退到 Driver_Behavior.csv
    data_path = choose_data_path(project_root)

    # 检查模型文件是否存在，避免运行时出现不明确的报错
    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型文件：{model_path}")

    # 读取原始数据
    df = pd.read_csv(data_path)

    # 检查特征列是否齐全
    validate_columns(df, FEATURE_COLUMNS)

    # 加载已经训练好的随机森林模型
    model = joblib.load(model_path)

    # 提取模型需要的输入特征，并执行预测
    feature_df = df[FEATURE_COLUMNS]
    predicted_labels = model.predict(feature_df)

    # 复制一份原始数据并追加预测结果，便于保存和展示
    result_df = df.copy()
    result_df["predicted_label"] = predicted_labels

    # 创建结果目录并保存预测结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # 打印本次实际使用的数据文件，老师演示时更直观
    print(f"本次使用的数据文件：{data_path}")
    print(f"本次加载的模型文件：{model_path}")

    # 输出前 10 行预测结果
    print("\n===== 前10行预测结果 =====")
    print(result_df.head(10))

    # 输出 predicted_label 的类别统计
    print("\n===== predicted_label 类别统计 =====")
    print(result_df["predicted_label"].value_counts(dropna=False))

    # 如果原始数据中存在真实标签，则同时打印真实标签和预测标签，并计算一致率
    if "behavior_label" in result_df.columns:
        print("\n===== 前10行真实标签与预测标签对比 =====")
        print(result_df[["behavior_label", "predicted_label"]].head(10))

        consistency_rate = (
            result_df["behavior_label"].astype(str)
            == result_df["predicted_label"].astype(str)
        ).mean()
        print(f"\n真实标签与预测标签一致率：{consistency_rate:.2%}")
    else:
        print("\n原始数据中不包含 behavior_label，跳过一致率计算。")

    print(f"\n预测结果已保存到：{output_path}")


if __name__ == "__main__":
    main()
