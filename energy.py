from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def run_energy_experiment(df, feature_columns, target_column, experiment_name):

    # 提取特征和标签
    X = df[feature_columns]
    y = df[target_column]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # 创建随机森林回归模型
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )

    # 训练模型
    model.fit(X_train, y_train)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    # 预测
    y_pred = model.predict(X_test)

    # 计算评价指标
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    # 特征重要性
    feature_importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)

    # 打印结果
    print(f"\n===== {experiment_name} =====")

    print("使用特征：")
    for feature in feature_columns:
        print(f"- {feature}")

    print(f"\n训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")

    print(f"\nMAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 score: {r2:.4f}")

    return model, feature_importance_df


def main():

    # 数据路径
    data_path = Path("data/raw/Vehicle_Energy.csv")

    # 全特征
    all_feature_columns = [
        "speed_kmph",
        "accel_x",
        "accel_y",
        "throttle",
        "brake_pressure",
        "steering_angle",
        "road_slope",
    ]

    # 简化特征（用于对比实验）
    reduced_feature_columns = [
        "speed_kmph",
        "accel_x",
        "throttle",
        "brake_pressure",
    ]

    # 目标变量（车辆能耗）
    target_column = "energy_consumption"

    # 读取数据
    df = pd.read_csv(data_path)

    # ===============================
    # 实验1：全特征能耗模型
    # ===============================
    full_model, feature_importance_df = run_energy_experiment(
        df=df,
        feature_columns=all_feature_columns,
        target_column=target_column,
        experiment_name="实验1：全特征能耗模型",
    )

    # 打印特征重要性
    print("\n===== 特征重要性（从高到低） =====")
    for _, row in feature_importance_df.iterrows():
        print(f"{row['feature']}: {row['importance']:.6f}")

    # 创建结果目录
    result_dir = Path("results")
    result_dir.mkdir(exist_ok=True)

    # 保存特征重要性
    feature_importance_path = result_dir / "energy_feature_importance.csv"

    feature_importance_df.to_csv(
        feature_importance_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\n特征重要性已保存到: {feature_importance_path}")

    # 创建模型目录
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    # 保存模型
    model_path = model_dir / "vehicle_energy_model.joblib"

    joblib.dump(full_model, model_path)

    print(f"模型已保存到: {model_path}")

    # ===============================
    # 实验2：简化特征能耗模型
    # ===============================
    run_energy_experiment(
        df=df,
        feature_columns=reduced_feature_columns,
        target_column=target_column,
        experiment_name="实验2：简化特征能耗模型",
    )


if __name__ == "__main__":
    main()
