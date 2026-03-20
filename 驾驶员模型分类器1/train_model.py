from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def run_experiment(df, feature_columns, target_column, experiment_name):
    # 从原始数据中取出当前实验使用的特征和标签
    X = df[feature_columns]
    y = df[target_column]

    # 将数据划分为训练集和测试集
    # test_size=0.2 表示 20% 数据作为测试集
    # stratify=y 表示按标签分层抽样，尽量保持类别比例一致
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 创建随机森林分类器
    # 随机森林可以直接处理多分类任务，这里用于三分类
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    # 使用训练集训练模型
    model.fit(X_train, y_train)

    # 使用测试集进行预测
    y_pred = model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)

    # 提取当前实验的特征重要性，并按从高到低排序
    feature_importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)

    # 清晰打印当前实验结果
    print(f"\n===== {experiment_name} =====")
    print("使用的特征列表:")
    for feature in feature_columns:
        print(f"- {feature}")

    print(f"\n训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    print(f"accuracy: {accuracy:.4f}")

    print("\nclassification_report:")
    print(classification_report(y_test, y_pred))

    print("confusion_matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, feature_importance_df


def main():
    # 定义数据文件路径
    data_path = Path("Driver_Behavior.csv")

    # 定义全特征列表
    all_feature_columns = [
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

    # 定义第二组实验使用的特征列表
    # 在全特征基础上，去掉 reaction_time 和 phone_usage 两列
    reduced_feature_columns = [
        "speed_kmph",
        "accel_x",
        "accel_y",
        "brake_pressure",
        "steering_angle",
        "throttle",
        "lane_deviation",
        "headway_distance",
    ]

    # 定义标签列
    target_column = "behavior_label"

    # 读取 CSV 数据
    df = pd.read_csv(data_path)

    # 实验1：使用全部特征训练随机森林模型
    full_model, full_feature_importance_df = run_experiment(
        df=df,
        feature_columns=all_feature_columns,
        target_column=target_column,
        experiment_name="实验1：全特征",
    )

    # 输出实验1的特征重要性，帮助了解哪些特征对模型影响更大
    print("\n===== 实验1特征重要性（从高到低） =====")
    for _, row in full_feature_importance_df.iterrows():
        print(f"{row['feature']}: {row['importance']:.6f}")

    # 创建结果保存目录
    result_dir = Path("results")
    result_dir.mkdir(exist_ok=True)

    # 将实验1的特征重要性保存为 CSV 文件
    feature_importance_path = result_dir / "feature_importance.csv"
    full_feature_importance_df.to_csv(
        feature_importance_path,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"\n特征重要性已保存到: {feature_importance_path}")

    # 创建模型保存目录
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    # 保存实验1训练好的模型，方便后续直接加载使用
    model_path = model_dir / "rf_model.joblib"
    joblib.dump(full_model, model_path)
    print(f"模型已保存到: {model_path}")

    # 实验2：去掉 reaction_time 和 phone_usage 后重新训练模型
    # 这样可以对比这两个特征是否明显影响模型效果
    run_experiment(
        df=df,
        feature_columns=reduced_feature_columns,
        target_column=target_column,
        experiment_name="实验2：去掉 reaction_time 和 phone_usage",
    )


if __name__ == "__main__":
    main()
