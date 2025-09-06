import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json,datetime

import matplotlib.pyplot as plt
import numpy as np


def plot_tofd_histogram(data=None, result_dir='results_tradition'):
    """
    Plot and save histogram of TOFD data.

    Args:
    data: numpy array or list with TOFD data. If None, synthetic data is used.
    output_path: path to save image
    """
    # If not provided, generate synthetic data
    if data is None:
        np.random.seed(42)  # 确保可重复性
        # synthetic data: mostly near -0.02, some elsewhere
        main_data = np.random.normal(-0.02, 0.05, 200)  # 主峰区域
        other_data = np.concatenate([
            np.random.uniform(-0.61, -0.1, 10),
            np.random.uniform(0.3, 2.07, 40)
        ])
        data = np.concatenate([main_data, other_data])

    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(6, 6))

    # 绘制直方图
    counts, bins, patches = ax.hist(data, bins=50, color='blue', alpha=0.7)

    # labels
    # ax.set_title('Sample Count', fontsize=14)
    ax.set_xlabel('TOFD(μs)', fontsize=20)
    ax.set_ylabel('Sample Count', fontsize=20)

    # x ticks
    x_ticks = [-0.61, -0.31, -0.02, 0.28, 0.58, 0.88, 1.17, 1.47, 1.77, 2.07]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{x:.2f}' for x in x_ticks])

    # y ticks
    y_ticks = [-200, -100, -50, 0, 50, 100, 150, 200]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(y) for y in y_ticks])

    # axis ranges
    ax.set_xlim(-0.61, 2.07)
    ax.set_ylim(0, 250)  # 略高于200以显示完整柱子

    # grid
    ax.grid(True, alpha=0.3)

    # 保存图像
    histogram_path = os.path.join(result_dir, "histogram.jpg")
    plt.savefig(histogram_path, dpi=600, bbox_inches='tight')

    print(f"Histogram saved to: {histogram_path}")
    return counts, bins

def convert_and_analyze_csv(input_csv_file, output_csv_file, result_dir="./results",
                            bin_width=0.03, std_threshold=3.0):
    """
    Convert various CSV formats to unified schema and analyze errors.

    Args:
    input_csv_file: input CSV path
    output_csv_file: output CSV path (unified)
    result_dir: directory to save results
    bin_width: histogram bin width (default 0.03)
    std_threshold: z-score threshold for outliers (default 3.0)
    """

    # 确保结果目录存在
    os.makedirs(result_dir, exist_ok=True)

    # 读取输入CSV文件
    try:
        df = pd.read_csv(input_csv_file)

        # map columns to unified names
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'true_v' in col or 'true' in col_lower:
                column_mapping[col] = 'true_value'
            elif 'prediction_v' in col or 'predicted' in col_lower:
                column_mapping[col] = 'predicted_value'
            elif 'ID' in col or 'id' in col_lower or 'index' in col_lower:
                # keep ID if exists
                pass

        # 重命名列
        df_renamed = df.rename(columns=column_mapping)

        # ensure required columns
        required_columns = ['true_value', 'predicted_value']
        if not all(col in df_renamed.columns for col in required_columns):
            print("Error: CSV must contain true and predicted columns")
            return None

        # compute error (pred - true)
        df_renamed['error'] = df_renamed['predicted_value'] - df_renamed['true_value']

        # select columns
        keep_columns = ['true_value', 'predicted_value', 'error']
        # keep original ID column if any
        for col in df.columns:
            if 'ID' in col or 'id' in col.lower() or 'index' in col.lower():
                keep_columns.insert(0, col)
                break

        # create unified DataFrame
        df_unified = df_renamed[keep_columns]

        # save CSV
        df_unified.to_csv(output_csv_file, index=False)
        print(f"Unified CSV saved: {output_csv_file}")

        # analyze errors
        stats = plot_error_analysis_from_csv(
            csv_file=output_csv_file,
            result_dir=result_dir,
            bin_width=bin_width,
            std_threshold=std_threshold
        )

        return stats

    except Exception as e:
        print(f"Error processing CSV: {e}")
        return None

def plot_error_analysis_from_csv(csv_file, result_dir="./results", bin_width=0.01, std_threshold=3.0):
    """
    Read CSV and analyze errors, plot histogram, compute metrics.

    Args:
    csv_file: CSV path containing true_value, predicted_value, error
    result_dir: directory to save results
    bin_width: histogram bin width (default 0.03)
    std_threshold: z-score threshold for outliers (default 3.0)
    """

    # 确保结果目录存在
    os.makedirs(result_dir, exist_ok=True)
    # 从CSV文件读取数据
    try:
        df = pd.read_csv(csv_file)

        # ensure required columns
        required_columns = ['true_value', 'predicted_value', 'error']
        if not all(col in df.columns for col in required_columns):
            print("Error: CSV must contain 'true_value', 'predicted_value', 'error'")
            return None

        # extract data
        true_values = df['true_value'].values[:1000]
        predicted_values = df['predicted_value'].values[:1000]
        errors = df['error'].values[:1000]

        # auto bin width (optional)
        # bin_width = (errors.max() - errors.min()) / min(10, len(errors))

        # remove NaNs
        valid_mask = ~np.isnan(errors) & ~np.isnan(true_values) & ~np.isnan(predicted_values)
        true_values = true_values[valid_mask]
        predicted_values = predicted_values[valid_mask]
        errors = errors[valid_mask]

        # record original row indices (1-based)
        valid_indices = np.where(valid_mask)[0] + 2  # +2是因为CSV从第2行开始（第1行是标题）

    except Exception as e:
        print(f"Error reading data: {e}")
        return None

    #画图
    # print("--------------")
    # plot_tofd_histogram(errors)
    # print("-----实例完成----")

    # statistics
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    std_error = np.std(errors)

    # extra metrics
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mare = np.mean(np.abs(errors) / (np.abs(true_values) + 1e-8))  # 防止除零

    # outliers
    z_scores = np.abs((errors - mean_error) / std_error)
    outliers_mask = z_scores > std_threshold
    outliers = errors[outliers_mask]

    # map outliers to original row indices
    outlier_indices = [valid_indices[i] for i in range(len(errors)) if outliers_mask[i]]
    outlier_z_scores = z_scores[outliers_mask]
    outlier_true_values = true_values[outliers_mask]
    outlier_predicted_values = predicted_values[outliers_mask]

    # print metrics
    print(f"\nData statistics:")
    print(f"Total: {len(errors)}")
    print(f"Outliers: {len(outliers)} ({len(outliers) / len(errors) * 100:.2f}%)")
    print(f"Mean error: {mean_error:.6f}")
    print(f"Median error: {median_error:.6f}")
    print(f"Std error: {std_error:.6f}")
    print(f"Min error: {np.min(errors):.6f}")
    print(f"Max error: {np.max(errors):.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MARE: {mare:.6f}")

    # 保存评估指标到文件
    metrics_data = {
        "total_count": len(errors),
        "outlier_count": len(outliers),
        "outlier_percentage": f"{len(outliers) / len(errors) * 100:.2f}%",
        "mean_error": float(mean_error),
        "median_error": float(median_error),
        "std_error": float(std_error),
        "min_error": float(np.min(errors)),
        "max_error": float(np.max(errors)),
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "mare": float(mare),
        "bin_width": bin_width,
        "std_threshold": std_threshold
    }

    # 保存为JSON文件
    metrics_json_path = os.path.join(result_dir, "another_cnn_evaluation_metrics.json")
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    print(f"\nMetrics saved to file:")
    print(f"  - {metrics_json_path}")

    if len(outliers) > 0:
        print("\nOutliers (sorted by deviation):")
        # 按偏离程度排序
        sorted_indices = np.argsort(np.abs(outliers - mean_error))[::-1]
        for i, idx in enumerate(sorted_indices):
            print(f"Outlier {i + 1}:")
            print(f"  Row: {outlier_indices[idx]}")
            print(f"  True: {outlier_true_values[idx]:.6f}")
            print(f"  Pred: {outlier_predicted_values[idx]:.6f}")
            print(f"  Error: {outliers[idx]:.6f}")
            print(f"  Z-score: {outlier_z_scores[idx]:.2f}")
            print()
    else:
        print("\nNo outliers detected")

    # 设置字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20

    # histogram
    min_error = min(errors)
    max_error = max(errors)

    bins = np.arange(min_error, max_error + bin_width, bin_width)
    hist, bin_edges = np.histogram(errors, bins=bins)

    plt.figure(figsize=(6, 6))
    bars = plt.bar(bin_edges[:-1], hist, width=bin_width, align='edge', edgecolor='black')
    plt.xlabel('Error(μs)',fontsize=20)
    plt.ylabel('Sample Count',fontsize=20)
    plt.tick_params(width=1)

    # # 标记异常值范围
    # lower_bound = mean_error - std_threshold * std_error
    # upper_bound = mean_error + std_threshold * std_error
    #
    # # 绘制异常值边界线
    # plt.axvline(lower_bound, color='red', linestyle='--', alpha=0.7,
    #             label=f'Lower bound ({std_threshold}σ)')
    # plt.axvline(upper_bound, color='red', linestyle='--', alpha=0.7,
    #             label=f'Upper bound ({std_threshold}σ)')
    #
    # # 标记平均值
    # plt.axvline(mean_error, color='green', linestyle='-', alpha=0.7,
    #             label=f'Mean: {mean_error:.4f}')

    # value labels
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width() / 2,
                     height,
                     f'{int(height)}',
                     ha='center',
                     va='bottom',
                     fontsize=12,
                     fontname='Times New Roman')

    # tick params
    plt.tick_params(width=1)

    # X ticks
    xtick_pos = np.linspace(min_error, max_error, 7)
    xtick_labels = [f'{x:.2f}' for x in xtick_pos]
    plt.xticks(xtick_pos, labels=xtick_labels, fontsize=16, fontname='Times New Roman')

    # Y ticks [0,50,100,150,200,250]
    # plt.xticks([-0.61,-0.31,-0.2,0,0.2,0.4, ], fontname='Times New Roman', fontsize=16)
    plt.yticks([0, 50, 100, 150, 200], fontname='Times New Roman', fontsize=16)

    # remove top/right spines (optional)
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)

    # plt.title(f'Error Distribution (Outliers: {len(outliers)}/{len(errors)})')
    # plt.legend()
    plt.tight_layout()

    # save histogram
    histogram_path = os.path.join(result_dir, "another_cnn_error_histogram.jpg")
    plt.savefig(histogram_path, dpi=600, bbox_inches='tight')
    plt.close()
    #
    # # 创建箱线图以可视化异常值
    # plt.figure(figsize=(8, 6))
    # plt.boxplot(errors)
    # plt.ylabel('Error Value')
    # plt.title('Boxplot of Error Values')
    # boxplot_path = os.path.join(result_dir, "error_boxplot.jpg")
    # plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    # plt.close()

    # # 创建预测值 vs 真实值散点图
    # plt.figure(figsize=(8, 6))
    # plt.scatter(true_values, predicted_values, alpha=0.5, label='Normal data')
    #
    # # 标记异常值
    # if len(outliers) > 0:
    #     plt.scatter(outlier_true_values, outlier_predicted_values,
    #                 alpha=0.7, color='red', marker='x', s=50, label='Outliers')
    #
    # min_val = min(np.min(true_values), np.min(predicted_values))
    # max_val = max(np.max(true_values), np.max(predicted_values))
    # plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal fit')
    # plt.xlabel('True Value')
    # plt.ylabel('Predicted Value')
    # plt.title('Predicted vs True Values')
    # plt.grid(True)
    # plt.legend()
    # scatter_path = os.path.join(result_dir, "pred_vs_true_scatter.jpg")
    # plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    # plt.close()

    return {
        'total_count': len(errors),
        'outlier_count': len(outliers),
        'mean_error': mean_error,
        'median_error': median_error,
        'std_error': std_error,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mare': mare,
        'outliers': outliers,
        'outlier_indices': outlier_indices,
        'outlier_z_scores': outlier_z_scores,
        'outlier_true_values': outlier_true_values,
        'outlier_predicted_values': outlier_predicted_values
    }

def find_data_gaps(counts, min_gap_size=3):
    """
    找出数据中的间隙（连续零值的区域）

    参数:
    counts (list): 数据计数列表
    min_gap_size (int): 最小间隙大小

    返回:
    list: 间隙的起始和结束索引列表
    """
    gaps = []
    in_gap = False
    gap_start = 0

    for i, count in enumerate(counts):
        if count == 0 and not in_gap:
            in_gap = True
            gap_start = i
        elif count > 0 and in_gap:
            in_gap = False
            if i - gap_start >= min_gap_size:
                gaps.append((gap_start, i - 1))

    # 处理最后一个元素是零的情况
    if in_gap and len(counts) - gap_start >= min_gap_size:
        gaps.append((gap_start, len(counts) - 1))

    # 按间隙大小排序（从大到小）
    gaps.sort(key=lambda x: x[1] - x[0], reverse=True)

    return gaps

if __name__ == "__main__":
    # plot_tofd_histogram()
    # # 调用函数进行格式转换和误差分析
    # stats = convert_and_analyze_csv(
    #     input_csv_file="test_results_CNN/cnn_15DB_20250829_102116/predictions.csv",  # 输入CSV文件路径
    #     output_csv_file="test_results_CNN/cnn_15DB_20250829_102116/predictions_new.csv",  # 输出CSV文件路径
    #     # input_csv_file="test_results_another_CNN/another_cnn_15DB_20250829_155714/predictions.csv",  # 输入CSV文件路径
    #     # output_csv_file="test_results_another_CNN/another_cnn_15DB_20250829_155714/predictions_new.csv",  # 输出CSV文件路径
    #     result_dir="./plt_sw_results",  # 结果保存目录
    #     bin_width=0.03,  # 直方图bin宽度
    #     std_threshold=3.0  # 异常值判断的标准差阈值
    # )
    #
    # if stats:
    #     # 打印统计信息
    #     print(f"\n分析完成!")
    #     print(f"统一格式的CSV文件已保存")

    stats = plot_error_analysis_from_csv(
        # csv_file="results_tradition/ADE_LM_results.csv",  # 替换为你的CSV文件路径
        # csv_file="test_results_CNN/cnn_15DB_20250829_102116/predictions_new.csv",  # 替换为你的CSV文件路径
        csv_file="test_results_another_CNN/another_cnn_15DB_20250829_155714/predictions_new.csv",  # 替换为你的CSV文件路径
        # csv_file="test_results_sw/15DB_results_20250827_231547/predictions.csv",  # 替换为你的CSV文件路径
        result_dir="./plt_15DB_results",  # 结果保存目录
        bin_width=0.028,  # 直方图bin宽度
        std_threshold=3.0  # 异常值判断的标准差阈值
    )

    if stats:
        print(f"\nAnalysis done!")