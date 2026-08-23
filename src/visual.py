# -- coding: utf-8 --
import logging

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def visualize_results(results_dict):
    plt.figure(figsize=(12, 6))
    plt.plot(results_dict['train_loss'], label='训练损失')
    plt.title(" 训练过程损失曲线")
    plt.xlabel(" 训练轮次")
    plt.ylabel(" 综合损失值")
    plt.legend()
    plt.savefig("results/training_loss.png", dpi=300, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(14, 7))
    plt.plot(results_dict['test_results']['true_values'], label='实际值', marker='o', linestyle='--', alpha=0.7)
    plt.plot(results_dict['test_results']['predictions'], label='预测值', marker='s', alpha=0.7)
    plt.fill_between(range(len(results_dict['test_results']['predictions'])), results_dict['test_results']['predictions'] - 200, results_dict['test_results']['predictions'] + 200, alpha=0.2, color='orange')
    plt.title(" 预测效果对比")
    plt.xlabel(" 时间周期")
    plt.ylabel(" 作业量需求")
    plt.grid(True)
    plt.legend()
    plt.savefig("results/prediction_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    errors = np.array(results_dict['test_results']['predictions']) - np.array(results_dict['test_results']['true_values'])
    plt.figure(figsize=(12, 6))
    plt.hist(errors, bins=20, alpha=0.7)
    plt.title("预测误差分布")
    plt.xlabel("误差")
    plt.ylabel("频次")
    plt.savefig("results/prediction_error_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(12, 6))
    plt.plot(results_dict['test_results']['cost_gap_p'], label='成本差距')
    plt.title("成本差距随时间的变化")
    plt.xlabel("时间周期")
    plt.ylabel("成本差距")
    plt.legend()
    plt.savefig("results/cost_gap.png", dpi=300, bbox_inches='tight')
    plt.close()
def visualize_results(results, label=None):
    ...
    # 保存图像时加入 label
    if label:
        plt.savefig(f"results/train_loss_{label}.png")
    else:
        plt.savefig("results/train_loss.png")

def visualize_multi_period_solution(result, filename="multi_period_solution.png"):
    if result["status"] != "optimal":
        logging.error(f"无法可视化非最优解，状态: {result['status']}")
        return
    n_terminals = result["renovation_status"].shape[0]
    planning_horizon = result["renovation_status"].shape[1]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    ax1.plot(range(1, planning_horizon + 1), result["demand_values"], 'k-', linewidth=2, label="需求预测")
    ax1.bar(range(1, planning_horizon + 1), result["delay_amounts"], color='r', alpha=0.5, label="延迟量")
    ax1.set_ylabel("作业量/延迟量")
    ax1.set_title("作业量需求与延迟")
    ax1.legend()
    ax1.grid(True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    hatches = ['/', '\\', 'x', '.', '*']
    for b in range(n_terminals):
        for t in range(planning_horizon):
            if result["renovation_status"][b, t] > 0.5:
                ax2.add_patch(plt.Rectangle((t + 0.5, b + 0.5), 1, 1, fill=True, color='gray', alpha=0.7, hatch='///'))
        for t in range(planning_horizon):
            if result["renovation_decisions"][b, t] > 0.5:
                ax2.plot(t + 1, b + 1, 'ro', markersize=10)
    ax2.set_yticks(range(1, n_terminals + 1))
    ax2.set_yticklabels([f"箱区{b + 1}" for b in range(n_terminals)], fontsize=8)
    ax2.set_ylabel("箱区")
    ax2.set_title("改造状态 (灰色=改造中，红点=改造开始)")
    ax2.grid(True)
    bottom = np.zeros(planning_horizon)
    for b in range(n_terminals):
        ax3.bar(range(1, planning_horizon + 1), result["workload_allocation"][b, :],
                bottom=bottom, label=f"箱区{b + 1}", alpha=0.7,
                color=colors[b % len(colors)], hatch=hatches[b % len(hatches)])
        bottom += result["workload_allocation"][b, :]
    ax3.plot(range(1, planning_horizon + 1), result["demand_values"], 'k--', linewidth=2, label="需求预测")
    month_names = ["一月", "二月", "三月", "四月", "五月", "六月",
                   "七月", "八月", "九月", "十月", "十一月", "十二月"]
    ax3.set_xlabel("月份")
    ax3.set_ylabel("作业量分配")
    ax3.set_title("各箱区作业量分配")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(f"results/{filename}", dpi=300, bbox_inches='tight')
    plt.close()