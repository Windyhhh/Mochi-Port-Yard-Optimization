# -- coding: utf-8 --
"""
基于SPO+的码头堆场改造优化系统（NUM_ZONES 版）
整合了LSTM需求预测与运筹优化模型（修正版）

"""
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import matplotlib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mochi.model import MultiStepLSTMPredictor, MultiStepGRUPredictor

# ================= 基本设置 =================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
matplotlib.use('Agg')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
Path("results").mkdir(exist_ok=True)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ⭐ 统一控制箱区数量
NUM_ZONES = 30
DATA_FILE = "yard_12years_30zones.csv"
# ================= 数据处理 & 数据集 =================
class DataProcessor:
    def normalize(self, data):
        min_val = data.min(axis=0)
        max_val = data.max(axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8), min_val, max_val

    def denormalize(self, data, min_val, max_val):
        return data * (max_val - min_val + 1e-8) + min_val

class TerminalDataset(Dataset):
    def __init__(self, data, month_sin, month_cos, year_index, long_trend, seq_length=60, forecast_horizon=12):
        self.data = data
        self.month_sin = month_sin
        self.month_cos = month_cos
        self.year_index = year_index
        self.long_trend = long_trend
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.num_zones = min(data.shape[1], NUM_ZONES)

    def __len__(self):
        return len(self.data) - self.seq_length - self.forecast_horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length][:, :self.num_zones]
        m_sin = self.month_sin[idx:idx + self.seq_length].reshape(-1, 1)
        m_cos = self.month_cos[idx:idx + self.seq_length].reshape(-1, 1)
        y_idx = self.year_index[idx:idx + self.seq_length].reshape(-1, 1)
        trend = self.long_trend[idx:idx + self.seq_length].reshape(-1, 1)
        x_all = np.hstack([x, m_sin, m_cos, y_idx, trend])
        y = self.data[idx + self.seq_length:idx + self.seq_length + self.forecast_horizon][:, :self.num_zones]
        return torch.tensor(x_all, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ================= 区域类型 & 优化器 =================
class TerminalTypeManager:
    def __init__(self):
        self.zone_types = {}

    def set_terminal_types_by_ranges(self, nearshore_range=[], midshore_range=[], offshore_range=[]):
        for idx in nearshore_range:
            self.zone_types[idx] = 1  # 靠泊侧
        for idx in midshore_range:
            self.zone_types[idx] = 2  # 缓冲区
        for idx in offshore_range:
            self.zone_types[idx] = 3  # 陆侧

    def get_zone_types(self):
        return self.zone_types

class TerminalOptimizer:
    def __init__(self, w_origin, w_enhanced, n_terminals=NUM_ZONES, planning_horizon=12, balance_gamma=500000, c_ren=3000000,                 # ⭐ 新增：单位改造成本
                 min_total_renovations=0,       # ⭐ 新增：最少改造数（敏感性时设0，让模型自己选 H*）
                 exact_total_renovations=None   # ⭐ 新增：恰好改造数（不需要固定H时保持None）
                 ):
        self.n_terminals = n_terminals
        self.planning_horizon = planning_horizon
        self.workload_original = w_origin[:n_terminals]
        self.workload_enhanced = w_enhanced[:n_terminals]


        # 成本与参数
        self.c_ren = 3000000
        self.c_delay = 150
        self.c_base = 100
        self.c_tran = 10
        self.eta = 0.3
        self.renovation_period = 2
        self.max_simultaneous_renovations = 5
        self.balance_gamma = balance_gamma
        self.min_total_renovations = min_total_renovations
        self.exact_total_renovations = exact_total_renovations

        self.multi_period_model = None
        self.demand_vars = None
        self.x_vars = None
        self.s_vars = None
        self.q_vars = None
        self.z_vars = None
        self.is_renovated_vars = None
        self.r_vars = None
        self._build_multi_period_model()

    def _build_multi_period_model(self):
        try:
            model = gp.Model("MultiPeriodTerminalOptimization")
            T = range(1, self.planning_horizon + 1)
            B = range(1, self.n_terminals + 1)

            x = model.addVars([(b, t) for b in B for t in T], vtype=GRB.BINARY, name="x")
            s = model.addVars([(b, t) for b in B for t in T], vtype=GRB.BINARY, name="s")
            q = model.addVars([(b, t) for b in B for t in T], vtype=GRB.CONTINUOUS, lb=0, name="q")
            z = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="z")
            demand = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="demand")
            is_renovated = model.addVars([(b, t) for b in B for t in T], vtype=GRB.BINARY, name="is_renovated")
            r = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0.0, name="starts_per_period")
            for t in T:
                model.addConstr(r[t] == gp.quicksum(x[b, t] for b in B), name=f"starts_count_{t}")

            # 每个箱区最多改造一次
            for b in B:
                model.addConstr(gp.quicksum(x[b, t] for t in T) <= 1, name=f"unique_renovation_{b}")

            # 12个月内至少改造 5 个箱区（按需调整或删除）
            min_renovations = 5
            model.addConstr(gp.quicksum(x[b, t] for b in B for t in T) >= min_renovations, name="min_total_renovation")

            # 改造状态 s[b,t]
            for b in B:
                for t in T:
                    renovation_starts = []
                    for dur in range(1, min(self.renovation_period + 1, t + 1)):
                        if t - dur + 1 >= 1:
                            renovation_starts.append(x[b, t - dur + 1])
                    if renovation_starts:
                        model.addConstr(s[b, t] == gp.quicksum(renovation_starts), name=f"renovation_status_{b}_{t}")
                    else:
                        model.addConstr(s[b, t] == 0, name=f"renovation_status_{b}_{t}")

            # 是否已完成改造 is_renovated[b,t]
            for b in B:
                for t in T:
                    completed_sum = gp.quicksum(x[b, tau] for tau in range(1, t - self.renovation_period + 1) if tau >= 1)
                    model.addConstr(is_renovated[b, t] <= completed_sum, name=f"is_renovated_upper_{b}_{t}")
                    model.addConstr(is_renovated[b, t] * 100 >= completed_sum, name=f"is_renovated_lower_{b}_{t}")

            # 产能约束
            for b in B:
                for t in T:
                    model.addConstr(
                        q[b, t] <= self.workload_original[b - 1] * (1 - s[b, t]) * (1 - is_renovated[b, t]) +
                                   self.workload_enhanced[b - 1] * (1 - s[b, t]) * is_renovated[b, t],
                        name=f"capacity_constraint_{b}_{t}"
                    )

            # 需求平衡 + 同时改造上限
            for t in T:
                model.addConstr(gp.quicksum(q[b, t] for b in B) + z[t] == demand[t], name=f"demand_balance_{t}")
                model.addConstr(gp.quicksum(s[b, t] for b in B) <= self.max_simultaneous_renovations,
                                name=f"max_simultaneous_renovation_{t}")

            # 末尾不能开工（保证完工）
            for b in B:
                for t in range(self.planning_horizon - self.renovation_period + 2, self.planning_horizon + 1):
                    model.addConstr(x[b, t] == 0, name=f"complete_renovation_{b}_{t}")

            # 延迟比例上限
            for t in T:
                model.addConstr(z[t] <= 0.2 * demand[t], name=f"delay_tolerance_{t}")

            # 成本函数
            renovation_cost = gp.quicksum(self.c_ren * x[b, t] for b in B for t in T)
            operating_cost = gp.LinExpr(0)
            for b in B:
                for t in T:
                    unit_cost = self.c_base * (1 - self.eta) * is_renovated[b, t] + self.c_base * (1 - is_renovated[b, t]) + self.c_tran
                    operating_cost += q[b, t] * unit_cost
            delay_cost = gp.quicksum(self.c_delay * z[t] for t in T)
            total_cost = renovation_cost + operating_cost + delay_cost


            Tlen = float(self.planning_horizon)
            sum_r = gp.quicksum(r[t] for t in T)
            sum_r2 = gp.quicksum(r[t] * r[t] for t in T)
            var_expr = ((1.0 / Tlen) * sum_r2 - (1.0 / (Tlen * Tlen))
                        * (sum_r * sum_r))
            # 总目标
            total_obj = total_cost + self.balance_gamma * var_expr
            model.setObjective(total_obj, GRB.MINIMIZE)
            model.setParam('OutputFlag', 0)

            # 保存变量
            self.multi_period_model = model
            self.x_vars = x
            self.s_vars = s
            self.q_vars = q
            self.z_vars = z
            self.demand_vars = demand
            self.is_renovated_vars = is_renovated
            self.r_vars = r
            return model
        except gp.GurobiError as e:
            logging.error(f"Error building multi-period model: {e}")
            return None

    def solve_multi_period(self, demand_values, true_demand_values=None):
        if len(demand_values) != self.planning_horizon:
            logging.error(f"需求预测长度({len(demand_values)})与规划期长度({self.planning_horizon})不匹配")
            return None

        try:
            for t in range(1, self.planning_horizon + 1):
                total_demand_t = float(np.sum(demand_values[t - 1]))
                self.demand_vars[t].lb = total_demand_t
                self.demand_vars[t].ub = total_demand_t

            self.multi_period_model.optimize()

            if self.multi_period_model.status == GRB.OPTIMAL:
                B = range(1, self.n_terminals + 1)
                T = range(1, self.planning_horizon + 1)

                x_solution = np.zeros((self.n_terminals, self.planning_horizon))
                s_solution = np.zeros((self.n_terminals, self.planning_horizon))
                is_renovated_solution = np.zeros((self.n_terminals, self.planning_horizon))
                q_solution = np.zeros((self.n_terminals, self.planning_horizon))
                z_solution = np.zeros(self.planning_horizon)
                r_solution = np.zeros(self.planning_horizon)

                for b in B:
                    for t in T:
                        x_solution[b - 1, t - 1] = round(self.x_vars[b, t].x)
                        s_solution[b - 1, t - 1] = round(self.s_vars[b, t].x)
                        q_solution[b - 1, t - 1] = self.q_vars[b, t].x
                        is_renovated_solution[b - 1, t - 1] = round(self.is_renovated_vars[b, t].x)
                for t in T:
                    z_solution[t - 1] = self.z_vars[t].x
                    r_solution[t - 1] = self.r_vars[t].x

                renovation_value = np.sum(x_solution) * self.c_ren
                delay_value = np.sum(z_solution) * self.c_delay

                operating_value = 0.0
                for b in range(self.n_terminals):
                    for t in range(self.planning_horizon):
                        if s_solution[b, t] > 0.5:
                            # 正在改造 → 禁止作业，所以这里应该保证 q=0
                            continue
                        elif is_renovated_solution[b, t] > 0.5:
                            # 已完工 → 成本降低
                            unit_cost = self.c_base * (1 - self.eta) + self.c_tran
                        else:
                            # 未改造 → 原始成本
                            unit_cost = self.c_base + self.c_tran
                        operating_value += q_solution[b, t] * unit_cost

                total_value = renovation_value + operating_value + delay_value
                # 计算均衡性方差（与建模时一致）：Var(R)
                Rt = r_solution
                Rt_mean = float(np.mean(Rt)) if Rt.size > 0 else 0.0
                balance_variance = float(np.mean((Rt - Rt_mean) ** 2)) if Rt.size > 0 else 0.0
                balance_penalty = float(self.balance_gamma * balance_variance)

                # 含惩罚的总目标（与 Gurobi 目标一致）
                total_objective_with_penalty = float(self.multi_period_model.objVal)

                # SPO 联训用：若给了真实需求，则再求一次真实需求下的最优用于 decision_error
                predicted_opt_obj = total_objective_with_penalty
                decision_error = 0.0
                if true_demand_values is not None:
                    for t in range(1, self.planning_horizon + 1):
                        total_true_t = float(np.sum(true_demand_values[t - 1]))
                        self.demand_vars[t].lb = total_true_t
                        self.demand_vars[t].ub = total_true_t
                    self.multi_period_model.optimize()
                    if self.multi_period_model.status == GRB.OPTIMAL:
                        true_opt_obj = float(self.multi_period_model.objVal)
                        eps = 1e-8
                        decision_error = max(0.0, (predicted_opt_obj - true_opt_obj) / (true_opt_obj + eps))
                H_star = int(np.round(np.sum(x_solution)))  # x[b,t] 每个箱区只会在开工月=1，其余=0

                return {
                    "status": "optimal",
                    "renovation_decisions": x_solution,
                    "renovation_status": s_solution,
                    "is_renovated": is_renovated_solution,
                    "workload_allocation": q_solution,
                    "delay_amounts": z_solution,
                    "demand_values": demand_values,
                    "total_cost": total_value,
                    "total_objective_with_penalty": total_objective_with_penalty,
                    "renovation_cost": renovation_value,
                    "operating_cost": operating_value,
                    "delay_cost": delay_value,
                    "starts_per_period": r_solution,
                    "balance_variance": balance_variance,
                    "balance_penalty": balance_penalty,
                    "decision_error": decision_error,
                    "num_renovations": H_star,
                }
            else:
                logging.warning(f"优化未成功，状态：{self.multi_period_model.status}")
                return {"status": f"not_optimal: {self.multi_period_model.status}"}
        except gp.GurobiError as e:
            logging.error(f"Error solving multi-period model: {e}")
            return {"status": f"error: {str(e)}"}

    def solve_with_true_demand(self, true_demand_values):
        if len(true_demand_values) != self.planning_horizon:
            logging.error(f"真实需求长度({len(true_demand_values)})与规划期长度({self.planning_horizon})不匹配")
            return None
        try:
            for t in range(1, self.planning_horizon + 1):
                total_demand_t = float(np.sum(true_demand_values[t - 1]))
                self.demand_vars[t].lb = total_demand_t
                self.demand_vars[t].ub = total_demand_t
            self.multi_period_model.optimize()
            if self.multi_period_model.status == GRB.OPTIMAL:
                return {"status": "optimal", "total_cost": self.multi_period_model.objVal}
            else:
                logging.warning(f"真实需求下优化未成功，状态：{self.multi_period_model.status}")
                return {"status": f"not_optimal: {self.multi_period_model.status}"}
        except gp.GurobiError as e:
            logging.error(f"真实需求求解错误: {e}")
            return {"status": f"error: {str(e)}"}

    def get_renovation_schedule(self, result):
        if result["status"] != "optimal":
            return None
        schedule = []
        x_solution = result["renovation_decisions"]
        for b in range(self.n_terminals):
            start = None
            for t in range(self.planning_horizon):
                if x_solution[b, t] > 0.5:
                    start = t + 1
                    break
            schedule.append({
                "terminal": b + 1,
                "start_month": start,
                "end_month": (start + self.renovation_period - 1) if start else None
            })
        return schedule

    def get_workload_statistics(self, result):
        if result["status"] != "optimal":
            return None
        q_solution = result["workload_allocation"]
        z_solution = result["delay_amounts"]
        demand_values = result["demand_values"]
        s_solution = result["renovation_status"]
        is_ren = result.get("is_renovated", np.zeros_like(s_solution))  # 兼容旧结果

        actual_n_terminals = q_solution.shape[0]
        monthly_stats = []
        for t in range(self.planning_horizon):
            total_allocated = float(np.sum(q_solution[:, t]))
            delay = float(z_solution[t])

            # 逐箱区累加“可用产能”（改造中=0；已完工=新产能；未完工=老产能）
            total_capacity = 0.0
            for b in range(actual_n_terminals):
                if s_solution[b, t] > 0.5:
                    cap_bt = 0.0
                elif is_ren[b, t] > 0.5:
                    cap_bt = float(self.workload_enhanced[b])
                else:
                    cap_bt = float(self.workload_original[b])
                total_capacity += cap_bt

            utilization = (total_allocated / total_capacity) if total_capacity > 1e-9 else 0.0
            monthly_stats.append({
                "month": t + 1,
                "demand": demand_values[t],
                "allocated": total_allocated,
                "delayed": delay,
                "utilization": utilization
            })
        return monthly_stats


# =============== SPO 联训器===============
class SPOTrainer:
    def __init__(self, predictor, optimizer, terminal_optimizer, lambda_spo=2.0, lambda_pred=1.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = predictor.to(self.device)
        self.optimizer = optimizer
        self.terminal_optimizer = terminal_optimizer
        self.lambda_spo = lambda_spo
        self.lambda_pred = lambda_pred

    def _equal_split_matrix(self, s_vec, num_zones, horizon):
        """
        将每期总需求 s_vec (shape: [H]) 均分到各箱区，得到 (H, NUM_ZONES) 的矩阵。
        说明：你的优化模型只用总需求(∑_b q_{b,t})，均分不会影响可行性/目标值。
        """
        s_vec = np.asarray(s_vec, dtype=float).reshape(-1)  # [H]
        per_zone = (s_vec / float(num_zones)).reshape(-1, 1)  # [H,1]
        return np.repeat(per_zone, num_zones, axis=1)  # [H,NUM_ZONES]

    def train(self, train_loader, epochs=100, lr=0.0015):
        """
        使用 MSE + SPO+ 近似 的联合损失：
          - MSE: 保持预测精度
          - SPO+: L_spo = sum_t (2 * s_hat_t - s_true_t) * R*_t( 2*s_hat - s_true )
            其中 s_hat_t = sum_b D_hat[t,b]（每期总需求），R*_t 是在镜像需求下的“每期开工数”最优解。
        注意：R* 视为常数系数，梯度只流向 s_hat（从而流向 D_hat）。
        """
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.predictor.parameters(), lr=lr, weight_decay=1e-4)
        horizon = self.terminal_optimizer.planning_horizon
        num_zones = self.terminal_optimizer.n_terminals

        for epoch in range(epochs):
            self.predictor.train()
            epoch_pred_loss_sum = 0.0
            epoch_spo_loss_sum = 0.0
            batch_count = 0

            for x_batch, y_batch in train_loader:
                # 小批量里逐个样本处理（保持与你现有代码一致，便于调试）
                for i in range(x_batch.shape[0]):
                    x_single = x_batch[i].unsqueeze(0).to(self.device)  # [1, seq, feat]
                    y_single = y_batch[i].unsqueeze(0).to(self.device)  # [1, H, Z]

                    # 预测
                    preds_single = self.predictor(x_single)  # [1, H, Z]
                    pred_loss = criterion(preds_single, y_single)

                    # ---- SPO+ 近似：构造可微损失 ----
                    # 每期总需求（预测/真实），shape: [H]
                    s_hat = preds_single.sum(dim=2).squeeze(0)  # sum over zones
                    s_true = y_single.sum(dim=2).squeeze(0)

                    # 镜像需求（防止负值，做 ReLU 截断）
                    s_cf = torch.clamp(2.0 * s_hat - s_true, min=0.0)  # [H]

                    # 转为 numpy，拼出 (H, Z) 的需求矩阵；优化器只看总和，均分即可
                    s_cf_np = s_cf.detach().cpu().numpy()
                    D_cf_np = self._equal_split_matrix(s_cf_np, num_zones, horizon)  # [H, Z]

                    # 求解镜像需求下的最优，取每期开工数 R*_t
                    spo_cf = self.terminal_optimizer.solve_multi_period(D_cf_np)
                    if not spo_cf or spo_cf.get("status") != "optimal":
                        # 若优化失败，则该样本的 SPO 损失置 0，避免中断训练
                        spo_plus_loss = torch.zeros((), device=self.device, dtype=torch.float32)
                    else:
                        R_cf = spo_cf["starts_per_period"]  # np.array([H])
                        R_cf_t = torch.tensor(R_cf, dtype=torch.float32, device=self.device)
                        # L_spo = sum_t (2 s_hat_t - s_true_t) * R*_t
                        spo_plus_loss = torch.dot(2.0 * s_hat - s_true, R_cf_t)

                    total_loss = self.lambda_pred * pred_loss + self.lambda_spo * spo_plus_loss

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    epoch_pred_loss_sum += float(pred_loss.detach().cpu())
                    epoch_spo_loss_sum += float(spo_plus_loss.detach().cpu())
                    batch_count += 1

            # 统计输出
            avg_pred = epoch_pred_loss_sum / max(1, batch_count)
            avg_spo = epoch_spo_loss_sum / max(1, batch_count)
            print(f"Epoch [{epoch + 1}/{epochs}] MSE: {avg_pred:.4f} | SPO+: {avg_spo:.4f}")


# ================= 主流程 =================
def main():
    Path("results").mkdir(exist_ok=True)

    # 数据加载（注意：若数据列数小于 NUM_ZONES 会自动只取现有列）
    df = pd.read_csv(DATA_FILE)
    full_data = df.values[:, :NUM_ZONES]

    # 时序特征
    months = np.tile(np.arange(1, 13), 12)[:144]
    month_sin = np.sin(2 * np.pi * months / 12)
    month_cos = np.cos(2 * np.pi * months / 12)
    year_index = np.repeat(np.arange(12), 12)[:144] / 12
    long_trend = np.arange(144) / 144

    # 归一化
    processor = DataProcessor()
    normalized_data, min_val, max_val = processor.normalize(full_data)

    # 训练（1-8年）
    train_data = normalized_data[:96]
    train_msin, train_mcos, train_yidx = month_sin[:96], month_cos[:96], year_index[:96]
    train_trend = long_trend[:96]

    # 测试（9-12年）
    test_data = normalized_data[96:]
    test_msin, test_mcos, test_yidx = month_sin[96:], month_cos[96:], year_index[96:]
    test_trend = long_trend[96:]

    train_dataset = TerminalDataset(train_data, train_msin, train_mcos, train_yidx, train_trend,
                                    seq_length=60, forecast_horizon=12)
    test_dataset  = TerminalDataset(test_data,  test_msin,  test_mcos,  test_yidx,  test_trend,
                                    seq_length=60, forecast_horizon=12)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # 选择模型
    model = MultiStepLSTMPredictor(
        input_size=NUM_ZONES + 4,   # zones + 4个时序特征
        hidden_size=64,
        num_layers=2,
        output_size=NUM_ZONES,
        forecast_horizon=12,
        dropout=0.2
    )
    # 或者：
    #model = MultiStepGRUPredictor(
    #     input_size=NUM_ZONES + 4,
    #     hidden_size=64,
    #     num_layers=2,
    #     output_size=NUM_ZONES,
    #     forecast_horizon=12,
    #     dropout=0.0
    # )

    criterion = nn.MSELoss()
    optimizer_nn = optim.Adam(model.parameters(), lr=0.0015)

    # 预训练 100 轮（1-8年）
    loss_list = []
    for epoch in range(100):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer_nn.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer_nn.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        loss_list.append(avg_loss)
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1}/100] Loss: {avg_loss:.4f}")

    # 累计微调 + 一次性预测12个月
    seq_length = 60
    horizon = 12

    def build_dataset_until(end_idx):
        data = normalized_data[:end_idx]
        msin = month_sin[:end_idx]
        mcos = month_cos[:end_idx]
        yidx = year_index[:end_idx]
        trnd = long_trend[:end_idx]
        return TerminalDataset(data, msin, mcos, yidx, trnd, seq_length=seq_length, forecast_horizon=horizon)

    def predict_next_12(end_idx):
        x_hist = normalized_data[end_idx - seq_length:end_idx, :NUM_ZONES]
        m_sin = month_sin[end_idx - seq_length:end_idx].reshape(-1, 1)
        m_cos = month_cos[end_idx - seq_length:end_idx].reshape(-1, 1)
        y_idx = year_index[end_idx - seq_length:end_idx].reshape(-1, 1)
        trnd = long_trend[end_idx - seq_length:end_idx].reshape(-1, 1)
        x_all = np.hstack([x_hist, m_sin, m_cos, y_idx, trnd])
        x_tensor = torch.tensor(x_all, dtype=torch.float32).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            preds_norm = model(x_tensor).cpu().numpy()[0]  # (12, NUM_ZONES)
        preds_denorm = processor.denormalize(preds_norm, min_val, max_val)
        return preds_denorm

    opt_finetune = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    folds = [
        (96, 108, "4-8年预测9年"),
        (108, 120, "5-9年预测10年"),
        (120, 132, "6-10年预测11年"),
        (132, 144, "7-11年预测12年"),
    ]
    preds_last_fold = None

    for train_end, test_end, tag in folds:
        train_dataset_cum = build_dataset_until(train_end)
        train_loader_cum = DataLoader(train_dataset_cum, batch_size=8, shuffle=True)

        model.train()
        for ep in range(30):
            total_loss = 0.0
            for xb, yb in train_loader_cum:
                opt_finetune.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                opt_finetune.step()
                total_loss += loss.item()

        preds_12 = predict_next_12(train_end)
        y_true_12 = full_data[train_end:test_end, :NUM_ZONES]

        mae = mean_absolute_error(y_true_12, preds_12)
        rmse = np.sqrt(mean_squared_error(y_true_12, preds_12))
        r2 = r2_score(y_true_12, preds_12)
        print(f"[{tag}] MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

        if tag == "7-11年预测12年":
            preds_last_fold = preds_12

    # =============== 送入优化模型（一次） ===============
    terminal_type_manager = TerminalTypeManager()

    # ⚠ 若 NUM_ZONES != 20，请同步修改下面三组分区范围，否则会 KeyError
    terminal_type_manager.set_terminal_types_by_ranges(
        nearshore_range=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        midshore_range=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        offshore_range=[21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    )
    zone_types = terminal_type_manager.get_zone_types()

    phi_k = {1: 1.2, 2: 1.0, 3: 0.8}
    eta_k = {1: 1.6, 2: 1.5, 3: 1.4}

    w_origin = np.array([phi_k[zone_types[i + 1]] * 7000 for i in range(NUM_ZONES)])
    w_enhanced = np.array([phi_k[zone_types[i + 1]] * eta_k[zone_types[i + 1]] * 7000 for i in range(NUM_ZONES)])

    optimizer = TerminalOptimizer(
        w_origin=w_origin,
        w_enhanced=w_enhanced,
        n_terminals=NUM_ZONES,
        planning_horizon=12,
        balance_gamma=0
    )

    result = optimizer.solve_multi_period(preds_last_fold)
    schedule = optimizer.get_renovation_schedule(result)
    if schedule is not None:
        print("📌 改造计划：")
        for item in schedule:
            print(item)
    else:
        print("⚠️ 未生成改造计划（优化未成功或无可行解）")

    stats = optimizer.get_workload_statistics(result)
    if stats is not None:
        print("\n📊 每月预测作业量与优化结果：")
        for t, month_stat in enumerate(stats, 1):
            demand = month_stat['demand']
            demand_total = np.sum(demand)
            allocated = month_stat['allocated']
            delayed = month_stat['delayed']
            utilization = month_stat['utilization'] * 100
            print(f"Month {t}: TotalDemand={demand_total:.1f}, Allocated={allocated:.1f}, "
                  f"Delayed={delayed:.1f}, Utilization={utilization:.1f}%")
    else:
        print("⚠️ 无法生成每月作业统计（优化未成功）")

    # 展示：含惩罚的总目标 + 分项成本 + 惩罚项
    print(f"\n🎯 总目标值(含均衡惩罚): {result['total_objective_with_penalty']:.2f}")
    print(f"   固定改造成本: {result['renovation_cost']:.2f}")
    print(f"   运营成本: {result['operating_cost']:.2f}")
    print(f"   延迟成本: {result['delay_cost']:.2f}")
    if 'balance_penalty' in result:
        print(f"   均衡惩罚(γ·Var): {result['balance_penalty']:.4f}  (γ={optimizer.balance_gamma})")

    # =============== NRE 计算与对比 ===============
    # 预测第12年
    x_hist = normalized_data[72:132][:, :NUM_ZONES]
    m_sin = month_sin[72:132].reshape(-1, 1)
    m_cos = month_cos[72:132].reshape(-1, 1)
    y_idx = year_index[72:132].reshape(-1, 1)
    trend = long_trend[72:132].reshape(-1, 1)
    x_all = np.hstack([x_hist, m_sin, m_cos, y_idx, trend])
    x_tensor = torch.tensor(x_all, dtype=torch.float32).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        preds = model(x_tensor).cpu().numpy()[0]
    preds_denorm = processor.denormalize(preds, min_val, max_val)

    y_true = df.values[132:144, :NUM_ZONES]

    result_opt = optimizer.solve_multi_period(y_true)
    if result_opt["status"] == "optimal":
        optimal_obj = result_opt["total_objective_with_penalty"]
        print(f"用真实数据的最优目标(含惩罚): {optimal_obj:.2f}")
    else:
        print("⚠️ 真实数据优化失败")
        optimal_obj = None

    result_pred = optimizer.solve_multi_period(preds_denorm)
    if result_pred["status"] == "optimal":
        predicted_obj = result_pred["total_objective_with_penalty"]
        print(f"用预测数据的优化目标(含惩罚): {predicted_obj:.2f}")
    else:
        print("⚠️ 预测数据优化失败")
        predicted_obj = None


    if optimal_obj is not None and predicted_obj is not None:
        # 归一化遗憾值：使用“含惩罚的目标”，与训练中的 decision loss 一致
        nre = (predicted_obj - optimal_obj) / optimal_obj
        nre_abs = abs(nre)
        print(f"\n📈 归一化遗憾值 NRE: {nre_abs:.4f} ({nre_abs * 100:.2f}%)")
    else:
        print("⚠️ 无法计算NRE，优化失败")

    print("=== 第12年改造计划对比分析 ===")

    # 预测 vs 真实 方案对比
    optimizer_pred = TerminalOptimizer(w_origin=w_origin, w_enhanced=w_enhanced,
                                       n_terminals=NUM_ZONES, planning_horizon=12)
    result_pred = optimizer_pred.solve_multi_period(preds_denorm)

    optimizer_true = TerminalOptimizer(w_origin=w_origin, w_enhanced=w_enhanced,
                                       n_terminals=NUM_ZONES, planning_horizon=12)
    result_true = optimizer_true.solve_multi_period(y_true)

    if result_pred["status"] == "optimal" and result_true["status"] == "optimal":
        schedule_pred = optimizer_pred.get_renovation_schedule(result_pred)
        schedule_true = optimizer_true.get_renovation_schedule(result_true)

        print(f"\n✅ 两种优化都成功完成")
        print(f"预测数据总目标(含惩罚): {result_pred['total_objective_with_penalty']:.2f}")
        print(f"真实数据总目标(含惩罚): {result_true['total_objective_with_penalty']:.2f}")

        print(f"\n📋 改造计划详细对比:")
        print("=" * 80)
        print(f"{'箱区ID': >6} {'预测计划': >15} {'真实计划': >15} {'是否一致': >10}")
        print("=" * 80)

        differences = []
        same_decisions = 0

        for i in range(NUM_ZONES):
            pred_plan = schedule_pred[i]
            true_plan = schedule_true[i]
            pred_start = pred_plan['start_month'] if pred_plan['start_month'] is not None else "无改造"
            true_start = true_plan['start_month'] if true_plan['start_month'] is not None else "无改造"
            is_same = pred_plan['start_month'] == true_plan['start_month']
            same_indicator = "✓" if is_same else "✗"
            if is_same:
                same_decisions += 1
            else:
                differences.append({'terminal': i + 1, 'pred': pred_start, 'true': true_start})
            print(f"{i + 1: >6} {str(pred_start): >15} {str(true_start): >15} {same_indicator: >10}")

        print("=" * 80)
        print(f"决策一致的箱区数: {same_decisions}/{NUM_ZONES} ({same_decisions / NUM_ZONES * 100:.1f}%)")

        print(f"\n📅 按月分组的改造计划对比:")
        pred_monthly = {}
        for item in schedule_pred:
            if item['start_month'] is not None:
                pred_monthly.setdefault(item['start_month'], []).append(item['terminal'])

        true_monthly = {}
        for item in schedule_true:
            if item['start_month'] is not None:
                true_monthly.setdefault(item['start_month'], []).append(item['terminal'])

        all_months = sorted(set(list(pred_monthly.keys()) + list(true_monthly.keys())))
        print("-" * 60)
        print(f"{'月份': >4} {'预测计划': >25} {'真实计划': >25}")
        print("-" * 60)
        for month in all_months:
            pred_terminals = pred_monthly.get(month, [])
            true_terminals = true_monthly.get(month, [])
            pred_str = ', '.join(map(str, pred_terminals)) if pred_terminals else "无"
            true_str = ', '.join(map(str, true_terminals)) if true_terminals else "无"
            print(f"{month: >4} {pred_str: >25} {true_str: >25}")
        print("-" * 60)

        print(f"\n💰 成本结构详细对比:")
        print("-" * 70)
        print(f"{'成本类型': >15} {'预测方案': >15} {'真实方案': >15} {'差异': >15}")
        print("-" * 70)
        cost_items = [
            ('改造成本', result_pred['renovation_cost'], result_true['renovation_cost']),
            ('运营成本', result_pred['operating_cost'], result_true['operating_cost']),
            ('延迟成本', result_pred['delay_cost'], result_true['delay_cost']),
            ('总成本', result_pred['total_cost'], result_true['total_cost'])
        ]
        for name, pred_cost, true_cost in cost_items:
            diff = pred_cost - true_cost
            print(f"{name: >15} {pred_cost: >15.0f} {true_cost: >15.0f} {diff: >15.0f}")
        print("-" * 70)

        # 与训练一致，使用含惩罚的目标计算 NRE
        nre = (result_pred['total_objective_with_penalty'] - result_true['total_objective_with_penalty']) / result_true['total_objective_with_penalty']
        nre_abs = abs(nre)
        print(f"\n📈 归一化遗憾值 NRE: {nre_abs:.4f} ({nre_abs * 100:.2f}%)")
    else:
        print("⚠️ 优化失败，无法生成对比")
        if result_pred["status"] != "optimal":
            print(f"预测数据优化状态: {result_pred['status']}")
        if result_true["status"] != "optimal":
            print(f"真实数据优化状态: {result_true['status']}")

    print("\n" + "=" * 50)
    print("对比分析完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()
