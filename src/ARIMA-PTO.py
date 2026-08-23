import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import gurobipy as gp
from gurobipy import GRB
import logging
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score

# ========= 全局可配置 =========
NUM_ZONES = 20                      # ← 想研究几个箱区就改这里（比如 5、10、20）
DATA_FILE = "yard_12years_20zones.csv"  # ← 换成对应列数的数据文件
PLANNING_HORIZON = 12               # 规划期（月份）
MIN_TOTAL_RENOVATIONS = 5           # 全年最少改造数（会自动 min(NUM_ZONES, 这个值)）
MAX_SIMULTANEOUS_RENOVATIONS = 5    # 同时最多开工改造数（会自动 cap 到 NUM_ZONES）
RENOVATION_PERIOD = 2               # 单个箱区改造历时（月）

# ========= 归一化工具（本脚本里只用于指标展示）=========
class DataProcessor:
    def normalize(self, data):
        min_val = data.min(axis=0)
        max_val = data.max(axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8), min_val, max_val

    def denormalize(self, data, min_val, max_val):
        return data * (max_val - min_val + 1e-8) + min_val

# ========= 箱区类型管理器 =========
class TerminalTypeManager:
    def __init__(self):
        self.zone_types = {}

    def set_terminal_types_by_ranges(self, nearshore_range=[], midshore_range=[], offshore_range=[]):
        for idx in nearshore_range:
            self.zone_types[idx] = 1
        for idx in midshore_range:
            self.zone_types[idx] = 2
        for idx in offshore_range:
            self.zone_types[idx] = 3

    def get_zone_types(self):
        return self.zone_types

# ========= ARIMA 预测器（按列做，列数=NUM_ZONES）=========
def arima_forecast(df, forecast_horizon=12, num_zones=NUM_ZONES, arima_order=(1,1,1), train_len=132):
    """
    用前 train_len 个月（默认 132=前11年）训练 ARIMA，
    外推 forecast_horizon 个月（默认 12）——对应第12年外推测试。
    """
    preds = []
    cols = df.columns[:num_zones]
    for col in cols:
        series = df[col].values[:train_len]  # ← 用前 11 年（1..132月）训练
        try:
            model = ARIMA(series, order=arima_order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=forecast_horizon)
        except Exception as e:
            logging.warning(f"[ARIMA] {col} 预测失败：{e}")
            last = series[-1] if len(series) else 0.0
            forecast = np.full(forecast_horizon, last)
        preds.append(forecast)
    return np.array(preds).T

# ========= 优化器 =========
class TerminalOptimizer:
    def __init__(self, w_origin, w_enhanced,
                 n_terminals=NUM_ZONES,
                 planning_horizon=PLANNING_HORIZON,
                 renovation_period=RENOVATION_PERIOD,
                 min_total_renovations=MIN_TOTAL_RENOVATIONS,
                 max_simultaneous_renovations=MAX_SIMULTANEOUS_RENOVATIONS):
        self.n_terminals = n_terminals
        self.planning_horizon = planning_horizon
        self.workload_original = w_origin[:n_terminals]
        self.workload_enhanced = w_enhanced[:n_terminals]
        self.c_ren = 3000000
        self.c_delay = 150
        self.c_base = 100
        self.c_tran = 10
        self.eta = 0.3
        self.renovation_period = renovation_period
        self.max_simultaneous_renovations = min(max_simultaneous_renovations, n_terminals)
        self.min_total_renovations = min(min_total_renovations, n_terminals)

        self.multi_period_model = None
        self.demand_vars = None
        self.x_vars = None
        self.s_vars = None
        self.q_vars = None
        self.z_vars = None
        self.is_renovated_vars = None
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

            # 每个箱区最多改造一次
            for b in B:
                model.addConstr(gp.quicksum(x[b, t] for t in T) <= 1, name=f"unique_renovation_{b}")

            # 全年最少改造数（避免 infeasible：用 min(n_terminals, 配置值)）
            if self.min_total_renovations > 0:
                model.addConstr(gp.quicksum(x[b, t] for b in B for t in T) >= self.min_total_renovations,
                                name="min_total_renovation")

            # 改造状态 s[b,t]：过去 renovation_period 月内是否开工
            for b in B:
                for t in T:
                    starts = []
                    for r in range(1, min(self.renovation_period + 1, t + 1)):
                        starts.append(x[b, t - r + 1])
                    model.addConstr(s[b, t] == (gp.quicksum(starts) if starts else 0),
                                    name=f"s_{b}_{t}")

            # 已完成改造 is_renovated[b,t]：在 t-RENOVATION_PERIOD 之前是否曾开工
            for b in B:
                for t in T:
                    completed_sum = gp.quicksum(x[b, tau] for tau in range(1, t - self.renovation_period + 1)
                                                if tau >= 1)
                    model.addConstr(is_renovated[b, t] <= completed_sum, name=f"isreno_up_{b}_{t}")
                    model.addConstr(is_renovated[b, t] * 100 >= completed_sum, name=f"isreno_lo_{b}_{t}")

            # 产能约束
            for b in B:
                for t in T:
                    model.addConstr(
                        q[b, t] <= self.workload_original[b - 1] * (1 - s[b, t]) * (1 - is_renovated[b, t]) +
                                   self.workload_enhanced[b - 1] * (1 - s[b, t]) * is_renovated[b, t],
                        name=f"cap_{b}_{t}"
                    )

            # 需求平衡 + 同期改造上限
            for t in T:
                model.addConstr(gp.quicksum(q[b, t] for b in B) + z[t] == demand[t], name=f"balance_{t}")
                model.addConstr(gp.quicksum(s[b, t] for b in B) <= self.max_simultaneous_renovations,
                                name=f"simreno_{t}")

            # 尾部禁止开工（确保能完工）
            for b in B:
                for t in range(self.planning_horizon - self.renovation_period + 2, self.planning_horizon + 1):
                    model.addConstr(x[b, t] == 0, name=f"no_late_start_{b}_{t}")

            # 延迟上限（这里 20%）
            for t in T:
                model.addConstr(z[t] <= 0.2 * demand[t], name=f"delay_cap_{t}")

            # 成本
            renovation_cost = gp.quicksum(3000000 * x[b, t] for b in B for t in T)
            operating_cost = gp.LinExpr(0)
            for b in B:
                for t in T:
                    unit_cost = 100 * (1 - 0.3) * is_renovated[b, t] + 100 * (1 - is_renovated[b, t]) + 10
                    operating_cost += q[b, t] * unit_cost
            delay_cost = gp.quicksum(150 * z[t] for t in T)
            model.setObjective(renovation_cost + operating_cost + delay_cost, GRB.MINIMIZE)
            model.setParam('OutputFlag', 0)

            self.multi_period_model = model
            self.x_vars, self.s_vars = x, s
            self.q_vars, self.z_vars = q, z
            self.demand_vars = demand
            self.is_renovated_vars = is_renovated
        except gp.GurobiError as e:
            logging.error(f"Build model error: {e}")

    def solve_multi_period(self, demand_values):
        if len(demand_values) != self.planning_horizon:
            logging.error(f"需求预测长度({len(demand_values)})与规划期({self.planning_horizon})不匹配")
            return {"status": "bad_input"}
        try:
            # 固定每月总需求为传入值按列求和
            for t in range(1, self.planning_horizon + 1):
                total_demand_t = float(np.sum(demand_values[t - 1]))
                self.demand_vars[t].lb = total_demand_t
                self.demand_vars[t].ub = total_demand_t
            self.multi_period_model.optimize()

            if self.multi_period_model.status != GRB.OPTIMAL:
                return {"status": f"not_optimal:{self.multi_period_model.status}"}

            B = range(1, self.n_terminals + 1)
            T = range(1, self.planning_horizon + 1)
            x_sol = np.zeros((self.n_terminals, self.planning_horizon))
            s_sol = np.zeros((self.n_terminals, self.planning_horizon))
            q_sol = np.zeros((self.n_terminals, self.planning_horizon))
            z_sol = np.zeros(self.planning_horizon)
            for b in B:
                for t in T:
                    x_sol[b - 1, t - 1] = round(self.x_vars[b, t].x)
                    s_sol[b - 1, t - 1] = round(self.s_vars[b, t].x)
                    q_sol[b - 1, t - 1] = self.q_vars[b, t].x
            for t in T:
                z_sol[t - 1] = self.z_vars[t].x

            renovation_value = np.sum(x_sol) * 3000000
            delay_value = np.sum(z_sol) * 150
            operating_value = 0
            for b in range(self.n_terminals):
                for t in range(self.planning_horizon):
                    is_reno = s_sol[b, t]
                    unit_cost = (100 * (1 - 0.3) + 10) if is_reno > 0 else (100 + 10)
                    operating_value += q_sol[b, t] * unit_cost
            total_value = renovation_value + operating_value + delay_value

            return {
                "status": "optimal",
                "renovation_decisions": x_sol,
                "renovation_status": s_sol,
                "workload_allocation": q_sol,
                "delay_amounts": z_sol,
                "demand_values": demand_values,
                "total_cost": total_value,
                "renovation_cost": renovation_value,
                "operating_cost": operating_value,
                "delay_cost": delay_value
            }
        except gp.GurobiError as e:
            logging.error(f"Solve error: {e}")
            return {"status": f"error:{str(e)}"}

    def get_renovation_schedule(self, result):
        if result.get("status") != "optimal":
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
        if result.get("status") != "optimal":
            return None
        q_solution = result["workload_allocation"]
        z_solution = result["delay_amounts"]
        demand_values = result["demand_values"]
        monthly_stats = []
        for t in range(self.planning_horizon):
            total_allocated = float(np.sum(q_solution[:, t]))
            delay = float(z_solution[t])
            demand_vec = demand_values[t]
            monthly_stats.append({
                "month": t + 1,
                "demand": demand_vec,
                "allocated": total_allocated,
                "delayed": delay,
            })
        return monthly_stats

def main():
    Path("results").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # === Step 1: 读数据 & 预测 ===
    df = pd.read_csv(DATA_FILE)
    assert df.shape[1] >= NUM_ZONES, f"数据列数({df.shape[1]}) < NUM_ZONES({NUM_ZONES})"
    df = df.iloc[:, :NUM_ZONES]

    # 第12年（外推12个月）ARIMA 预测：用前 11 年（1..132 月）训练，预测 133..144 月
    preds_denorm = arima_forecast(
        df, forecast_horizon=PLANNING_HORIZON, num_zones=NUM_ZONES, train_len=132
    )

    # 类型与能力参数 ===
    manager = TerminalTypeManager()
    manager.set_terminal_types_by_ranges(
        nearshore_range=[1,2,3,4,5,6,7],
        midshore_range=[8,9,10,11,12,13,14],
        offshore_range=[15,16,17,18,19,20]
    )
    zone_types = manager.get_zone_types()

    phi_k = {1: 1.2, 2: 1.0, 3: 0.8}   # 原始产能系数
    eta_k = {1: 1.6, 2: 1.5, 3: 1.4}   # 改造后增益系数
    w_origin   = np.array([phi_k[zone_types[i + 1]] * 7000 for i in range(NUM_ZONES)])
    w_enhanced = np.array([phi_k[zone_types[i + 1]] * eta_k[zone_types[i + 1]] * 7000 for i in range(NUM_ZONES)])

    optimizer = TerminalOptimizer(
        w_origin=w_origin,
        w_enhanced=w_enhanced,
        n_terminals=NUM_ZONES,
        planning_horizon=PLANNING_HORIZON,
        renovation_period=RENOVATION_PERIOD,
        min_total_renovations=MIN_TOTAL_RENOVATIONS,
        max_simultaneous_renovations=MAX_SIMULTANEOUS_RENOVATIONS
    )

    # 用“第12年预测”送入优化
    result = optimizer.solve_multi_period(preds_denorm)
    if result["status"] == "optimal":
        print(f"\n✅ 优化完成（基于第12年预测），ARIMA + PTO 总成本 = {result['total_cost']:.2f}")
    else:
        print(f"⚠️ 优化失败，状态：{result['status']}")
        return

    print("\n📈 第12年的预测作业量（每月×各箱区）:")
    for m, row in enumerate(preds_denorm, 1):
        print(f"Month {m}: " + ", ".join(f"{v:.1f}" for v in row))

    schedule = optimizer.get_renovation_schedule(result)
    print("\n📌 改造计划：")
    for item in schedule:
        print(item)

    stats = optimizer.get_workload_statistics(result)
    print("\n📊 每月需求与优化结果：")
    for t, s in enumerate(stats, 1):
        demand_total = float(np.sum(s["demand"]))
        print(f"Month {t}: TotalDemand={demand_total:.1f}, "
              f"Allocated={s['allocated']:.1f}, Delayed={s['delayed']:.1f}")

    # === 评估 ARIMA 在“第12年”的误差：与 133..144 月真实对齐 ===
    y_true = df.values[132:144, :NUM_ZONES]   # 第12年真实（133..144）
    y_pred = preds_denorm                     # 第12年预测
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print("\n📊 ARIMA 预测性能（第12年外推）:")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.4f}")

    # 成本拆分
    print(f"\n🎯 成本：Total={result['total_cost']:.2f} | "
          f"Renov={result['renovation_cost']:.2f} | "
          f"Opex={result['operating_cost']:.2f} | "
          f"Delay={result['delay_cost']:.2f}")

    # === 用第12年真实值做优化，计算 NRE（预测方案 vs 真实最优）===
    result_opt = optimizer.solve_multi_period(y_true)
    if result_opt["status"] == "optimal":
        optimal_cost = result_opt["total_cost"]
        nre = (result["total_cost"] - optimal_cost) / optimal_cost
        print(f"\n✅ 真实数据优化完成（第12年）：OptimalCost={optimal_cost:.2f}")
        print(f"✅ 归一化遗憾值 NRE: {nre:.4f}")
    else:
        print("\n⚠️ 真实数据优化失败，无法计算 NRE。")


if __name__ == "__main__":
    main()
