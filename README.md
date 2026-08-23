# 🚢 Mochi Port Yard Optimization | Mochi 码头堆场智能优化系统

> **LSTM demand forecasting + Gurobi mixed-integer optimization for multi-period port terminal renovation planning. Predict-Then-Optimize (PTO) and Smart Predict-then-Optimize (SPO) with NRE evaluation.**
>
> LSTM 需求预测 + Gurobi 混合整数规划，多周期码头箱区改造优化。支持 PTO（先预测后优化）和 SPO（预测优化联合训练），NRE 归一化遗憾值评估。

---

## 🌟 Why This Project? | 项目亮点

Port terminals face a critical challenge: **when and which yard zones to renovate** to maximize throughput while minimizing total cost (renovation + operating + delay costs). This project implements a **dual-engine optimization system** combining **LSTM/GRU time-series forecasting** with **Gurobi mixed-integer programming (MIP)** for multi-period renovation planning. It supports both standard **Predict-Then-Optimize (PTO)** and advanced **Smart Predict-then-Optimize (SPO)** with decision-focused learning, evaluated via **Normalized Regret Error (NRE)**.

码头面临的关键挑战：**何时、改造哪些箱区**，以最大化吞吐量同时最小化总成本（改造+运营+延迟成本）。本项目实现了**双引擎优化系统**，结合 **LSTM/GRU 时间序列预测** 与 **Gurobi 混合整数规划（MIP）** 进行多周期改造规划。支持标准 **PTO（先预测后优化）** 和先进的 **SPO（预测优化联合训练）决策聚焦学习**，通过 **NRE（归一化遗憾值）** 评估。

| Feature | Details |
|---------|---------|
| **Forecasting** | LSTM / GRU multi-step predictor (12-month horizon) |
| **Optimization** | Gurobi MIP multi-period renovation scheduling |
| **Paradigms** | PTO (Predict-Then-Optimize) + SPO (Smart PTO) |
| **Evaluation** | NRE (Normalized Regret Error) |
| **Data** | 12 years of monthly terminal data (5–30 zones) |
| **Features** | Seasonal (sin/cos), year index, long-term trend |
| **Zones** | Nearshore / Midshore / Offshore with type-specific parameters |
| **Constraints** | Max simultaneous renovations, min renovations, delay tolerance |

---

## 🏗️ Architecture | 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                   Historical Terminal Data                    │
│         12 years × monthly × N zones (5/10/15/20/25/30)    │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              Feature Engineering & Normalization               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Zone workloads (N zones)                                │  │
│  │  + month_sin, month_cos (seasonality)                   │  │
│  │  + year_index (annual trend)                             │  │
│  │  + long_trend (global trend)                             │  │
│  │  → Min-Max normalization                                 │  │
│  └─────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│   LSTM Predictor │ │   GRU Predictor  │ │   ARIMA Baseline │
│  (MultiStepLSTM) │ │  (MultiStepGRU)  │ │  (ARIMA-PTO.py)  │
│  2 layers, 64h   │ │  2 layers, 64h   │ │  Statistical      │
│  dropout=0.2      │ │  dropout=0.0      │ │  comparison       │
└────────┬─────────┘ └──────────────────┘ └──────────────────┘
         │ 12-month forecast
         ▼
┌─────────────────────────────────────────────────────────────┐
│              Gurobi MIP Multi-Period Optimizer                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Decision Variables:                                     │  │
│  │  • x[b,t]: binary — renovate zone b in month t         │  │
│  │  • s[b,t]: binary — zone b under renovation in month t │  │
│  │  • q[b,t]: continuous — workload allocated to zone b   │  │
│  │  • z[t]: continuous — delayed workload in month t       │  │
│  │                                                          │  │
│  │  Objective: MINIMIZE total cost                         │  │
│  │  = renovation_cost + operating_cost + delay_cost        │  │
│  │                                                          │  │
│  │  Constraints:                                           │  │
│  │  • Each zone renovated at most once                     │  │
│  │  • Min 5 renovations in 12 months                       │  │
│  │  • Max 5 simultaneous renovations                        │  │
│  │  • Capacity: original or enhanced (post-renovation)     │  │
│  │  • Demand balance: allocated + delayed = demand          │  │
│  │  • Delay tolerance: ≤ 20% of demand                     │  │
│  └─────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              NRE Evaluation & Comparison                       │
│  NRE = |Cost(predicted_demand) - Cost(true_demand)| / Cost(true_demand)
│  Measures how much optimization quality degrades due to prediction error
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 Two Optimization Paradigms | 两种优化范式

### PTO (Predict-Then-Optimize) | 先预测后优化

The standard two-stage approach:
1. **Train predictor** to minimize prediction error (MSE)
2. **Feed predictions** to the optimizer
3. **Evaluate** using true demand

Simple and interpretable, but suboptimal because prediction error doesn't directly correlate with optimization quality.

### SPO (Smart Predict-then-Optimize) | 预测优化联合训练

Decision-focused learning:
1. **Train predictor** with a composite loss: `λ_pred × MSE + λ_spo × DecisionLoss`
2. **DecisionLoss** measures the impact of prediction errors on the optimization objective
3. The predictor learns to make errors that matter less for the downstream optimization

This directly optimizes for end-to-end decision quality, not just prediction accuracy.

---

## 📊 Gurobi MIP Formulation | Gurobi 混合整数规划模型

### Decision Variables | 决策变量

| Variable | Type | Description |
|----------|------|-------------|
| `x[b,t]` | Binary | Renovate zone `b` starting in month `t` |
| `s[b,t]` | Binary | Zone `b` is under renovation in month `t` |
| `q[b,t]` | Continuous | Workload allocated to zone `b` in month `t` |
| `z[t]` | Continuous | Delayed (unmet) workload in month `t` |
| `is_renovated[b,t]` | Binary | Zone `b` has completed renovation by month `t` |

### Objective | 目标函数

```
Minimize:  Σ_{b,t} c_ren · x[b,t]           (renovation cost)
         + Σ_{b,t} c_op[b,t] · q[b,t]       (operating cost)
         + Σ_t c_delay · z[t]                 (delay cost)
```

### Key Constraints | 关键约束

- **Unique renovation**: Each zone renovated at most once
- **Minimum renovations**: At least 5 zones in 12 months
- **Simultaneous limit**: At most 5 zones under renovation at once
- **Capacity**: `q[b,t] ≤ w_original[b]·(1-s)·(1-renovated) + w_enhanced[b]·(1-s)·renovated`
- **Demand balance**: `Σ_b q[b,t] + z[t] = demand[t]`
- **Delay tolerance**: `z[t] ≤ 0.2 · demand[t]`

---

## 🚀 Quick Start | 快速开始

### Installation | 安装

```bash
pip install -r requirements.txt
# Requires: torch, pandas, numpy, gurobipy, scikit-learn, matplotlib
```

> **Note**: Gurobi requires a valid license. Academic licenses are free.

### Run PTO (Predict-Then-Optimize) | 运行 PTO

```bash
cd src
python PTO.py
```

This will:
1. Load 12-year terminal data (30 zones default)
2. Train LSTM predictor on years 1–8
3. Cumulative fine-tuning on years 9–11
4. Predict year 12 (12-month horizon)
5. Solve Gurobi MIP for renovation schedule
6. Compare predicted vs. true demand optimization
7. Compute NRE

### Run SPO (Smart Predict-then-Optimize) | 运行 SPO

```bash
python SPO.py
```

### Run ARIMA Baseline | 运行 ARIMA 基线

```bash
python ARIMA-PTO.py
```

### Train Model Only | 仅训练模型

```bash
python trainer.py
```

### Visualize Results | 可视化结果

```bash
python visual.py
```

---

## 📁 Project Structure | 项目结构

```
Mochi-Port-Yard-Optimization/
├── src/
│   ├── PTO.py                    # Predict-Then-Optimize main (30KB)
│   ├── SPO.py                    # Smart PTO with decision loss (37KB)
│   ├── ARIMA-PTO.py              # ARIMA baseline comparison (16KB)
│   ├── model.py                   # LSTM/GRU model definitions
│   ├── trainer.py                 # Model training scripts
│   ├── visual.py                  # Visualization tools
│   ├── generate20.py              # Data generation script
│   └── utils/
│       ├── auc.py                 # AUC calculation
│       └── norm.py                # Normalization utilities
├── data/
│   ├── terminal_data.csv          # Terminal operational data
│   ├── yard_12years_5zones.csv   # 12-year, 5-zone data
│   ├── yard_12years_10zones.csv  # 12-year, 10-zone data
│   ├── yard_12years_15zones.csv  # 12-year, 15-zone data
│   ├── yard_12years_20zones.csv  # 12-year, 20-zone data
│   ├── yard_12years_25zones.csv  # 12-year, 25-zone data
│   └── yard_12years_30zones.csv  # 12-year, 30-zone data
├── results/
│   ├── multi_period_example.png   # Multi-period optimization visualization
│   ├── yard_plot_fixed.png        # Yard zone allocation plot
│   ├── train_loss.png             # Training loss curve
│   ├── sensitivity_costs_K1_15.csv # Sensitivity analysis
│   └── sensitivity_c_ren.csv      # Renovation cost sensitivity
├── mochi_blog.md                  # Technical blog (64KB)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📈 Zone Types & Parameters | 箱区类型与参数

### Zone Classification | 箱区分类

| Zone Type | Location | φ (capacity factor) | η (enhancement factor) |
|-----------|----------|---------------------|----------------------|
| **Nearshore** | Zones 1–10 | 1.2 | 1.6 |
| **Midshore** | Zones 11–20 | 1.0 | 1.5 |
| **Offshore** | Zones 21–30 | 0.8 | 1.4 |

### Cost Parameters | 成本参数

| Parameter | Value | Description |
|-----------|-------|-------------|
| `c_ren` | 3,000,000 | Renovation cost per zone |
| `c_delay` | 150 | Delay cost per unit workload |
| `c_base` | 100 | Base operating cost per unit |
| `c_tran` | 10 | Transportation cost per unit |
| `η` | 0.3 | Operating cost reduction post-renovation |
| `renovation_period` | 2 | Months per renovation |
| `max_simultaneous` | 5 | Max zones under renovation |

---

## 📊 Evaluation Metrics | 评估指标

### NRE (Normalized Regret Error) | 归一化遗憾值

```
NRE = |Cost(predicted_demand) - Cost(true_demand)| / Cost(true_demand)
```

Measures the relative cost increase when using predicted demand instead of true demand for optimization. Lower is better.

### Prediction Metrics | 预测指标

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of Determination)

### Optimization Metrics | 优化指标

- **Total cost** (renovation + operating + delay)
- **Renovation schedule** (which zones, when)
- **Workload utilization** (% of capacity used)
- **Delay ratio** (delayed / total demand)

---

## 📚 References | 参考文献

1. **Elmachtoub, A. N., & Grigas, P.** (2022). *Smart "Predict, then Optimize".* Management Science, 68(1), 9-26.
2. **Mandi, J., et al.** (2020). *Smart predict-and-optimize for hard combinatorial optimization problems.* AAAI.
3. **Gurobi Optimization.** (2024). *Gurobi Optimizer Reference Manual.*
4. **Hochreiter, S., & Schmidhuber, J.** (1997). *Long short-term memory.* Neural Computation, 9(8), 1735-1780.
5. **Bertsimas, D., & Kallus, N.** (2020). *From predictive to prescriptive analytics.* Management Science, 66(3), 1025-1044.

---

## 📄 License | 许可证

MIT License — free to use, modify, and distribute.

---

<div align="center">

**Built with 🚢 for smart port operations research**

[Report Bug](https://github.com/Windyhhh/Mochi-Port-Yard-Optimization/issues) · [Request Feature](https://github.com/Windyhhh/Mochi-Port-Yard-Optimization/issues)

</div>
