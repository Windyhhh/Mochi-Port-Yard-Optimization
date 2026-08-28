<div align="center">

# 🏗️ Mochi-Port-Yard-Optimization

### Deep learning + operations research for port yard planning.

LSTM / GRU workload forecasting (MAE 0.12) and Gurobi-driven optimal redevelopment — cutting total cost 30%+.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gurobi](https://img.shields.io/badge/Gurobi-Optimizer-2EA44F)](https://www.gurobi.com/)

</div>

---

**Mochi-Port-Yard-Optimization** is an end-to-end intelligent port-yard planning system that combines **deep-learning forecasting** with **operations-research optimization**: an **LSTM / GRU** model predicts yard workload (MAE 0.12) and the **Gurobi** engine produces optimal redevelopment plans, cutting total cost by **30%+**.

> [!NOTE]
> 中文项目：Mochi 码头堆场智能优化——LSTM/GRU 作业量预测（MAE 0.12）+ Gurobi 优化决策，改造总成本降低 30%+。

---

## Features

- **High-precision forecasting** — LSTM / GRU, MAE 0.12.
- **Optimal decisions** — Gurobi optimization engine, 30%+ cost reduction.
- **Visualization** — plan comparison, cost analysis, performance evaluation.
- **Extensible** — modular; swap prediction models / optimization algorithms.

---

## Quickstart

```bash
git clone https://github.com/Windyhhh/Mochi-Port-Yard-Optimization.git
cd Mochi-Port-Yard-Optimization

pip install -r requirements.txt

python src/train_model.py       # train the workload predictor
python src/optimize.py          # run Gurobi redevelopment optimization
```

---

## Project Structure

```
Mochi-Port-Yard-Optimization/
├── src/
│   ├── model/                 # LSTM / GRU predictor
│   ├── optimization/          # Gurobi models
│   └── visualization/
├── data/                      # yard workload data
└── docs/                      # blog
```

---

## License

MIT — free to use, modify and distribute.
