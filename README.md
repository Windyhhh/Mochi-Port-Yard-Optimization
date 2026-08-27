<div align="center">

# ⚓ Mochi-Port-Yard-Optimization

### LSTM forecast + Gurobi MIP for multi-period port yard redesign.

PTO / SPO paradigms with NRE evaluation over 5–30 zone terminal data.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Gurobi](https://img.shields.io/badge/Gurobi-MIP-EE3124)](https://www.gurobi.com/)

</div>

---

**Mochi-Port-Yard-Optimization** optimizes port yard redesign with **LSTM prediction** + **Gurobi MIP** over multiple periods, using **PTO / SPO** paradigms and **NRE** evaluation — tested across 5–30 zone terminal layouts.

> [!NOTE]
> 中文项目：港口堆场优化——LSTM 预测 + Gurobi MIP 多期码头改造，PTO/SPO 范式 + NRE 评估。

---

## Quickstart

```bash
git clone https://github.com/Windyhhh/Mochi-Port-Yard-Optimization.git
cd Mochi-Port-Yard-Optimization

pip install -r requirements.txt

# Run the PTO (prediction-then-optimization) pipeline
python src/PTO.py

# ARIMA-based alternative
python src/ARIMA-PTO.py
```

Terminal data (5–30 zones over 12 years) ships in `data/`.

---

## Features

- **LSTM + Gurobi MIP** — forecast then optimize multi-period redesign.
- **PTO / SPO paradigms** — prediction-then-optimization vs. decision-focused.
- **NRE evaluation** — robustness metrics.
- **Scalable layouts** — 5–30 zone terminals.

---

## Project Structure

```
Mochi-Port-Yard-Optimization/
├── src/
│   ├── PTO.py            # prediction-then-optimization
│   └── ARIMA-PTO.py
├── data/                 # yard_12years_{5..30}zones.csv
├── results/              # multi-period examples, sensitivity
└── requirements.txt
```

---

## License

MIT — free to use, modify and distribute.
