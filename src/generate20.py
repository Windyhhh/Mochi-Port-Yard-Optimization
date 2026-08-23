import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# -----------------------
class SeasonalGrowthDataGenerator:
    def __init__(self, base_values=None, monthly_factors=None, years=12,
                 noise_scale=30, growth_range=(0.08, 0.10), seed=42):
        np.random.seed(seed)
        self.base_values = base_values or [2000]
        self.monthly_factors = monthly_factors or [0.78, 0.65, 0.83, 0.76, 0.88, 0.91, 0.85, 0.89, 0.97, 0.96, 1.0, 0.9]
        self.noise_scale = noise_scale
        self.years = years
        self.months = years * 12
        self.growth_range = growth_range

    def generate(self, n_yards=1):
        all_yards_data = {}
        n_groups = len(self.base_values)
        yards_per_group = n_yards // n_groups

        for group_idx, base_value in enumerate(self.base_values):
            for yard_in_group in range(1, yards_per_group + 1):
                yard_id = group_idx * yards_per_group + yard_in_group
                data = []
                current_base = base_value
                for year in range(self.years):
                    growth = np.random.uniform(*self.growth_range)
                    for month_idx in range(12):
                        seasonal_value = current_base * self.monthly_factors[month_idx]
                        noise = np.random.normal(0, self.noise_scale)
                        value = max(0, seasonal_value + noise)
                        data.append(int(round(value)))
                    current_base *= (1 + growth)
                all_yards_data[f"Yard_{yard_id}"] = data

        return pd.DataFrame(all_yards_data)

# -----------------------
# 生成数据
generator = SeasonalGrowthDataGenerator(base_values=[3500,3100,2800,2600,2400])
df = generator.generate(n_yards=30)
df.to_csv("yard_12years_30zones.csv", index=False)
# 可视化前6个箱区
plt.figure(figsize=(12, 6))
for i in range(min(6, df.shape[1])):
    plt.plot(df.iloc[:, i], label=df.columns[i])
plt.title("港口箱区12年作业量数据（四组不同base_value）")
plt.xlabel("月份")
plt.ylabel("作业量")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("yard_plot_fixed.png", dpi=300)
