import torch
import torch.nn as nn
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

class MultiStepLSTMPredictor(nn.Module):
    def __init__(self,
                 input_size,        # 必须显式传入
                 hidden_size=64,
                 num_layers=2,
                 output_size=None,  # 必须显式传入
                 forecast_horizon=12,
                 dropout=0.2):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        out_tail = lstm_out[:, -self.forecast_horizon:, :]
        y = self.fc(out_tail)
        return y


class MultiStepGRUPredictor(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=64,
                 num_layers=2,
                 output_size=None,
                 forecast_horizon=12,
                 dropout=0.0):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.output_size = output_size

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, output_size * forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        out = out.view(-1, self.forecast_horizon, self.output_size)
        return out


def arima_forecast_batch(data_matrix, forecast_horizon=12, arima_order=(2, 1, 2)):
    forecasts = []
    for i in range(data_matrix.shape[1]):
        series = data_matrix[:, i]
        try:
            model = ARIMA(series, order=arima_order)
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=forecast_horizon)
        except Exception as e:
            print(f"[ARIMA] Zone {i} forecast failed: {e}")
            forecast = np.full(forecast_horizon, np.nan)
        forecasts.append(forecast)
    return np.stack(forecasts, axis=1)  # shape: [forecast_horizon, num_zones]
