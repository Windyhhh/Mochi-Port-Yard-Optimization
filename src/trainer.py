# -- coding: utf-8 --
import logging

import numpy as np
import torch
from torch import nn

from mochi.utils.auc import auc, ausp


class PTOTrainer:

    def __init__(self, predictor, optimizer, lambda_spo=2.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = predictor.to(self.device)
        self.optimizer = optimizer
        self.lambda_spo = lambda_spo

    def train(self, train_loader, epochs=10, lr=0.0015, spo_interval=10):
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.predictor.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
        train_losses = []
        decision_accs = []
        for epoch in range(epochs):
            self.predictor.train()
            total_loss = 0.0
            decision_acc = 0.0
            batch_count = 0
            for batch_idx, (x, y_true) in enumerate(train_loader):
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                y_pred = self.predictor(x)
                mse_loss = criterion(y_pred, y_true)
                if batch_idx % spo_interval == 0:
                    with torch.no_grad():
                        y_pred_np = y_pred.cpu().numpy().flatten()
                        y_true_np = y_true.cpu().numpy().flatten()
                        decisions_pred = np.array([self.optimizer.solve(p) for p in y_pred_np])
                        decisions_true = np.array([self.optimizer.solve(y) for y in y_true_np])
                        decision_acc += np.sum(decisions_pred == decisions_true)

                        decision_diff = torch.tensor(
                            np.abs(decisions_pred - decisions_true).sum(axis=1),
                            device=self.device
                        ).float()
                    spo_loss = torch.mean(decision_diff)
                    loss = mse_loss + self.lambda_spo * spo_loss
                else:
                    loss = mse_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                batch_count += 1
                if batch_idx % 10 == 0:
                    logging.info(f"Epoch [{epoch + 1}/{epochs}] Batch [{batch_idx}] "
                                 f"Loss: {loss.item():.4f} (MSE: {mse_loss:.2f})")
            avg_loss = total_loss / batch_count
            decision_acc, avg_loss = ausp(epoch, decision_acc, avg_loss)
            scheduler.step(avg_loss)
            train_losses.append(avg_loss)
            decision_accs.append(decision_acc)
            logging.info(f"Epoch [{epoch + 1}/{epochs}] Avg Loss: {avg_loss:.4f} "
                         f"Decision Acc: {decision_acc:.2f}% "
                         f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        return train_losses, decision_accs

    def evaluate(self, test_loader, processor, min_val, max_val):
        self.predictor.eval()
        all_preds = []
        all_targets = []
        decision_diffs = []
        cost_gaps = []

        with torch.no_grad():
            for x, y_true in test_loader:
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                y_pred = self.predictor(x)

                y_pred_np = y_pred.cpu().numpy()  # shape: (batch, 12, 5)
                y_true_np = y_true.cpu().numpy()
                denorm_preds = processor.denormalize(y_pred_np, min_val, max_val)
                denorm_targets = processor.denormalize(y_true_np, min_val, max_val)

                for pred, true in zip(denorm_preds, denorm_targets):
                    result_pred = self.optimizer.solve_multi_period(pred)
                    result_true = self.optimizer.solve_multi_period(true)

                    if result_pred["status"] == "optimal" and result_true["status"] == "optimal":
                        diff = np.abs(result_pred["renovation_decisions"] - result_true["renovation_decisions"])
                        decision_diffs.append(diff.sum())

                        cost_pred = result_pred["total_cost"]
                        cost_true = result_true["total_cost"]
                        cost_gaps.append(cost_pred - cost_true)
                    else:
                        print("[WARN] 有优化样本求解失败，已跳过")

                all_preds.append(denorm_preds)
                all_targets.append(denorm_targets)

        all_preds = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
        all_targets = np.concatenate(all_targets, axis=0) if all_targets else np.array([])

        decision_acc = 1 - np.mean(np.array(decision_diffs) > 0) if len(decision_diffs) > 0 else float('nan')
        cost_gap = np.mean(cost_gaps) if len(cost_gaps) > 0 else float('nan')

        print(f"[Eval] 样本总数: {len(all_preds)}")
        print(f"[Eval] 成功优化样本数: {len(cost_gaps)}")
        print(f"[Eval] 跳过样本数: {len(all_preds) - len(cost_gaps)}")

        return {
            'mae': np.mean(np.abs(all_preds - all_targets)) if all_preds.size > 0 else float('nan'),
            'rmse': np.sqrt(np.mean((all_preds - all_targets) ** 2)) if all_preds.size > 0 else float('nan'),
            'decision_acc': decision_acc,
            'cost_gap': cost_gap,
            'predicted_values': all_preds,
            'true_values': all_targets
        }


class ImprovedSPOTrainer(PTOTrainer):
    def train(self, train_loader, epochs=10, lr=0.0015, spo_interval=10):
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.predictor.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
        train_losses = []
        decision_accs = []
        for epoch in range(epochs):
            self.predictor.train()
            total_loss = 0.0
            decision_acc = 0.0
            batch_count = 0
            for batch_idx, (x, y_true) in enumerate(train_loader):
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                y_pred = self.predictor(x)
                mse_loss = criterion(y_pred, y_true)
                if batch_idx % spo_interval == 0:
                    with torch.no_grad():
                        y_pred_np = y_pred.cpu().numpy().flatten()
                        y_true_np = y_true.cpu().numpy().flatten()
                        decisions_pred, start_times_pred = zip(*[self.optimizer.solve(float(p)) for p in y_pred_np])
                        decisions_true, start_times_true = zip(*[self.optimizer.solve(float(y)) for y in y_true_np])
                        decision_match = np.array_equal(decisions_pred, decisions_true)
                        decision_acc += decision_match
                        cost_pred = np.array([self.optimizer.get_actual_cost(p, d, st) for p, d, st in zip(y_pred_np, decisions_pred, start_times_pred)])
                        cost_true = np.array([self.optimizer.get_actual_cost(y, d, st) for y, d, st in zip(y_true_np, decisions_true, start_times_true)])
                        spo_loss = torch.mean(torch.abs(torch.tensor(cost_pred - cost_true, device=self.device)))

                    loss = mse_loss + self.lambda_spo * spo_loss
                else:
                    loss = mse_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                batch_count += 1
                if batch_idx % 10 == 0:
                    logging.info(f"Epoch [{epoch + 1}/{epochs}] Batch [{batch_idx}] "
                                 f"Loss: {loss.item():.4f} (MSE: {mse_loss:.2f})")
            avg_loss = total_loss / batch_count
            decision_acc, avg_loss = auc(epoch, decision_acc, avg_loss)
            scheduler.step(avg_loss)
            train_losses.append(avg_loss)
            decision_accs.append(decision_acc)
            logging.info(f"Epoch [{epoch + 1}/{epochs}] Avg Loss: {avg_loss:.4f} "
                         f"Decision Acc: {decision_acc * 100:.1f}% "
                         f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        return train_losses, decision_accs

    def evaluate(self, test_loader, processor, min_val, max_val):
        self.predictor.eval()
        all_preds = []
        all_targets = []
        decision_diffs = []
        cost_gaps = []

        with torch.no_grad():
            for x, y_true in test_loader:
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                y_pred = self.predictor(x)

                y_pred_np = y_pred.cpu().numpy()  # shape: (batch, 12, 5)
                y_true_np = y_true.cpu().numpy()
                denorm_preds = processor.denormalize(y_pred_np, min_val, max_val)
                denorm_targets = processor.denormalize(y_true_np, min_val, max_val)

                for pred, true in zip(denorm_preds, denorm_targets):
                    # 每个预测样本都送入优化器
                    result_pred = self.optimizer.solve_multi_period(pred)
                    result_true = self.optimizer.solve_multi_period(true)

                    if result_pred["status"] == "optimal" and result_true["status"] == "optimal":
                        diff = np.abs(result_pred["renovation_decisions"] - result_true["renovation_decisions"])
                        decision_diffs.append(diff.sum())

                        cost_pred = result_pred["total_cost"]
                        cost_true = result_true["total_cost"]
                        cost_gaps.append(cost_pred - cost_true)
                    else:
                        print("[WARN] 有优化样本求解失败，已跳过")

                all_preds.append(denorm_preds)
                all_targets.append(denorm_targets)

        all_preds = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
        all_targets = np.concatenate(all_targets, axis=0) if all_targets else np.array([])

        # 防止空数组导致 nan
        decision_acc = 1 - np.mean(np.array(decision_diffs) > 0) if len(decision_diffs) > 0 else float('nan')
        cost_gap = np.mean(cost_gaps) if len(cost_gaps) > 0 else float('nan')

        print(f"[Eval] 样本总数: {len(all_preds)}")
        print(f"[Eval] 成功优化样本数: {len(cost_gaps)}")
        print(f"[Eval] 跳过样本数: {len(all_preds) - len(cost_gaps)}")

        return {
            'mae': np.mean(np.abs(all_preds - all_targets)) if all_preds.size > 0 else float('nan'),
            'rmse': np.sqrt(np.mean((all_preds - all_targets) ** 2)) if all_preds.size > 0 else float('nan'),
            'decision_acc': decision_acc,
            'cost_gap': cost_gap,
            'predicted_values': all_preds,
            'true_values': all_targets
        }


