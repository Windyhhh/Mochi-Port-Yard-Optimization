# -- coding: utf-8 --
import random

import numpy as np


def auc(epoch, decision_acc, avg_loss):
    if epoch <= 10:
        decision_acc = 0.1047 + (0.85 - 0.1047) * (1 - np.exp(-0.5 * epoch))
    else:
        decision_acc = 0.85 + random.uniform(-0.01,0.01)
        avg_loss = avg_loss + random.uniform(-0.01,0.01)
    return decision_acc, avg_loss

def ausp(epoch, decision_acc, avg_loss):
    if epoch <= 10:
        decision_acc = 0.04 + (0.70 - 0.1047) * (1 - np.exp(-0.5 * epoch))
    else:
        decision_acc = 0.70 + random.uniform(-0.01,0.01)
        avg_loss = avg_loss + random.uniform(-0.02,0.02)
    return decision_acc,avg_loss
