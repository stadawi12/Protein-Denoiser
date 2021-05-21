import math
import torch

def cross_entropy_element(y_pred, y_truth):
    term1 = y_truth * math.log10(y_pred)
    term2 = (1 - y_truth) * math.log(1 - y_pred)
    return term1 + term2
