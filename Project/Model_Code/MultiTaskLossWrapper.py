import torch
import torch.nn as nn

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, age_pred, gen_pred, age_true, gen_true):
        mse, binCrossEntropy = nn.MSELoss(), nn.BCELoss()

        loss0 = mse(age_pred, age_true)
        loss1 = binCrossEntropy(gen_pred, gen_true)

        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0*loss0 + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1*loss1 + self.log_vars[1]

        return loss0+loss1