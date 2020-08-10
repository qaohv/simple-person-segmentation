import torch
import torch.nn as nn


class BinaryFocalLoss2d(nn.Module):
    def __init__(self, gamma=2, size_average=True):
        super().__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logit, target, class_weight=None):
        target = target.contiguous().view(-1, 1).long()

        if class_weight is None:
            class_weight = [1] * 2

        prob = torch.sigmoid(logit)
        prob = prob.contiguous().view(-1, 1)
        prob = torch.cat((1 - prob, prob), 1)
        select = torch.FloatTensor(len(prob), 2).zero_().cuda()
        select.scatter_(1, target, 1.)

        class_weight = torch.FloatTensor(class_weight).cuda().view(-1, 1)
        class_weight = torch.gather(class_weight, 0, target)

        prob = (prob * select).sum(1).view(-1, 1)
        prob = torch.clamp(prob, 1e-8, 1 - 1e-8)
        batch_loss = - class_weight * (torch.pow((1 - prob), self.gamma)) * prob.log()

        return batch_loss.mean() if self.size_average else batch_loss
