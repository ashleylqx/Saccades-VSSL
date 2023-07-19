import torch
from torch import nn


eps = 1e-7

class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss() # combines log_softmax and nll_loss

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze() # there are bsz classification predictions
        label = torch.zeros([bsz]).cuda().long() # for each prediction, the gt class label is 0
        loss = self.criterion(x, label)
        return loss


