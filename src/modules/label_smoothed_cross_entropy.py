from torch import nn
import torch.nn.functional as F


class LabelSmoothedCrossEntropyLoss(nn.Module):

    def __init__(self, eps, padding_idx=None, size_average=False, weight=None):
        super().__init__()
        self.eps = eps
        self.padding_idx = padding_idx
        self.size_average = size_average
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        # lprobs,input -> [batch_size_tokens,target_vocab_size]
        lprobs = F.log_softmax(input, dim=-1)
        target = target.view(-1, 1)
        
        # nll_loss get [batch_sentence*seqlength(~=batch_size_tokens), 1]
        # nll means no label smooth loss
        nll_loss = -lprobs.gather(dim=-1, index=target)
        # smooth loss calculates the sum of non-target loss
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        if self.padding_idx is not None:
            # non_pad_mask -> [batch_size_tokens,1]
            non_pad_mask = target.ne(self.padding_idx)
            # ignore pad word loss
            nll_loss = nll_loss[non_pad_mask]
            sample_size = nll_loss.size(0)
            smooth_loss = smooth_loss[non_pad_mask]

        if self.size_average:
            nll_loss = nll_loss.mean()
            smooth_loss = smooth_loss.mean()
        else:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()

        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss

        return loss,sample_size

