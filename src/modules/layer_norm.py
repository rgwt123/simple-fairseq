import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Applies Layer Normalization over the last dimension."""

    def __init__(self, features, eps=1e-5):
        super().__init__()
        self.features = features
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.dummy = None
        self.w = None
        self.b = None

    def forward(self, input):
        shape = input.size()

        # In order to force the cudnn path, everything needs to be
        # contiguous. Hence the check here and reallocation below.
        if not input.is_contiguous():
            input = input.contiguous()
        input = input.view(1, -1, shape[-1])

        # Expand w and b buffers if necessary.
        n = input.size(1)
        cur = self.dummy.numel() if self.dummy is not None else 0
        if cur == 0:
            self.dummy = input.data.new(n)
            self.w = input.data.new(n).fill_(1)
            self.b = input.data.new(n).zero_()
        elif n > cur:
            self.dummy.resize_(n)
            self.w.resize_(n)
            self.w[cur:n].fill_(1)
            self.b.resize_(n)
            self.b[cur:n].zero_()
        dummy = self.dummy[:n]
        w = self.w[:n]
        b = self.b[:n]
        output = F.batch_norm(input, dummy, dummy, w, b, True, 0., self.eps)
        return torch.addcmul(self.bias, 1, output.view(*shape), self.gain)