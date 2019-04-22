import os
from collections import namedtuple

LatentState = namedtuple('LatentState', 'dec_input, input_len')


def build_mt_model(params, cuda=True):
    if params.attention:
        from .attention import build_attention_model
        return build_attention_model(params, cuda=cuda)
    else:
        return None
