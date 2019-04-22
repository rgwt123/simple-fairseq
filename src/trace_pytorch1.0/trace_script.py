import torch

import argparse
import torch
from src.logger import create_logger
import os
from src.model import build_mt_model
from src.data.loader import load_data
import subprocess
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser(description='Settings')
parser.add_argument("--train_data", type=str, default='huawei/120w.bin',
                    help="train data dir")
parser.add_argument("--max_len", type=int, default=100,
                    help="max length of sentences")
parser.add_argument("--reload_model", type=str, default='',
                    help="reload model")
parser.add_argument("--batch_size", type=int, default=128,
                    help="batch size")
parser.add_argument("--batch_size_tokens", type=int, default=-1,
                    help="batch size tokens")
parser.add_argument("--src_n_words", type=int, default=0,
                    help="data")
parser.add_argument("--tgt_n_words", type=int, default=0,
                    help="data")
parser.add_argument("--dropout", type=float, default=0,
                    help="Dropout")
parser.add_argument("--label-smoothing", type=float, default=0,
                    help="Label smoothing")
parser.add_argument("--attention", type=bool, default=True,
                    help="Use an attention mechanism")
parser.add_argument("--transformer", type=bool, default=True,
                    help="Use Transformer")
parser.add_argument("--emb_dim", type=int, default=512,
                    help="Embedding layer size")
parser.add_argument("--n_enc_layers", type=int, default=6,
                    help="Number of layers in the encoders")
parser.add_argument("--n_dec_layers", type=int, default=6,
                    help="Number of layers in the decoders")
parser.add_argument("--hidden_dim", type=int, default=512,
                    help="Hidden layer size")

parser.add_argument("--transformer_ffn_emb_dim", type=int, default=2048,
                    help="Transformer fully-connected hidden dim size")
parser.add_argument("--attention_dropout", type=float, default=0,
                    help="attention_dropout")
parser.add_argument("--relu_dropout", type=float, default=0,
                    help="relu_dropout")
parser.add_argument("--encoder_attention_heads", type=int, default=8,
                    help="encoder_attention_heads")
parser.add_argument("--decoder_attention_heads", type=int, default=8,
                    help="decoder_attention_heads")
parser.add_argument("--encoder_normalize_before", type=bool, default=False,
                    help="encoder_normalize_before")
parser.add_argument("--decoder_normalize_before", type=bool, default=False,
                    help="decoder_normalize_before")
parser.add_argument("--share_encdec_emb", type=bool, default=False,
                    help="share encoder and decoder embedding")
parser.add_argument("--share_decpro_emb", type=bool, default=True,
                    help="share decoder input and project embedding")
parser.add_argument("--beam_size", type=int, default=5,
                    help="beam search size")
parser.add_argument("--length_penalty", type=float, default=1.0,
                    help="length penalty")
parser.add_argument("--clip_grad_norm", type=float, default=5.0,
                    help="clip grad norm")
parser.add_argument("--id",type=int, default=0)
parser.add_argument("--checkpoint_dir", type=str)
params = parser.parse_args()
params.checkpoint_dir = 'huawei_delay4'
params.id = 29
params.reload_model = '{}/model_epoch{}.pt'.format(params.checkpoint_dir, params.id)
params.translate_file = 'huawei/valid.bpe.zh'
params.src_dico_file = 'huawei/dict.bpe.zh'
params.tgt_dico_file = 'huawei/dict.bpe.en'

if __name__ == '__main__':
    data = load_data(params, name='test')
    encoder, decoder, _ = build_mt_model(params, cuda=False)
    encoder.eval()
    decoder.eval()

    test_seq = torch.LongTensor(10 , 1).random_(0, params.src_n_words)
    test_seq_length = torch.LongTensor([test_seq.size()[0]])

    print(test_seq, test_seq_length)
    # Trace the model
    traced_encoder = torch.jit.trace(encoder, (test_seq, test_seq_length))
    # print(traced_encoder.graph)
    print('trace finish')
    traced_encoder.save('huawei_delay4/traced_encoder.pth')
    print('save finish')


