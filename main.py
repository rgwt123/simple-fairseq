import argparse
from src.logger import create_logger
import os
import torch
logger = create_logger('train.log')

parser = argparse.ArgumentParser(description='Settings')
parser.add_argument("--train_data", type=str, default='data/cwmt.bin',
                    help="train data dir")
parser.add_argument("--max_len", type=int, default=100,
                    help="max length of sentences")
parser.add_argument("--reload_model", type=str, default='',
                    help="reload model")
parser.add_argument("--batch_size", type=int, default=80,
                    help="batch size sentences")
parser.add_argument("--batch_size_tokens", type=int, default=4000,
                    help="batch size tokens")
parser.add_argument("--src_n_words", type=int, default=0,
                    help="data")
parser.add_argument("--tgt_n_words", type=int, default=0,
                    help="data")
parser.add_argument("--dropout", type=float, default=0.1,
                    help="Dropout")
parser.add_argument("--label-smoothing", type=float, default=0.1,
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
parser.add_argument("--update_freq", type=int, default=1)
parser.add_argument("--optim", type=str, default="adam_inverse_sqrt,lr=0.0005")
parser.add_argument("--gpu_num", type=int, default=1)
parser.add_argument("--checkpoint_dir", type=str, default="all_models/base")
parser.add_argument("--seed", type=int, default=1244)
parser.add_argument("--max_epoch", type=int, default=5)
parser.add_argument("--save_freq_update", type=int, default=5000)
parser.add_argument("--save_optimizer", type=bool, default=False,
                    help="save optimizer parameters")
parser.add_argument("--do_eval", type=bool, default=True,
                    help="do evalution during training")
parser.add_argument("--model_name",type=str, default="")
parser.add_argument("--src_dico_file", type=str, default='')
parser.add_argument("--tgt_dico_file", type=str, default='')
parser.add_argument("--translate_file", type=str, default='')
parser.add_argument("--reference_file", type=str, default='')


if __name__ == '__main__':
    params = parser.parse_args()

    if params.gpu_num == 1:
        from single_train import main
        main(params)
    else:
        from multiprocessing_train import main
        logger.info('GPU numbers: %s',params.gpu_num)
        main(params)



