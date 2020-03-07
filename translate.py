import argparse
import torch
from src.logger import create_logger
import os
from src.model import build_mt_model
from src.data.loader import load_data
import subprocess
import re

logger = create_logger('translate.log')

parser = argparse.ArgumentParser(description='Settings')
parser.add_argument("--train_data", type=str, default='',
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
parser.add_argument("--lstm", type=bool, default=False,
                    help="Use LSTM")
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
parser.add_argument("--model_name",type=str, default="")
parser.add_argument("--checkpoint_dir", type=str, default='')
parser.add_argument("--src_dico_file", type=str, default='')
parser.add_argument("--tgt_dico_file", type=str, default='')
parser.add_argument("--translate_file", type=str, default='')
parser.add_argument("--gpu_num", type=int, default=1)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--reference_file", type=str, default='')
params = parser.parse_args()
params.reload_model = '{}/{}'.format(params.checkpoint_dir, params.model_name)
params.out_file = '{}/predict_{}'.format(params.checkpoint_dir, params.model_name[:-3])
if __name__ == '__main__':
    data = load_data(params, name='test')
    encoder, decoder, _ = build_mt_model(params)
    encoder.eval()
    decoder.eval()
    iterator = data.get_iterator(shuffle=False, group_by_size=False)()
    file = open(params.out_file, 'w',encoding='utf-8')
    total = 0
    with torch.no_grad():
        for (sen1, len1) in iterator:
            len1, bak_order = len1.sort(descending=True)
            sen1 = sen1[:,bak_order]
            sen1 = sen1.cuda()
            encoded = encoder(sen1, len1)
            sent2, len2, _ = decoder.generate(encoded)
            total += len2.size(0)
            logger.info('Translating %i sentences.' % total)
            for j in bak_order.argsort().tolist():
                file.write(params.tgt_dico.idx2string(sent2[:, j]).replace('@@ ', '')+'\n')
    file.close()
    # calculate bleu value
    '''
    command = f'perl scripts/multi-bleu.perl {params.reference_file} < {params.out_file}'
    print(command)
    p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    result = p.communicate()[0].decode("utf-8")
    bleu = re.findall(r"BLEU = (.+?),", result)[0]
    logger.info(result)
    logger.info(bleu)
    file.close()
    with open('{}/bleu.log'.format(params.checkpoint_dir),'a+') as f:
        f.write(str(params.model_name)+' '+result)
    '''
