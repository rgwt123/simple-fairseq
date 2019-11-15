from logging import getLogger
import os
import torch
from torch import nn
from ..modules.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyLoss

logger = getLogger()


def build_attention_model(params, cuda=True):
    if params.transformer:
        encoder, decoder = build_transformer_enc_dec(params)
    elif params.lstm:
        encoder, decoder = build_lstm_enc_dec(params)
    else:
        logger.error('waiting for write ...')
        exit(0)

    loss_weight = torch.FloatTensor(params.tgt_n_words).fill_(1)
    loss_weight[params.pad_index] = 0
    if params.label_smoothing <= 0:
        loss_fn = nn.CrossEntropyLoss(loss_weight, reduction="elementwise_mean")
    else:
        loss_fn = LabelSmoothedCrossEntropyLoss(
            params.label_smoothing,
            params.pad_index,
            size_average=False,
            weight=loss_weight
        )

    decoder.loss_fn = loss_fn

    if cuda:
        encoder.cuda()
        decoder.cuda()

    if params.reload_model != '':
        assert os.path.isfile(params.reload_model)
        logger.info('Reloading model from %s ...'% params.reload_model)
        if cuda:
            reloaded = torch.load(params.reload_model)
        else:
            reloaded = torch.load(params.reload_model, 'cpu')
        logger.info('Reloading encoder...')
        encoder.load_state_dict(reloaded['encoder'])
        logger.info('Reloading decoder...')
        decoder.load_state_dict(reloaded['decoder'])
        params.now_epoch = reloaded['epoch'] + 1
        # fix for previous version
        num_updates = reloaded.get('num_updates', 0)
    else:
        num_updates = 0

    logger.info('num_updates: %i ' % num_updates)
    return encoder, decoder, num_updates


def build_transformer_enc_dec(params):
    from .transformer import TransformerEncoder, TransformerDecoder

    params.left_pad_source = False
    params.left_pad_target = False

    params.encoder_embed_dim = params.emb_dim
    params.encoder_ffn_embed_dim = params.transformer_ffn_emb_dim
    params.encoder_layers = params.n_enc_layers

    params.decoder_embed_dim = params.emb_dim
    params.decoder_ffn_embed_dim = params.transformer_ffn_emb_dim
    params.decoder_layers = params.n_dec_layers

    logger.info("============ Building transformer attention model - Encoder ...")
    encoder = TransformerEncoder(params)
    logger.info("")
    logger.info("============ Building transformer attention model - Decoder ...")
    decoder = TransformerDecoder(params, encoder)
    logger.info("")
    return encoder, decoder

def build_lstm_enc_dec(params):
    from .lstm import LSTMEncoder, LSTMDecoder

    params.left_pad_source = False
    params.left_pad_target = False

    params.encoder_embed_dim = params.emb_dim
    params.encoder_layers = params.n_enc_layers

    params.decoder_embed_dim = params.emb_dim
    params.decoder_layers = params.n_dec_layers

    logger.info("============ Building LSTM attention model - Encoder ...")
    encoder = LSTMEncoder(params)
    logger.info("")
    logger.info("============ Building LSTM attention model - Decoder ...")
    decoder = LSTMDecoder(params, encoder)
    logger.info("")
    return encoder, decoder