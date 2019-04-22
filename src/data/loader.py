from logging import getLogger
import torch
import torch.distributed as dist
from .dataset import ParallelDataset, MonolingualDataset
from .dictionary import EOS_WORD, PAD_WORD, UNK_WORD, BOS_WORD, Dictionary
logger = getLogger()


def load_data(params, name='train'):
    if name == 'train':
        data = torch.load(params.train_data)
        src_dico = data['src_dico']
        tgt_dico = data['tgt_dico']
    elif name == 'test':
        src_dico = Dictionary.read_vocab(params.src_dico_file)
        tgt_dico = Dictionary.read_vocab(params.tgt_dico_file)

    eos_index = src_dico.index(EOS_WORD) # 1
    pad_index = src_dico.index(PAD_WORD) # 2
    unk_index = src_dico.index(UNK_WORD) # 3
    bos_index = src_dico.index(BOS_WORD) # 0

    params.eos_index =  eos_index
    params.bos_index = bos_index
    params.pad_index = pad_index
    params.unk_index = unk_index

    params.src_dico = src_dico
    params.tgt_dico = tgt_dico

    params.src_n_words = len(src_dico)
    params.tgt_n_words = len(tgt_dico)

    if name == 'train':
        para_data = ParallelDataset(data['src_sentences'], data['src_positions'].numpy(), data['src_dico'],
                                data['tgt_sentences'], data['tgt_positions'].numpy(), data['tgt_dico'], params)
        para_data.remove_long_sentences(params.max_len)

        logger.info('============ Data summary')
        logger.info('Dictionary, %i and %i words.' % (len(params.src_dico), len(params.tgt_dico)))
        logger.info('%i and %i parallel sentences.' % (len(para_data.lengths1), len(para_data.lengths2)))

        return para_data

    elif name == 'test':
        sent, pos = index_file(params)
        mono_data = MonolingualDataset(sent, pos.numpy(), params.src_dico, params)
        return mono_data

    else:
        logger.error('Not implemented!')
        exit(0)


def index_file(params):
    positions_s = []
    sentences_s = []
    unk_words_s = {}

    fs = open(params.translate_file, 'r', encoding='utf-8')
    for i, line in enumerate(fs):
        s = line.rstrip().split()
        count_unk_s = 0
        indexed_s = []
        for w in s:
            word_id = params.src_dico.index(w, no_unk=False)
            if word_id < 4 and word_id != params.src_dico.unk_index:
                logger.warning('Found unexpected special word "%s" (%i)!!' % (w, word_id))
                continue
            indexed_s.append(word_id)
            if word_id == params.src_dico.unk_index:
                unk_words_s[w] = unk_words_s.get(w, 0) + 1
                count_unk_s += 1
        # add sentence
        positions_s.append([len(sentences_s), len(sentences_s) + len(indexed_s)])
        sentences_s.extend(indexed_s)
        sentences_s.append(-1)
    fs.close()
    positions_s = torch.LongTensor(positions_s)
    sentences_s = torch.LongTensor(sentences_s)

    return sentences_s, positions_s