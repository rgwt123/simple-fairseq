import os
import sys

from src.logger import create_logger
from src.data.dictionary import Dictionary

if __name__ == '__main__':
    logger = create_logger(None)

    src_voc_path = 'data_v2_20190408/dict.reptag.bpe.cn2en'
    src_txt_path = 'data_v2_20190408/train.reptag.bpe.cn'
    tgt_voc_path = 'data_v2_20190408/dict.reptag.bpe.cn2en'
    tgt_txt_path = 'data_v2_20190408/train.reptag.bpe.en'

    bin_path = 'data/reptag.bin'
    assert os.path.isfile(src_voc_path)
    assert os.path.isfile(src_txt_path)
    assert os.path.isfile(tgt_voc_path)
    assert os.path.isfile(tgt_txt_path)

    src_dico = Dictionary.read_vocab(src_voc_path)
    tgt_dico = Dictionary.read_vocab(tgt_voc_path)

    data = Dictionary.index_data(src_txt_path, tgt_txt_path, src_dico, tgt_dico, bin_path)
    if data is None:
        exit(0)
    logger.info("%i words (%i unique) in %i sentences." % (
        len(data['src_sentences']) - len(data['src_positions']),
        len(data['src_dico']),
        len(data['src_positions'])
    ))
    logger.info("%i words (%i unique) in %i sentences." % (
        len(data['tgt_sentences']) - len(data['tgt_positions']),
        len(data['tgt_dico']),
        len(data['tgt_positions'])
    ))
    if len(data['src_unk_words']) > 0:
        logger.info("%i unknown words (%i unique), covering %.2f%% of the data." % (
            sum(data['src_unk_words'].values()),
            len(data['src_unk_words']),
            sum(data['src_unk_words'].values()) * 100. / (len(data['src_sentences']) - len(data['src_positions']))
        ))
        if len(data['src_unk_words']) < 30:
            for w, c in sorted(data['src_unk_words'].items(), key=lambda x: x[1])[::-1]:
                logger.info("%s: %i" % (w, c))
    else:
        logger.info("0 unknown word.")

    if len(data['tgt_unk_words']) > 0:
        logger.info("%i unknown words (%i unique), covering %.2f%% of the data." % (
            sum(data['tgt_unk_words'].values()),
            len(data['tgt_unk_words']),
            sum(data['tgt_unk_words'].values()) * 100. / (len(data['tgt_sentences']) - len(data['tgt_positions']))
        ))
        if len(data['tgt_unk_words']) < 30:
            for w, c in sorted(data['tgt_unk_words'].items(), key=lambda x: x[1])[::-1]:
                logger.info("%s: %i" % (w, c))
    else:
        logger.info("0 unknown word.")
