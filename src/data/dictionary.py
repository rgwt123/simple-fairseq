import os
import torch
from logging import getLogger

logger = getLogger()

BOS_WORD = '<s>'
EOS_WORD = '</s>'
PAD_WORD = '<pad>'
UNK_WORD = '<unk>'


class Dictionary(object):
    def __init__(self, id2word, word2id):
        assert len(id2word) == len(word2id)
        self.id2word = id2word
        self.word2id = word2id
        self.bos_index = word2id[BOS_WORD]
        self.eos_index = word2id[EOS_WORD]
        self.pad_index = word2id[PAD_WORD]
        self.unk_index = word2id[UNK_WORD]
        self.check_valid()

    def __len__(self):
        """Returns the number of words in the dictionary"""
        return len(self.id2word)

    def __getitem__(self, i):
        """
        Returns the word of the specified index.
        """
        return self.id2word[i]

    def __contains__(self, w):
        """
        Returns whether a word is in the dictionary.
        """
        return w in self.word2id

    def __eq__(self, y):
        """
        Compare this dictionary with another one.
        """
        self.check_valid()
        y.check_valid()
        if len(self.id2word) != len(y):
            return False
        return all(self.id2word[i] == y[i] for i in range(len(y)))

    def check_valid(self):
        """
        Check that the dictionary is valid.
        """
        assert self.bos_index == 0
        assert self.eos_index == 1
        assert self.pad_index == 2
        assert self.unk_index == 3
        assert len(self.id2word) == len(self.word2id)
        for i in range(len(self.id2word)):
            assert self.word2id[self.id2word[i]] == i

    def idx2string(self, indexes):
        str = []
        for i in indexes:
            i = i.item()
            if i == self.bos_index:
                continue
            if i == self.eos_index:
                break
            str.append(self.id2word[i])
        return ' '.join(str)

    def str2idx(self, str):
        return torch.Tensor([self.index[w] for w in str.split()])

    def index(self, word, no_unk=False):
        """
        Returns the index of the specified word.
        """
        if no_unk:
            return self.word2id[word]
        else:
            return self.word2id.get(word, self.unk_index)

    def prune(self, max_vocab):
        """
        Limit the vocabulary size.
        """
        assert max_vocab >= 1
        self.id2word = {k: v for k, v in self.id2word.items() if k < max_vocab}
        self.word2id = {v: k for k, v in self.id2word.items()}
        self.check_valid()

    @staticmethod
    def read_vocab(vocab_path):
        """
        Create a dictionary from a vocabulary file.
        """
        skipped = 0
        assert os.path.isfile(vocab_path), vocab_path
        word2id = {BOS_WORD: 0, EOS_WORD: 1, PAD_WORD: 2, UNK_WORD: 3}
        f = open(vocab_path, 'r', encoding='utf-8')
        for i, line in enumerate(f):
            if '\u2028' in line:
                skipped += 1
                continue
            line = line.rstrip().split()
            assert len(line) == 2, (i, line)
            assert line[0] not in word2id and line[1].isdigit(), (i, line)
            word2id[line[0]] = 4 + i - skipped  # shift because of extra words
        f.close()
        id2word = {v: k for k, v in word2id.items()}
        dico = Dictionary(id2word, word2id)
        logger.info("Read %i words from the vocabulary file." % len(dico))
        if skipped > 0:
            logger.warning("Skipped %i empty lines!" % skipped)
        return dico

    @staticmethod
    def index_data(src_txt_path, tgt_txt_path, src_dico, tgt_dico, bin_path):
        """
        Index sentences with a dictionary.
        """
        if os.path.isfile(bin_path):
            print("Exsited file %s ..." % bin_path)
            return None

        positions_s = []
        sentences_s = []
        unk_words_s = {}

        positions_t = []
        sentences_t = []
        unk_words_t = {}

        # index sentences
        fs = open(src_txt_path, 'r', encoding='utf-8')
        ft = open(tgt_txt_path, 'r', encoding='utf-8')
        for i, line in enumerate(fs):
            line_t = ft.readline()
            if i % 1000000 == 0 and i > 0:
                print(i)
            s = line.rstrip().split()
            s_t = line_t.rstrip().split()
            # skip empty sentences
            if (len(s) == 0) or (len(s_t) == 0):
                print("Empty sentence in line %i." % i)
                # continue
            # index sentence words
            count_unk_s = 0
            count_unk_t = 0
            indexed_s = []
            indexed_t = []
            for w in s:
                word_id = src_dico.index(w, no_unk=False)
                if word_id < 4 and word_id != src_dico.unk_index:
                    logger.warning('Found unexpected special word "%s" (%i)!!' % (w, word_id))
                    continue
                indexed_s.append(word_id)
                if word_id == src_dico.unk_index:
                    unk_words_s[w] = unk_words_s.get(w, 0) + 1
                    count_unk_s += 1
            for w in s_t:
                word_id = tgt_dico.index(w, no_unk=False)
                if word_id < 4 and word_id != tgt_dico.unk_index:
                    logger.warning('Found unexpected special word "%s" (%i)!!' % (w, word_id))
                    continue
                indexed_t.append(word_id)
                if word_id == tgt_dico.unk_index:
                    unk_words_t[w] = unk_words_t.get(w, 0) + 1
                    count_unk_t += 1
            # add sentence
            positions_s.append([len(sentences_s), len(sentences_s) + len(indexed_s)])
            sentences_s.extend(indexed_s)
            sentences_s.append(-1)

            positions_t.append([len(sentences_t), len(sentences_t) + len(indexed_t)])
            sentences_t.extend(indexed_t)
            sentences_t.append(-1)

        fs.close()
        ft.close()

        # tensorize data
        positions_s = torch.LongTensor(positions_s)
        sentences_s = torch.LongTensor(sentences_s)
        positions_t = torch.LongTensor(positions_t)
        sentences_t = torch.LongTensor(sentences_t)

        data = {
            'src_dico': src_dico,
            'tgt_dico':tgt_dico,
            'src_positions': positions_s,
            'tgt_positions':positions_t,
            'src_sentences': sentences_s,
            'tgt_sentences':sentences_t,
            'src_unk_words': unk_words_s,
            'tgt_unk_words':unk_words_t
        }
        print("Saving the data to %s ..." % bin_path)
        torch.save(data, bin_path)

        return data
