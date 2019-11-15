from logging import getLogger
import math
import numpy as np
import torch


logger = getLogger()


class Dataset(object):
    def __init__(self, params):
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.unk_index = params.unk_index
        self.bos_index = params.bos_index
        self.batch_size = params.batch_size
        self.batch_size_tokens = params.batch_size_tokens
        self.gpu_num = params.gpu_num
        self.seed = params.seed

    def batch_sentences(self, sentences):
        """
        Take as input a list of n sentences (torch.LongTensor vectors) and return
        a tensor of size (s_len, n) where s_len is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        lengths = torch.LongTensor([len(s) + 2 for s in sentences])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.pad_index)

        sent[0] = self.bos_index
        for i, s in enumerate(sentences):
            sent[1:lengths[i] - 1, i].copy_(s)
            sent[lengths[i] - 1, i] = self.eos_index

        return sent, lengths


class MonolingualDataset(Dataset):
    def __init__(self, sent1, pos1, dico1, params):
        super(MonolingualDataset, self).__init__(params)
        self.sent1 = sent1
        self.pos1 = pos1
        self.dico1 = dico1
        self.lengths1 = (self.pos1[:, 1] - self.pos1[:, 0])
        self.is_parallel = False

        # check number of sentences
        assert len(self.pos1) == (self.sent1 == -1).sum()

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return len(self.pos1)

    def get_batches_iterator(self, batches):
        """
        Return a sentences iterator, given the associated sentence batches.
        """
        def iterator():
            for sentence_ids in batches:
                pos1 = self.pos1[sentence_ids]
                sent1 = [self.sent1[a:b] for a, b in pos1]
                yield self.batch_sentences(sent1)
        return iterator

    def get_iterator(self, shuffle=False, group_by_size=False, n_sentences=-1):
        """
        Return a sentences iterator.
        """
        np.random.seed(self.seed)

        n_sentences = len(self.pos1) if n_sentences == -1 else n_sentences
        assert 0 < n_sentences <= len(self.pos1)
        assert type(shuffle) is bool and type(group_by_size) is bool

        # select sentences to iterate over
        if shuffle:
            indices = np.random.permutation(len(self.pos1))[:n_sentences]
        else:
            indices = np.arange(n_sentences)

        # group sentences by lengths
        if group_by_size:
            indices = indices[np.argsort(self.lengths1[indices], kind='mergesort')]

        # create batches / optionally shuffle them
        batches = np.array_split(indices, math.ceil(len(indices) * 1. / self.batch_size))
        if shuffle:
            np.random.shuffle(batches)

        # return the iterator
        return self.get_batches_iterator(batches)


class ParallelDataset(Dataset):

    def __init__(self, sent1, pos1, dico1, sent2, pos2, dico2, params):
        super(ParallelDataset, self).__init__(params)
        self.sent1 = sent1
        self.sent2 = sent2
        self.pos1 = pos1
        self.pos2 = pos2
        self.dico1 = dico1
        self.dico2 = dico2
        self.lengths1 = (self.pos1[:, 1] - self.pos1[:, 0])
        self.lengths2 = (self.pos2[:, 1] - self.pos2[:, 0])
        self.is_parallel = True
        self.total = 0

        # check number of sentences
        assert len(self.pos1) == (self.sent1 == -1).sum()
        assert len(self.pos2) == (self.sent2 == -1).sum()

        self.remove_empty_sentences()

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return len(self.pos1)

    def remove_empty_sentences(self):
        """
        Remove empty sentences.
        """
        init_size = len(self.pos1)
        indices = np.arange(len(self.pos1))
        indices = indices[self.lengths1[indices] > 0]
        indices = indices[self.lengths2[indices] > 0]
        self.pos1 = self.pos1[indices]
        self.pos2 = self.pos2[indices]
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        logger.info("Removed %i empty sentences." % (init_size - len(indices)))

    def remove_long_sentences(self, max_len):
        """
        Remove sentences exceeding a certain length.
        """
        assert max_len > 0
        init_size = len(self.pos1)
        indices = np.arange(len(self.pos1))
        indices = indices[self.lengths1[indices] <= max_len]  # indices[True,False.....] = [0,1,2....]
        indices = indices[self.lengths2[indices] <= max_len]
        self.pos1 = self.pos1[indices]
        self.pos2 = self.pos2[indices]
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        logger.info("Removed %i too long sentences." % (init_size - len(indices)))

    def select_data(self, a, b):
        """
        Only retain a subset of the dataset.
        """
        assert 0 <= a <= b <= len(self.pos1)
        if a < b:
            self.pos1 = self.pos1[a:b]
            self.pos2 = self.pos2[a:b]
            self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
            self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        else:
            self.pos1 = torch.LongTensor()
            self.pos2 = torch.LongTensor()
            self.lengths1 = torch.LongTensor()
            self.lengths2 = torch.LongTensor()


    def get_batches_iterator(self, batches):
        """
        Return a sentences iterator, given the associated sentence batches.
        """
        def iterator():
            for sentence_ids in batches:
                pos1 = self.pos1[sentence_ids]
                pos2 = self.pos2[sentence_ids]
                sent1 = [self.sent1[a:b] for a, b in pos1]
                sent2 = [self.sent2[a:b] for a, b in pos2]
                yield self.batch_sentences(sent1), self.batch_sentences(sent2)
        return iterator

    def get_iterator(self, shuffle, group_by_size=False, n_sentences=-1, partition=None):
        """
        Return a sentences iterator.
        """
        np.random.seed(self.seed)
        self.seed += 1
        # 可能会影响数据的随机性
        # 多gpu可能需要这个 不然每个进程里的数据不一样
        
        n_sentences = len(self.pos1) if n_sentences == -1 else n_sentences
        assert 0 < n_sentences <= len(self.pos1)
        assert type(shuffle) is bool and type(group_by_size) is bool

        # select sentences to iterate over
        if shuffle:
            indices = np.random.permutation(len(self.pos1))[:n_sentences]
        else:
            indices = np.arange(n_sentences)

        # group sentences by lengths
        if group_by_size:
            indices = indices[np.argsort(200-self.lengths2[indices], kind='mergesort')]
            indices = indices[np.argsort(200-self.lengths1[indices], kind='mergesort')]
        # 这里得到了所有句子的id，按照句长从小到大排列的句子id
        # just same with ordered indices in fairseq's language_pair_dataset.py
        # change to 200-length for reverse because of padding in lstm

        # create batches / optionally shuffle them
        if self.batch_size_tokens == -1:
            batches = np.array_split(indices, math.ceil(len(indices) * 1. / self.batch_size))
        else:
            # need to split sentences to batch depending on max_batch_size_tokens
            max_tokens = self.batch_size_tokens
            batches = []
            batch = []

            def is_batch_full(num_tokens):
                if len(batch) == 0:
                    return False
                if num_tokens > max_tokens:
                    return True
                return False

            sample_len = 0
            sample_lens = []                
            id = 0
            while id < len(indices):
                idx = indices[id]
                sample_lens.append(max(self.lengths1[idx], self.lengths2[idx]))
                history = sample_len
                sample_len = max(sample_len, sample_lens[-1])
                num_tokens = len(batch) * sample_len
                if is_batch_full(num_tokens):
                    # prevent a sudden increase of num_tokens (Ex. 30*50=1500 -> 31*100=3100)
                    batch.pop()
                    id -= 1
                    batches.append(np.array(batch))
                    batch = []
                    sample_lens = []
                    sample_len = 0
                batch.append(idx)
                id += 1
                    
                
            if len(batch) > 0:
                batches.append(np.array(batch))
        batches = np.array(batches)

        if shuffle:
            np.random.shuffle(batches)

        self.total = len(batches)
        # partition
        if partition is not None:
            part_len = int((1.0/self.gpu_num)*self.total)
            batches = batches[part_len*partition:(partition+1)*part_len]

        self.total = len(batches)
        # return the iterator
        return self.get_batches_iterator(batches)
