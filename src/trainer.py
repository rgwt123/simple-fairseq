import re
import subprocess
from itertools import chain
from logging import getLogger

import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from src.utils import get_optimizer
from src.distributed_utils import all_gather_list,all_reduce_and_rescale_tensors


logger = getLogger()


class TrainerMT():
    def __init__(self, encoder, decoder, data, test_data, params, num_updates):
        self.encoder = encoder
        self.decoder = decoder
        self.data = data
        self.test_data = test_data
        self.params = params

        self.enc_dec_params = list(self.encoder.parameters())+list(self.decoder.parameters())
        # optimizers
        self.optimizer = get_optimizer(self.enc_dec_params, self.params.optim)
        self.optimizer._num_updates = num_updates
        # training statistics
        self.epoch = getattr(params, 'now_epoch', 0)
        self.n_iter = 0
        self.oom = 0
        self.n_sentences = 0
        self.stats = {
            'processed_s': 0,
            'processed_w': 0,
            'loss': []
        }
        self.sample_sizes = []

    def train_epoch(self):
        self.iterator = self.get_iterator()
        for (sent1, len1), (sent2, len2) in tqdm(self.iterator, mininterval=2, desc='  - (Training)   ', leave=False, total=self.data.total):
            self.train_step(sent1, len1, sent2, len2)
            # save only when 1. the main process 2. once in update_freq 3. every save_freq_update updates
            if ((self.n_iter+1) % self.params.update_freq == 0) and (self.params.rank == 0 or self.params.gpu_num == 1) and (self.optimizer._num_updates != 0 and self.optimizer._num_updates % self.params.save_freq_update == 0):
                checkpoint = {
                'encoder': self.encoder.state_dict(),
                'decoder': self.decoder.state_dict(),
                'params': self.params,
                'epoch': self.epoch,
                'num_updates': self.optimizer._num_updates
                }
                if self.params.save_optimizer:
                    checkpoint['optimizer'] = self.optimizer.state_dict()

                self.params.model_name = f'model_epoch{self.epoch}_update{self.optimizer._num_updates}.pt'
                torch.save(checkpoint, self.params.checkpoint_dir + '/' + self.params.model_name)

                # do evaluation
                if self.params.do_eval:
                    self.evaluate()
                    self.encoder.train()
                    self.decoder.train()

        # save epoch checkpoint
        if self.params.gpu_num == 1 or self.params.rank == 0:
            checkpoint = {
                'encoder': self.encoder.state_dict(),
                'decoder': self.decoder.state_dict(),
                'params': self.params,
                'epoch': self.epoch,
                'num_updates': self.optimizer._num_updates
            }
            if (self.epoch == self.params.max_epoch-1) or self.params.save_optimizer:
                checkpoint['optimizer'] = self.optimizer.state_dict()
            self.params.model_name = f'model_epoch{self.epoch}.pt'
            torch.save(checkpoint, self.params.checkpoint_dir + '/' + self.params.model_name)
            # do evaluation
            if self.params.do_eval:
                self.evaluate()
                self.encoder.train()
                self.decoder.train()
        self.epoch += 1

    def train_step(self, sent1, len1, sent2, len2):
        if self.params.update_freq == 1:
            need_zero = True
            need_reduction = True
        else:
            need_reduction = True if (self.n_iter+1) % self.params.update_freq == 0 else False
            need_zero = True if self.n_iter % self.params.update_freq == 0 else False
        self.encoder.train()
        self.decoder.train()
        sent1, sent2 = sent1.cuda(), sent2.cuda()
        try:
            if need_zero:
                self.optimizer.zero_grad()
            encoded = self.encoder(sent1, len1)
            scores = self.decoder(encoded, sent2[:-1])
            loss,sample_size = self.decoder.loss_fn(scores.view(-1, self.decoder.n_words), sent2[1:].view(-1))

            # check NaN
            if (loss != loss).data.any():
                logger.error("NaN detected")
                exit()
            # optimizer            
            loss.backward()
            self.sample_sizes.append(sample_size)
            # print(f'forward gpu-{self.params.rank},iter-{self.n_iter}{self.enc_dec_params[0].grad.data[0][0:20]}')
            
        except Exception as e:
            logger.error(e)
            torch.cuda.empty_cache()
            self.n_iter += 1
            self.oom += 1
            return

        if need_reduction:
            try:
                # sample_sizes contain gpu_num*update_delay numbers of tokens like [1948, 2013, ..]
                # now we get the total token nums of all delay batch and gpus
                if self.params.gpu_num > 1:
                    sample_sizes = all_gather_list(self.sample_sizes)
                    sample_sizes = list(chain.from_iterable(sample_sizes))
                    sample_size = sum(sample_sizes)
                    grads = [p.grad.data for p in self.enc_dec_params if p.requires_grad and p.grad is not None]
                    all_reduce_and_rescale_tensors(grads,float(sample_size)/self.params.gpu_num)
                else:
                    sample_size = sum(self.sample_sizes)
                    for p in self.enc_dec_params:
                        if p.requires_grad and p.grad is not None:
                            p.grad.data.mul_(1/float(sample_size))
                clip_grad_norm_(self.enc_dec_params, self.params.clip_grad_norm)
                self.optimizer.step()
                self.sample_sizes = []

            except Exception as e:
                logger.error(e)
                exit(0)

        # number of processed sentences / words
        self.stats['processed_s'] += len2.size(0)
        self.stats['processed_w'] += len2.sum()
        self.n_iter += 1
        del loss
        torch.cuda.empty_cache()

    def evaluate(self):
        test_iterator = self.test_data.get_iterator(shuffle=False, group_by_size=False)()
        self.encoder.eval()
        self.decoder.eval()
        self.params.out_file = '{}/predict_{}'.format(self.params.checkpoint_dir, self.params.model_name[:-3])

        file = open(self.params.out_file, 'w',encoding='utf-8')
        total = 0
        out_sents = []
        with torch.no_grad():
            for (sen1, len1) in test_iterator:
                len1, bak_order = len1.sort(descending=True)
                sen1 = sen1[:,bak_order]
                sen1 = sen1.cuda()
                encoded = self.encoder(sen1, len1)
                sent2, len2, _ = self.decoder.generate(encoded)
                total += len2.size(0)
                logger.info('Translating %i sentences.' % total)
                for j in bak_order.argsort().tolist():
                    out1 = self.params.tgt_dico.idx2string(sent2[:, j]).replace('@@ ', '')
                    out_sents.append(out1)
                    file.write(out1 + '\n')
        file.close()
        command = f'perl scripts/multi-bleu.perl {self.params.reference_file} < {self.params.out_file}'
        logger.info(command)
        p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        result = p.communicate()[0].decode("utf-8")
        bleu_multibleu = re.findall(r"BLEU = (.+?),", result)[0]
        logger.info(bleu_multibleu)
        with open('{}/bleu.log'.format(self.params.checkpoint_dir),'a+') as f:
            f.write(f'{self.params.model_name}\t{bleu_multibleu}\n')


    def get_iterator(self):
        if self.params.gpu_num == 1:
            iterator = self.data.get_iterator(shuffle=True, group_by_size=True)()
        else:
            iterator = self.data.get_iterator(shuffle=True, group_by_size=True, partition=self.params.rank)()
        return iterator
