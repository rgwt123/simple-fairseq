from tqdm import tqdm
from logging import getLogger
import torch
from torch.nn.utils import clip_grad_norm_
from src.utils import get_optimizer
import torch.distributed as dist
from src.distributed_utils import all_gather_list,all_reduce_and_rescale_tensors
from itertools import chain

logger = getLogger()


class TrainerMT():
    def __init__(self, encoder, decoder, data, params, num_updates):
        self.encoder = encoder
        self.decoder = decoder
        self.data = data
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

        # save epoch checkpoint
        if self.params.gpu_num == 1:
            save = True
        else:
            if self.params.rank == 0:
              save = True
            else:
              save = False
        if save:
            checkpoint = {
                'encoder': self.encoder.state_dict(),
                'decoder': self.decoder.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'params': self.params,
                'epoch': self.epoch,
                'num_updates': self.optimizer._num_updates
            }

            torch.save(checkpoint, '{}/model_epoch{}.pt'.format(self.params.checkpoint_dir, self.epoch))
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
                # print(f'before reduce sample_size:{self.params.rank},iter-{self.n_iter} {sample_sizes}')
                # now we get the total token nums of all delay batch and all gpus
                
                if self.params.gpu_num > 1:
                    sample_sizes = all_gather_list(self.sample_sizes)
                    sample_sizes = list(chain.from_iterable(sample_sizes))
                    sample_size = sum(sample_sizes)
                    grads = [p.grad.data for p in self.enc_dec_params if p.requires_grad and p.grad is not None]
                    all_reduce_and_rescale_tensors(grads,float(sample_size)/self.params.gpu_num)
                    # print(f'after reduce gpu-{self.params.rank},iter-{self.n_iter} {self.enc_dec_params[0].grad.data[0][0:20]}')
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

    def get_iterator(self):
        if self.params.gpu_num == 1:
            iterator = self.data.get_iterator(shuffle=True, group_by_size=True)()
        else:
            iterator = self.data.get_iterator(shuffle=True, group_by_size=True, partition=self.params.rank)()
        return iterator
