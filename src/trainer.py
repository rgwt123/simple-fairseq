from tqdm import tqdm
from logging import getLogger
import torch
from torch.nn.utils import clip_grad_norm_
from src.utils import get_optimizer
import torch.distributed as dist

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
            loss = self.decoder.loss_fn(scores.view(-1, self.decoder.n_words), sent2[1:].view(-1))

            # check NaN
            if (loss != loss).data.any():
                logger.error("NaN detected")
                exit()
            # optimizer
            loss.backward()
        except Exception as e:
            logger.error(e)
            torch.cuda.empty_cache()
            self.n_iter += 1
            self.oom += 1
            return

        if need_reduction:
            try:
                if self.params.gpu_num > 1:
                    size = float(dist.get_world_size())
                #for param in self.enc_dec_params:
                #        #dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                #    param.grad.data.mul_(1/float(self.params.update_freq))


                clip_grad_norm_(self.enc_dec_params, self.params.clip_grad_norm)
                self.optimizer.step()

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
