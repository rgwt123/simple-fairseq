from logging import getLogger
logger = getLogger()
import os
import torch
import torch.distributed as dist
import signal
import threading
import random
from src.model import build_mt_model
from src.data.loader import load_data
from src.trainer import TrainerMT
from src.distributed_utils import suppress_output,is_master
from tqdm import tqdm


def run(params, error_queue):
    try:
        # start training
        logger.info(params)
        if not torch.cuda.is_available():
            raise NotImplementedError('Training on CPU is not supported')
        torch.cuda.set_device(params.rank)
        torch.manual_seed(params.seed)
        logger.info('Process %s is now running in gpu:%s',os.getpid(), torch.cuda.current_device())

        data = load_data(params, 'train')
        print(data.get_iterator(shuffle=True, group_by_size=True, partition=params.rank))

        encoder, decoder, num_updates = build_mt_model(params)
        trainer = TrainerMT(encoder, decoder, data, params, num_updates)
        for i in range(trainer.epoch, params.max_epoch):
            logger.info("==== Starting epoch %i ...====" % trainer.epoch)
            trainer.train_epoch()
            tqdm.write('Finish epcoh %i.' % i)


    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((params.rank, traceback.format_exc()))


def init_processes(params, fn, error_queue):
    """ Initialize the distributed environment. """
    dist.init_process_group('nccl', init_method=params.init_method, rank=params.rank, world_size=params.gpu_num)
    logger.info('| distributed init (rank {}): {}'.format(params.rank, params.init_method))
    suppress_output(is_master(params))
    fn(params, error_queue)


def main(params):
    mp = torch.multiprocessing.get_context('spawn')
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)
    port = random.randint(10000, 20000)
    params.init_method = 'tcp://localhost:{port}'.format(port=port)
    processes = []
    for rank in range(params.gpu_num):
        params.rank = rank
        p = mp.Process(target=init_processes, args=(params, run, error_queue, ), daemon=True)
        p.start()
        error_handler.add_child(p.pid)
        processes.append(p)
    for p in processes:
        p.join()

class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        self.children_pids.append(pid)

    def error_listener(self):
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = "\n\n-- Tracebacks above this line can probably be ignored --\n\n"
        msg += original_trace
        raise Exception(msg)
