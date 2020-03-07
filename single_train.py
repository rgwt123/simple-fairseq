from src.model import build_mt_model
from src.data.loader import load_data
from src.trainer import TrainerMT
from tqdm import tqdm
from logging import getLogger
logger = getLogger()

def main(params):
    data = load_data(params, name='train')
    test_data = load_data(params, name='test')
    encoder, decoder, num_updates = build_mt_model(params)
    trainer = TrainerMT(encoder, decoder, data, test_data, params, num_updates)

    for i in range(trainer.epoch, params.max_epoch):
        logger.info("==== Starting epoch %i ...====" % trainer.epoch)
        trainer.train_epoch()
        tqdm.write('Finish epcoh %i.' % i)
