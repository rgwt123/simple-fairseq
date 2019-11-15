from src.model import build_mt_model
from src.data.loader import load_data
from src.trainer import TrainerMT
from tqdm import tqdm
from logging import getLogger
logger = getLogger()

def main(params):
    data = load_data(params, name='train')
    encoder, decoder, num_updates = build_mt_model(params)
    trainer = TrainerMT(encoder, decoder, data, params, num_updates)

    for i in range(trainer.epoch, 30):
        logger.info("==== Starting epoch %i ...====" % trainer.epoch)
        trainer.train_epoch()
        tqdm.write('Finish epcoh %i.' % i)
