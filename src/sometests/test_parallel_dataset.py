import argparse
from src.data.loader import load_data

parser = argparse.ArgumentParser(description='Settings')
params = parser.parse_args()
params.train_data = 'huawei/120w.bin'


if __name__ == '__main__':
    data = load_data(params)
    iterator = data.get_iterator(shuffle=True, group_by_size=True)()
    src, tgt = next(iterator)
    print(src[0].shape)
    for i in range(params.batch_size):
        print(params.tgt_dico.idx2string(tgt[0][:, i]))
        print(params.src_dico.idx2string(src[0][:,i]))