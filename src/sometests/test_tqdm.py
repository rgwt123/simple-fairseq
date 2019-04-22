from tqdm import tqdm
import time

def getiter(x):
  def iterator():
    for xx in x:
      yield x
  return iterator

itr = getiter(list(range(100)))()
for i in tqdm(itr,total=100):
  time.sleep(1)
