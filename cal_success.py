import pickle
import sys
tags = ['##0','##1','##2','##3','##4']
tablefile = open('test_nist/nist0{}.cn.table.pt'.format(sys.argv[2]),'rb')
changetable = pickle.load(tablefile)
file = open(sys.argv[1])

# w是词组，单个词或者多个词，en_words是一个list[]
def contain(w,en_words,tag='##'):
  words = w.split()
  length = len(words)
  if length == 1:
    for i in range(len(en_words)):
      if en_words[i]==words[0]:
        en_words[i] = tag
        return True
    return False
  else:
    for i in range(len(en_words)-length+1):
      if en_words[i] == words[0]:
        for j in range(length):
          if en_words[i+j]!=words[j]:
            break
          if j == length-1:
            en_words[i]=tag
            for j in range(1,length):
              en_words[i+j]=''
            return True
    return False

  
totalcopywords = 0
successcopywords = 0
totalcopysens = 0
successcopysens = 0
for linenum,enline in enumerate(file):
  enline = enline.strip()
  en_words = enline.split()
  table = changetable[linenum]
  length = len(table)
  success = True
  if length == 0: # 
    continue
  else:
    totalcopysens += 1
    totalcopywords += length
    for i in range(length):
      (_,enphrase), = table[i].items()
      if contain(enphrase,en_words):
        successcopywords += 1
      else:
        success = False
    if success:
      successcopysens += 1
file.close()
tablefile.close()

#print('file total {} lines, {} line need copy, we successfully copy {} sentences, {}'.format(linenum,totalcopysens,successcopysens,successcopysens*1.0/totalcopysens))
print(successcopysens/totalcopysens,'\t',successcopywords/totalcopywords)
#print('total {} phrases need copy, we successfully copy {} phrases, {}.'.format(totalcopywords,successcopywords,successcopywords*1.0/totalcopywords))


  
