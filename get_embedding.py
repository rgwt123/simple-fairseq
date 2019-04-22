import torch
import sys
import io

data = torch.load(sys.argv[1],'cpu')
src_dico = data['params'].src_dico
tgt_dico = data['params'].tgt_dico
print('dico lengths:',len(src_dico),len(tgt_dico))
# real words start from id 4
src_emb = data['encoder']['embeddings.weight']
tgt_emb = data['decoder']['embeddings.weight']
print('embeddings shape:',src_emb.shape,tgt_emb.shape)

# get torch.nn.parameter.Parameter
en_emb = tgt_emb
zh_emb = src_emb
zh_vocab = src_dico.id2word
en_vocab = tgt_dico.id2word

with io.open('zh.vec', 'w', encoding='utf-8') as f:
    f.write(u"%i %i\n" % zh_emb.size(0)-4,zh_emb.size(1))
    for i in range(4,len(zh_vocab)):
        f.write(u"%s %s\n" % (zh_vocab[i], " ".join('%.5f' % x for x in zh_emb[i])))
        

with io.open('en.vec', 'w', encoding='utf-8') as f:
    f.write(u"%i %i\n" % en_emb.size(0)-4,en_emb.size(1))
    for i in range(4,len(en_vocab)):
        f.write(u"%s %s\n" % (en_vocab[i], " ".join('%.5f' % x for x in en_emb[i])))
