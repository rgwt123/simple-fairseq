import torch

data=torch.load('huawei_delay4/layernorm_fix.pt','cpu')

encoder = data['encoder']
decoder = data['decoder']

name='layers.{}.layer_norms.{}.gain'
rename='layers.{}.layer_norms.{}.weight'

for i in range(0,6):
    for j in range(0,2):
        tname = name.format(i,j)
        trename = rename.format(i,j)
        encoder[trename] = encoder.pop(tname)

for i in range(0,6):
    for j in range(0,3):
        tname = name.format(i,j)
        trename = rename.format(i,j)
        decoder[trename] = decoder.pop(tname)

torch.save(data, 'huawei_delay4/fix_layernorm.pt')