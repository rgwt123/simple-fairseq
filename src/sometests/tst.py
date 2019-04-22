import re

optim_params = {}
s = 'adam_inverse_sqrt,lr=0.0005'
if "," in s:
    method = s[:s.find(',')]
    print(method)
    for x in s[s.find(',') + 1:].split(','):
        split = x.split('=')
        assert len(split) == 2
        assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
        optim_params[split[0]] = float(split[1])

print(optim_params)