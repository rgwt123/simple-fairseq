import re
import inspect
from logging import getLogger
from src.adam_inverse_sqrt_with_warmup import AdamInverseSqrtWithWarmup

logger = getLogger()


def get_optimizer(parameters, s):
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adam_inverse_sqrt':
        optim_fn = AdamInverseSqrtWithWarmup
        optim_params['lr'] = optim_params.get('lr', 0.0005)
        optim_params['betas'] = (optim_params.get('beta1', 0.9), optim_params.get('beta2', 0.98))
        optim_params['warmup_updates'] = optim_params.get('warmup_updates', 4000)
        optim_params['weight_decay'] = optim_params.get('weight_decay', 0.0001)
        optim_params.pop('beta1', None)
        optim_params.pop('beta2', None)
    else:
        raise Exception('write yourself method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getfullargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))
    logger.info(optim_params)
    return optim_fn(parameters, **optim_params)
