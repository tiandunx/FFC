import os
import json

support_types = ('str', 'int', 'bool', 'float', 'none')


def convert_param(original_lists):
    assert isinstance(original_lists, list), 'The type is not right : {:}'.format(original_lists)
    ctype, value = original_lists[0], original_lists[1]
    assert ctype in support_types, 'Ctype={:}, support={:}'.format(ctype, support_types)
    is_list = isinstance(value, list)
    if not is_list:
        value = [value]
    outs = []
    for x in value:
        if ctype == 'int':
            x = int(x)
        elif ctype == 'str':
            x = str(x)
        elif ctype == 'bool':
            x = bool(int(x))
        elif ctype == 'float':
            x = float(x)
        elif ctype == 'none':
            assert x == 'None', 'for none type, the value must be None instead of {:}'.format(x)
            x = None
        else:
            raise TypeError('Does not know this type : {:}'.format(ctype))
        outs.append(x)
    if not is_list:
        outs = outs[0]
    return outs


def load_config(path, extra=None, logger=None):
    path = str(path)
    assert os.path.exists(path), 'Can not find {:}'.format(path)
    # Reading data back
    with open(path, 'r') as f:
        data = json.load(f)
    content = {k: convert_param(v) for k, v in data.items()}
    if extra is not None:
        if isinstance(extra, dict):
            content = {**content, **extra}
    # Arguments = namedtuple('Configure', ' '.join(content.keys()))
    # content = Arguments(**content)
    if logger is not None:
        logger.info(content)
    return content


if __name__ == '__main__':
    config = load_config('../config/optim_config')
    print('Finish!')
