import numpy as np
import tempfile


def snap2model(path_snapshot, path_model=None):
    """convert snapshot to model

    :param path_snapshot: str
    :param path_model: str, default None
    :return: file descriptor (path_model is None) or None (otherwise)
    """
    snapshot = np.load(path_snapshot)

    model = dict()
    for key in snapshot.keys():
        parse = key.split('/')
        if parse[0] == 'updater' and parse[1] == 'optimizer:main':
            if parse[2] == 'model':
                model_key = '/'.join(parse[3:-1])
                model[model_key] = snapshot[key]

    if path_model is None:
        outfile = tempfile.TemporaryFile()
        np.savez(outfile, **model)
        outfile.seek(0)
        return outfile

    else:
        np.savez(path_model, **model)
        return None
