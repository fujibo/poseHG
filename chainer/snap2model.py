import numpy as np
import tempfile


def snap2model_parser(path_snapshot, path_model=None):
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


def snap2model_trainer(path_snapshot, path_model=None):
    import chainer
    from dataset import MPIIDataset
    from train import TrainChain
    from net import StackedHG

    train_data = MPIIDataset(split='train')
    model = StackedHG(16)
    train_chain = TrainChain(model)
    optimizer = chainer.optimizers.RMSprop(lr=2.5e-4)
    optimizer.setup(train_chain)

    # original batch size 6
    train_iter = chainer.iterators.SerialIterator(train_data, 1, repeat=True, shuffle=True)
    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=-1)
    trainer = chainer.training.Trainer(updater, (100, 'epoch'), out='')
    chainer.serializers.load_npz(path_snapshot, trainer)

    if path_model is None:
        outfile = tempfile.TemporaryFile()
        chainer.serializers.save_npz(outfile, model)
        outfile.seek(0)
        return outfile

    else:
        chainer.serializers.save_npz(path_model, model)
        return None
