import chainer

from chainer import functions as F

from chainer import training, serializers
from chainer.training import extensions
from chainer.backends import cuda

import numpy as np
from net import StackedHG


class TrainChain(chainer.Chain):
    def __init__(self, model):
        super(TrainChain, self).__init__()
        with self.init_scope():
            self.model = model

    def __call__(self, *in_data):
        if chainer.config.train:
            img, heatmap, idx = in_data

            # (N, C, H, W) -> (N, 16, 64, 64)
            output1, output2 = self.model(img)

            # calculate MSE
            loss = (output1 - heatmap) ** 2 + (output2 - heatmap) ** 2

            # (N, 16, 64, 64) -> (N, 16)
            loss = F.sum(F.sum(loss, axis=3), axis=2)

            loss = 0.5 * F.mean(loss[idx])
            chainer.report({'loss': loss}, self)

            return loss

        else:
            img, = in_data
            _, output = self.model(img)
            return output


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--out', default='results/temp')
    parser.add_argument('--resume', default='')
    parser.add_argument('--dataset', choices=['mpii', 'flic'], default='mpii')
    args = parser.parse_args()

    if args.dataset == 'mpii':
        from dataset import MPIIDataset
        train_data = MPIIDataset(split='train')

    model = StackedHG(16)
    train_chain = TrainChain(model)

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        train_chain.to_gpu()

    optimizer = chainer.optimizers.RMSprop(lr=2.5e-4)
    optimizer.setup(train_chain)

    train_iter = chainer.iterators.MultithreadIterator(train_data, 6, repeat=True, shuffle=True, n_threads=3)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (100, 'epoch'), out=args.out)

    interval = 1, 'epoch'
    trainer.extend(extensions.observe_lr(), trigger=interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.PrintReport(['epoch', 'lr', 'main/loss']))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.ExponentialShift('lr', 0.1),
                   trigger=(50, 'epoch'))
    trainer.extend(
        extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}'),
        trigger=(10, 'epoch'))
    if args.resume:
        serializers.load_npz(args.resume, trainer)

    trainer.run()
    serializers.save_npz(f'{args.out}/model.npz', model)


if __name__ == '__main__':
    main()
