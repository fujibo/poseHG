import chainer

from chainer import functions as F

from chainer import training, serializers
from chainer.training import extensions

from chainer.backends import cuda

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
            with cuda.get_device_from_id(self._device_id):
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
    parser.add_argument('--gpu', type=str, default='-1')
    parser.add_argument('--out', default='results/temp')
    parser.add_argument('--resume', default='')
    parser.add_argument('--dataset', choices=['mpii', 'flic'], default='mpii')
    args = parser.parse_args()

    if args.dataset == 'mpii':
        from dataset import MPIIDataset
        train_data = MPIIDataset(split='train')

    # '0,1' -> [0, 1]
    gpus = list(map(lambda device: int(device), args.gpu.split(',')))
    devices = {'main': gpus[0]}

    if len(gpus) >= 2:
        devices.update({'second': gpus[1]})

    model = StackedHG(16)
    train_chain = TrainChain(model)

    if devices['main'] >= 0:
        cuda.get_device_from_id(devices['main']).use()

    # 2.5 / 6 * 32
    # optimizer = chainer.optimizers.RMSprop(lr=1.33e-3)
    optimizer = chainer.optimizers.RMSprop(lr=2.5e-4)
    optimizer.setup(train_chain)

    # original batch size 6
    if len(gpus) >= 2:
        batch_size = 32
    else:
        batch_size = 6

    train_iter = chainer.iterators.MultithreadIterator(train_data, batch_size, repeat=True, shuffle=True, n_threads=3)

    updater = training.ParallelUpdater(train_iter, optimizer, devices=devices)
    trainer = training.Trainer(updater, (100, 'epoch'), out=args.out)

    interval = 1, 'epoch'
    trainer.extend(extensions.observe_lr(), trigger=interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.PrintReport(['epoch', 'lr', 'main/loss']))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.ExponentialShift('lr', 0.2),
                   trigger=(70, 'epoch'))
    trainer.extend(
        extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}'),
        trigger=(10, 'epoch'))
    if args.resume:
        serializers.load_npz(args.resume, trainer)

    trainer.run()
    serializers.save_npz(f'{args.out}/model.npz', model)


if __name__ == '__main__':
    main()
