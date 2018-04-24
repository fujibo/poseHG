import torch
from torch import nn
from torch.autograd import Variable

from net import StackedHG
from eval import evaluate

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--dataset', choices=['mpii', 'flic'])
args = parser.parse_args()

if args.dataset == 'mpii':
    from datasets import MPIIDataset
    train_data, test_data = MPIIDataset()

model = StackedHG(16)

if args.gpu >= 0:
    model.cuda(args.gpu)

optimizer = torch.optim.RMSprop(model.parameters(), lr=2.5e-4)
epochs = 100

model.train()
for epoch in epochs:
    for batch in it:
        pass

    else:
        model.eval()
        # evaluate here
        model.train()
