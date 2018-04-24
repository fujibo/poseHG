import torch
from torch.autograd import Variable
from net import StackedHG
import numpy as np
from torchvision.datasets.folder import default_loader


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--model')
parser.add_argument('--image')
args = parser.parse_args()

# model = StackedHG()
model = torch.load(args.model)
model.eval()

image = default_loader(args.image).astype(np.float32)
image = Variable(image, volatile=True)

# (1, 3, 256, 256) -> (1, 16, 64, 64)
heatmap = model([image])[0]
idx = np.argmax(heatmap.reshape(heatmap.shape[0], -1))
xx, yy = np.unravel_index(indices, (heatmap.shape[1:]))

keypoint = np.arange(heatmap.shape[0]), xx, yy
confidence = heatmap[np.arange(heatmap.shape[0]), xx, yy]

# visualize


# calculate MSE
