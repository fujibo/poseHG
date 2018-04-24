from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

import os


class MPIIDataset(Dataset):
    """MPII dataset."""

    def __init__(self, root_dir='~/data/MPII', split='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = os.path.expanduser(root_dir)
        with open(f'../pose-hg-demo/annot/{split}_images.txt') as f:
            fnames = f.read().split('\n')

        # 24987 = 18079 (train) + 6908 (test)
        train_indices = annot['RELEASE']['img_train'][0][0][0]
        self.transform = transform

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, idx):
        path = self._paths[idx]
        img_path = os.path.join(self.root_dir, path[0])
        image = io.imread(img_name)
        landmarks = self._paths.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
