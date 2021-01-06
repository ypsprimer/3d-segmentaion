import torch
from torch.utils.data import DataLoader, Dataset, Sampler

class TestDataset(Dataset):

    def __init__(self):
        self.data = torch.randn(size=(100, 256, 256))

    def __len__(self):
        return 100

    def __getitem__(self, item):
        return self.data[item, :, :]

class TestSampler(Sampler):

    def __init__(self, *args):
        pass

    def __len__(self):
        return 100

    def __iter__(self):
        return iter(range(100))


if __name__ == '__main__':
    dl = DataLoader(dataset=TestDataset(),
                    sampler=TestSampler(),
                    batch_size=4,
                    pin_memory=True,
                    num_workers=4)

    for _, sample in enumerate(dl):
        print(sample)