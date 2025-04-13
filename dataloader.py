import numpy as np


class Dataloader:
    def __init__(self, X, y, batch_size=64, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = X.shape[0]
        self.indices = np.arange(self.num_samples)
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
            
    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.current_idx >= self.num_samples:
            raise StopIteration
        idx = self.indices[self.current_idx:self.current_idx + self.batch_size]
        batch_X = self.X[idx]
        batch_y = self.y[idx]
        self.current_idx += self.batch_size
        return batch_X, batch_y
    