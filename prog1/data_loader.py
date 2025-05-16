import numpy as np
from dataset import Data
class DataLoader:

    def __init__(self, train_set, dev_set, batch_size):
        self.train_set = train_set
        self.dev_set = dev_set
        self.batch_size = batch_size
        self.index = 0
        self.epoch = 0
        self.updates = 0
        

    def batch_data(self):
        np.random.shuffle(self.train_set)

    def get_batch(self):
        self.updates = self.updates + 1
        current_batch = []
        # if next batch exceeds array size, restart index and reshuffle batches
        if (self.index + self.batch_size > len(self.train_set)):
            self.index = 0
            self.epoch = self.epoch + 1
            self.batch_data()

        for i in range(self.batch_size):
            current_batch.append(self.train_set[self.index + i])

        self.index = self.index + self.batch_size

        return current_batch