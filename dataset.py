import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pickle


class HandwrittenWords(Dataset):
    def __init__(self, filename):
        self.pad_symbol = pad_symbol = '<pad>'
        self.start_symbol = start_symbol = '<sos>'
        self.stop_symbol = stop_symbol = '<eos>'

        self.raw_data = dict()
        with open(filename, 'rb') as fp:
            self.raw_data = pickle.load(fp)

        self.symb2int = dict()
        self.symb2int['word'] = {start_symbol: 0, stop_symbol: 1, pad_symbol: 2}

        alphabet = 'abcdefghijklmnopqrstuvwxyz'

        for idx, char in enumerate(alphabet):
            self.symb2int['word'][char] = idx + 3

        self.int2symb = dict()
        self.int2symb['word'] = {v: k for k, v in self.symb2int['word'].items()}

        self.max_len = dict()
        self.max_len['word'] = max([len(seq[0]) for seq in self.raw_data]) + 1
        self.max_len['handwritten'] = max([seq[1].shape[1] for seq in self.raw_data])

        self.padded_data = dict()
        self.padded_data['word'] = {}
        self.padded_data['handwritten'] = {}

        for idx, item in enumerate(self.raw_data):
            word = list(item[0])
            word.append(self.stop_symbol)
            while len(word) < self.max_len['word']:
                word.append(self.pad_symbol)

            handwritten = item[1]
            # last_coord = handwritten[:, -1]
            # pad_coord = last_coord
            pad_coord = np.array([np.NAN, np.NAN])
            while handwritten.shape[1] < self.max_len['handwritten']:
                handwritten = np.concatenate((handwritten, pad_coord.reshape(-1, 1)), axis=1)

            self.padded_data['word'][idx] = word
            self.padded_data['handwritten'][idx] = handwritten

        self.dict_size = {'word': len(self.int2symb['word'])}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        handwritten = self.padded_data['handwritten'][idx]
        word = self.padded_data['word'][idx]
        word = [self.symb2int['word'][c] for c in word]

        return torch.tensor(handwritten), torch.tensor(word)

    def visualisation(self, idx):
        handwritten, word = [v.numpy() for v in self[idx]]
        word = [self.int2symb['word'][c] for c in word]

        plt.plot(handwritten[0], handwritten[1])
        plt.title(' '.join(word))
        plt.show()


if __name__ == "__main__":
    a = HandwrittenWords('data_train_val.p')
    for i in range(10):
        a.visualisation(np.random.randint(0, len(a)))
