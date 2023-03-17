import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import re
import pickle

class HandwrittenWords(Dataset):
    """Ensemble de donnees de mots ecrits a la main."""

    def __init__(self, filename):
        # Lecture du text
        self.pad_symbol     = pad_symbol = '<pad>'
        self.start_symbol   = start_symbol = '<sos>'
        self.stop_symbol    = stop_symbol = '<eos>'

        self.data = dict()
        with open(filename, 'rb') as fp:
            self.data = pickle.load(fp)

        # Extraction des symboles
        alphabet = 'abcdefghijklmnopqrstuvwxyz'

        self.symb2int = dict()
        self.symb2int['handwritten'] = {}
        self.symb2int['word'] = {start_symbol: 0, stop_symbol: 1, pad_symbol: 2}

        for i, char in enumerate(alphabet):
            self.symb2int['word'][char] = i + 3

        self.int2symb = dict()
        self.int2symb['handwritten'] = {}
        self.int2symb['word'] = {v: k for k, v in self.symb2int['word'].items()}

        # Ajout du padding aux séquences
        self.max_len = dict()
        self.max_len['word'] = max([len(seq[0]) for seq in self.data]) + 1
        self.max_len['handwritten'] = max([seq[1].shape[1] for seq in self.data])

        self.padded_data = dict()
        self.padded_data['word'] = {} # target
        self.padded_data['handwritten'] = {} # input to predict
        
        for i, item in enumerate(self.data):
            # split string into list of characters
            word = list(item[0])
            handwritten = item[1]

            word.append(self.stop_symbol)
            while len(word) < self.max_len['word']:
                word.append(self.pad_symbol)

            # get last value of handwritten sequence
            last_coord = handwritten[:, -1]
            while handwritten.shape[1] < self.max_len['handwritten']:
                # TODO: test padding with another value
                handwritten = np.concatenate((handwritten, last_coord.reshape(-1, 1)), axis=1)

            self.padded_data['word'][i] = word
            self.padded_data['handwritten'][i] = handwritten

        self.dict_size = {'word': len(self.int2symb['word']), 'handwritten': len(self.int2symb['handwritten'])}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        handwritten = self.padded_data['handwritten'][idx]
        word = self.padded_data['word'][idx]
        word = [self.symb2int['word'][i] for i in word]
        return torch.tensor(handwritten), torch.tensor(word)

    def visualisation(self, idx):
        handwritten, word = [i.numpy() for i in self[idx]]
        word = [self.int2symb['word'][c] for c in word]
        # print('Word: ', ' '.join(word))
        # print('Handwritten: ', handwritten)
        # plot handwritten 2 sequence of coordinates
        plt.plot(handwritten[0], handwritten[1])
        plt.title(' '.join(word))
        plt.show()

if __name__ == "__main__":
    # Code de test pour aider à compléter le dataset
    a = HandwrittenWords('data_trainval.p')
    for i in range(10):
        a.visualisation(np.random.randint(0, len(a)))