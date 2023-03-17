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

        self.char2int = dict()
        self.char2int = {start_symbol: 0, stop_symbol: 1, pad_symbol: 2}

        for i, char in enumerate(alphabet):
            self.char2int[char] = i + 3

        self.int2char = dict()
        self.int2char = {v: k for k, v in self.char2int.items()}

        # Ajout du padding aux séquences
        max_word_len = max([len(seq[0]) for seq in self.data]) + 1
        max_handwritten_seq_len = max([seq[1].shape[1] for seq in self.data]) + 1

        self.padded_data = dict()
        self.padded_data['word'] = {} # target
        self.padded_data['handwritten'] = {} # input to predict
        
        for i, item in enumerate(self.data):
            # split string into list of characters
            word = list(item[0])
            handwritten = item[1]

            word.append(self.stop_symbol)
            while len(word) < max_word_len:
                word.append(self.pad_symbol)

            # get last value of handwritten sequence
            last_coord = handwritten[:, -1]
            while handwritten.shape[1] < max_handwritten_seq_len:
                handwritten = np.concatenate((handwritten, last_coord.reshape(-1, 1)), axis=1)

            self.padded_data['word'][i] = word
            self.padded_data['handwritten'][i] = handwritten
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        handwritten = self.padded_data['handwritten'][idx]
        word = self.padded_data['word'][idx]
        word = [self.char2int[i] for i in word]
        return torch.tensor(handwritten), torch.tensor(word)

    def visualisation(self, idx):
        handwritten, word = [i.numpy() for i in self[idx]]
        word = [self.int2char[c] for c in word]
        print('Word: ', ' '.join(word))
        print('Handwritten: ', handwritten)

if __name__ == "__main__":
    # Code de test pour aider à compléter le dataset
    a = HandwrittenWords('data_trainval.p')
    for i in range(10):
        a.visualisation(np.random.randint(0, len(a)))