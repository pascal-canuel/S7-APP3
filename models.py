# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class trajectory2seq(nn.Module):
    def __init__(self, hidden_dim, n_layers, int2symb, symb2int, dict_size, device, maxlen):
        super(trajectory2seq, self).__init__()
        # Definition des parametres
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.maxlen = maxlen

        # Definition des couches
        # Couches pour rnn
        # self.handwritten_embedding = nn.Embedding(self.dict_size['handwritten'], hidden_dim)
        self.word_embedding = nn.Embedding(self.dict_size['word'], hidden_dim)
        self.encoder_layer = nn.GRU(2, hidden_dim, n_layers, batch_first=True)
        self.decoder_layer = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True)

        # Couches pour attention
        # À compléter

        # Couche dense pour la sortie
        self.fc = nn.Linear(hidden_dim, self.dict_size['word'])
        self.to(device)

    def encoder(self, x):
        y = torch.transpose(x, 1, 2)
        out, hidden = self.encoder_layer(y)

        return out, hidden

    def decoder(self, encoder_outs, hidden):
        # Initialisation des variables
        max_len = self.maxlen['word']  # Longueur max de la séquence d'un mot (avec padding)
        batch_size = hidden.shape[1]  # Taille de la batch
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long()  # Vecteur d'entrée pour décodage
        vec_out = torch.zeros((batch_size, max_len, self.dict_size['word'])).to(self.device)  # Vecteur de sortie du décodage

        # Boucle pour tous les symboles de sortie
        for i in range(max_len):
            vec_in.detach()
            vec_y = self.word_embedding(vec_in)
            out, hidden = self.decoder_layer(vec_y, hidden)
            out = self.fc(out)
            vec_in = torch.argmax(out, dim=2)
            vec_out[:, i, :] = out[:, 0, :]

        return vec_out, hidden, None

    def forward(self, x):
        # Passant avant
        out, h = self.encoder(x)
        out, hidden, attn = self.decoder(out, h)
        return out, hidden, attn
    

