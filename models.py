import torch
from torch import nn


class trajectory2seq(nn.Module):
    def __init__(self, hidden_dim, n_layers, int2symb, symb2int, dict_size, device, maxlen):
        super(trajectory2seq, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.maxlen = maxlen

        self.word_embedding = nn.Embedding(self.dict_size['word'], hidden_dim)
        self.encoder_layer = nn.GRU(2, hidden_dim, n_layers, batch_first=True)
        self.decoder_layer = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, self.dict_size['word'])
        self.to(device)

    def encoder(self, x):
        out, hidden = self.encoder_layer(x)

        return out, hidden

    def decoder(self, encoder_outs, hidden):
        max_len = self.maxlen['word']
        batch_size = hidden.shape[1]
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long()
        vec_out = torch.zeros((batch_size, max_len, self.dict_size['word'])).to(self.device)

        for i in range(max_len):
            vec_in.detach()
            vec_y = self.word_embedding(vec_in)
            out, hidden = self.decoder_layer(vec_y, hidden)
            out = self.fc(out)
            vec_in = torch.argmax(out, dim=2)
            vec_out[:, i, :] = out[:, 0, :]

        return vec_out, hidden, None

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        out, h = self.encoder(x)
        out, hidden, attn = self.decoder(out, h)
        return out, hidden, attn
    

