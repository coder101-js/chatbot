import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import time
import os
import json

# Define constants
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"
MAX_LENGTH = 20

# Dataset class
class ChatDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.pairs = [(pair['input'], pair['target']) for pair in data]
        self.token2idx = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
        self.idx2token = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

        for pair in self.pairs:
            for sentence in pair:
                for word in sentence.split():
                    if word not in self.token2idx:
                        self.token2idx[word] = len(self.idx2token)
                        self.idx2token.append(word)

        self.data = [
            (self.encode(pair[0]), self.encode(pair[1]))
            for pair in self.pairs
        ]

    def encode(self, sentence):
        return [self.token2idx.get(word, self.token2idx[UNK_TOKEN]) for word in sentence.split()] + [self.token2idx[EOS_TOKEN]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_vocab(self):
        return {
            "word2index": self.token2idx,
            "index2word": {i: w for i, w in enumerate(self.idx2token)}
        }

# Encoder RNN
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

# Decoder RNN
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x).unsqueeze(1)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output.squeeze(1))
        return output, hidden

# Utility functions

def pad_sequences(sequences, pad_value=0):
    max_len = max(len(seq) for seq in sequences)
    return [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = pad_sequences(inputs)
    targets = pad_sequences(targets)
    return torch.tensor(inputs), torch.tensor(targets)

def save_checkpoint(path, encoder, decoder, enc_opt, dec_opt, best_loss, vocab):
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'enc_opt': enc_opt.state_dict(),
        'dec_opt': dec_opt.state_dict(),
        'best_loss': best_loss,
        'vocab': vocab
    }, path)

# Training loop

def train():
    dataset = ChatDataset("data.json")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    vocab_size = len(dataset.token2idx)
    embed_size = 64
    hidden_size = 128

    encoder = Encoder(vocab_size, embed_size, hidden_size)
    decoder = Decoder(vocab_size, embed_size, hidden_size)

    enc_opt = optim.Adam(encoder.parameters(), lr=0.001)
    dec_opt = optim.Adam(decoder.parameters(), lr=0.001)

    best_loss = float('inf')

    for epoch in range(10):
        total_loss = 0
        for inputs, targets in dataloader:
            enc_opt.zero_grad()
            dec_opt.zero_grad()

            _, hidden = encoder(inputs)
            dec_input = torch.full((inputs.size(0),), dataset.token2idx[SOS_TOKEN], dtype=torch.long)
            loss = 0

            for t in range(targets.size(1)):
                output, hidden = decoder(dec_input, hidden)
                loss += F.cross_entropy(output, targets[:, t])
                dec_input = targets[:, t]

            loss.backward()
            enc_opt.step()
            dec_opt.step()

            total_loss += loss.item() / targets.size(1)

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        vocab = dataset.get_vocab()

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint("best_model.pth", encoder, decoder, enc_opt, dec_opt, best_loss, vocab)
            print(f"[🔥] Best model saved! Loss: {best_loss:.4f}")

        save_checkpoint("checkpoint.pth", encoder, decoder, enc_opt, dec_opt, best_loss, vocab)

    save_checkpoint("checkpoint_exit.pth", encoder, decoder, enc_opt, dec_opt, best_loss, vocab)

if __name__ == '__main__':
    train()
