import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os, time, signal, re, random, sys
from datetime import datetime

# ========== Dataset & Preprocessing ==========

def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9?.!,']+", " ", text)
    return text.strip()

class ChatDataset(Dataset):
    def __init__(self, path='data.txt'):
        self.pairs = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if ',' not in line:
                    continue
                q, a = line.strip().split(',', 1)

                self.pairs.append((clean(q), clean(a)))

        self.token2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
        self.idx2token = ['<pad>', '<sos>', '<eos>']
        self.build_vocab()

    def build_vocab(self):
        for q, a in self.pairs:
            for word in (q + " " + a).split():
                if word not in self.token2idx:
                    self.idx2token.append(word)
                    self.token2idx[word] = len(self.idx2token) - 1

    def encode(self, sentence):
        return [self.token2idx[w] for w in sentence.split() if w in self.token2idx]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        q, a = self.pairs[idx]
        q_ids = self.encode(q)
        a_ids = [1] + self.encode(a) + [2]  # <sos> ... <eos>
        return torch.tensor(q_ids), torch.tensor(a_ids)

def collate_fn(batch):
    qs, as_ = zip(*batch)
    qs = nn.utils.rnn.pad_sequence(qs, batch_first=True)
    as_ = nn.utils.rnn.pad_sequence(as_, batch_first=True)
    return qs, as_

# ========== Models ==========

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        x = self.embed(x)
        outputs, hidden = self.gru(x)
        return hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embed(x.unsqueeze(1))
        output, hidden = self.gru(x, hidden)
        return self.fc(output.squeeze(1)), hidden

# ========== Checkpoint Utils ==========

def save_checkpoint(path, encoder, decoder, enc_opt, dec_opt, best_loss, vocab):
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'enc_opt': enc_opt.state_dict(),
        'dec_opt': dec_opt.state_dict(),
        'best_loss': best_loss,
        'vocab': vocab
    }, path)

def load_checkpoint(path, encoder, decoder, enc_opt, dec_opt):
    checkpoint = torch.load(path)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    enc_opt.load_state_dict(checkpoint['enc_opt'])
    dec_opt.load_state_dict(checkpoint['dec_opt'])
    return checkpoint['best_loss'], checkpoint['vocab']

# ========== Global for SIGINT ==========

should_exit = False
def signal_handler(sig, frame):
    global should_exit
    print("\n[⚠️] Ctrl+C detected! Will save and exit after current batch.")
    should_exit = True
signal.signal(signal.SIGINT, signal_handler)

# ========== Main Training Loop ==========

def train():
    print("[🚀] Loading dataset...")
    dataset = ChatDataset('data.txt')
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    vocab_size = len(dataset.token2idx)
    encoder = Encoder(vocab_size, 128, 256)
    decoder = Decoder(vocab_size, 128, 256)
    enc_opt = optim.Adam(encoder.parameters(), lr=0.001)
    dec_opt = optim.Adam(decoder.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    best_loss = float('inf')
    start_time = time.time()
    pause_duration = random.randint(60, 120)  # 1–2 min rest
    last_pause = time.time()

    # Resume if checkpoint exists
    if os.path.exists("checkpoint.pth"):
        print("[💾] Resuming from checkpoint...")
        best_loss, vocab = load_checkpoint("checkpoint.pth", encoder, decoder, enc_opt, dec_opt)
        dataset.token2idx = vocab

    for epoch in range(1000):  # you control how long this runs
        for i, (src, tgt) in enumerate(loader):
            enc_hidden = encoder(src)
            dec_input = tgt[:, 0]
            loss = 0

            for t in range(1, tgt.size(1)):
                output, enc_hidden = decoder(dec_input, enc_hidden)
                loss += loss_fn(output, tgt[:, t])
                dec_input = tgt[:, t]  # teacher forcing

            enc_opt.zero_grad()
            dec_opt.zero_grad()
            loss.backward()
            enc_opt.step()
            dec_opt.step()

            avg_loss = loss.item() / tgt.size(1)
            print(f"[{epoch}:{i}] Loss: {avg_loss:.4f}")

            # Save if best
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint("best_model.pth", encoder, decoder, enc_opt, dec_opt, best_loss, dataset.token2idx)
                print(f"[🔥] Best model saved! Loss: {best_loss:.4f}")

            # Regular checkpoint
            if i % 20 == 0:
                save_checkpoint("checkpoint.pth", encoder, decoder, enc_opt, dec_opt, best_loss, dataset.token2idx)

            # Graceful exit on Ctrl+C
            if should_exit:
                save_checkpoint("checkpoint_exit.pth", encoder, decoder, enc_opt, dec_opt, best_loss, dataset.token2idx)
                print("[✅] Exit checkpoint saved. Bye!")
                sys.exit(0)

            # Pause every 30 min
            if time.time() - last_pause > 1800:
                print(f"[🧘] 30 minutes done. Resting for {pause_duration} sec...")
                time.sleep(pause_duration)
                last_pause = time.time()
                pause_duration = random.randint(60, 120)

train()
