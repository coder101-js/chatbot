import torch
import torch.nn as nn
import random

# ========== MODEL DEFINITIONS ==========

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.gru.hidden_size)

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = torch.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

# ========== UTILS ==========

def tokenize(sentence, vocab):
    return [vocab['word2index'].get(word, vocab['word2index'].get('<UNK>', 0)) for word in sentence.lower().split()]

def detokenize(indices, vocab):
    return ' '.join(vocab['index2word'].get(i, '<UNK>') for i in indices)

# ========== MODEL LOADER ==========

def load_model(file_path):
    checkpoint = torch.load(file_path, map_location=torch.device("cpu"))

    vocab = checkpoint["vocab"]

    # ✅ Build vocab dict if needed
    if isinstance(vocab, dict) and 'word2index' in vocab:
        word2index = vocab['word2index']
        index2word = vocab['index2word']
    elif isinstance(vocab, list):
        word2index = {word: idx for idx, word in enumerate(vocab)}
        index2word = {idx: word for idx, word in enumerate(vocab)}
        vocab = {
            'word2index': word2index,
            'index2word': index2word
        }
    else:
        raise ValueError("Unsupported vocab format ❌")

    input_size = output_size = len(vocab['word2index'])
    hidden_size = 256

    encoder = Encoder(input_size, hidden_size)
    decoder = Decoder(hidden_size, output_size)

    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    encoder.eval()
    decoder.eval()

    return encoder, decoder, vocab

# ========== CHAT LOOP ==========

def chat():
    encoder, decoder, vocab = load_model("best_model.pth")

    print("🤖 Chatbot: Hello! Type 'quit' to end the chat.")
    while True:
        sentence = input("🧠 You: ")
        if sentence.lower() == "quit":
            print("👋 Chatbot: Goodbye!")
            break

        input_indices = tokenize(sentence, vocab)
        if not input_indices:
            print("🤖 Chatbot: I didn't understand that.")
            continue

        input_tensor = torch.tensor(input_indices, dtype=torch.long)
        encoder_hidden = encoder.init_hidden()
        for idx in input_tensor:
            _, encoder_hidden = encoder(idx.view(1), encoder_hidden)

        decoder_input = torch.tensor([vocab['word2index'].get('<SOS>', 0)], dtype=torch.long)
        decoder_hidden = encoder_hidden

        output_indices = []
        for _ in range(20):
            output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = output.topk(1)
            next_index = topi.item()

            if next_index == vocab['word2index'].get('<EOS>', -1):
                break

            output_indices.append(next_index)
            decoder_input = torch.tensor([next_index])

        response = detokenize(output_indices, vocab)
        print("🤖 Chatbot:", response)

# ========== RUN ==========

if __name__ == "__main__":
    chat()
