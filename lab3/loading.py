from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
import csv
from collections import Counter
import numpy as np
from torch.nn.utils.rnn import pad_sequence


@dataclass
class Instance:
    text: str
    label: str


class NLPDataset(Dataset):
    def __init__(self, file_path, vocab=None):
        self.instances = []
        self.vocab = vocab
        self.label_vocab = LabelVocab()
        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                text, label = row[:-1][0], row[-1].strip()
                text = text.split()  # Podijeli tekst u tokene
                self.instances.append(Instance(text, label))
    
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        instance = self.instances[idx]
        text_indices = self.vocab.encode(instance.text) if self.vocab else instance.text
        label_index = self.label_vocab.encode(instance.label) if self.label_vocab else instance.label
        return torch.tensor(text_indices), torch.tensor(label_index)



class Vocab:
    def __init__(self, frequencies, max_size=-1, min_freq=0):
        self.itos = ['<PAD>', '<UNK>']
        self.stoi = {'<PAD>': 0, '<UNK>': 1}
        for token, freq in frequencies.most_common():
            if freq < min_freq or (max_size != -1 and len(self.itos) >= max_size):
                break
            self.stoi[token] = len(self.itos)
            self.itos.append(token)
    
    def encode(self, tokens):
        if isinstance(tokens, list):
            return [self.stoi.get(token, self.stoi['<UNK>']) for token in tokens]
        return self.stoi.get(tokens, self.stoi['<UNK>'])

class LabelVocab:
    def __init__(self):
        self.stoi = {'positive': 0, 'negative': 1}
        self.itos = {0: 'positive', 1: 'negative'}
    
    def encode(self, label):
        return self.stoi[label]
    
    def decode(self, index):
        return self.itos[index]

def build_vocab(dataset, max_size=-1, min_freq=1):
    frequencies = Counter(token for instance in dataset.instances for token in instance.text)
    return Vocab(frequencies, max_size, min_freq)

train_dataset = NLPDataset('sst_train_raw.csv')
text_vocab = build_vocab(train_dataset, max_size=15000, min_freq=1)
label_vocab = LabelVocab()
train_dataset.vocab = text_vocab


def load_glove_embeddings(file_path, vocab):
    embeddings = np.random.normal(0, 1, (len(vocab.itos), 300))
    embeddings[vocab.stoi['<PAD>']] = np.zeros(300)
    if file_path != "no":
        with open(file_path, 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                if word in vocab.stoi:
                    embeddings[vocab.stoi[word]] = vector
    return torch.tensor(embeddings, dtype=torch.float32)

glove_embeddings = load_glove_embeddings("sst_glove_6b_300d.txt", text_vocab)
embedding_layer = torch.nn.Embedding.from_pretrained(glove_embeddings, padding_idx=text_vocab.stoi['<PAD>'])



def pad_collate_fn(batch, pad_index=0):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts])
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=pad_index)
    labels = torch.stack(labels)
    return texts_padded, labels, lengths

batch_size = 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
texts, labels, lengths = next(iter(train_loader))
print(f"Texts: {texts}")
print(f"Labels: {labels}")
print(f"Lengths: {lengths}")
