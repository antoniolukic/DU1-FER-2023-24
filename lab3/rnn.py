import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, RMSprop, Adagrad
from nlpdataset import *
from baseline import *

class RNNModel(nn.Module):
    def __init__(self, cell_type, embedding_layer, hidden_size=150, num_layers=2, dropout=0.0, bidirectional=False, attention=False):
        super(RNNModel, self).__init__()
        self.embedding_layer = embedding_layer
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.attention = attention
        
        if cell_type == 'vanilla':
            self.rnn1 = nn.RNN(input_size=embedding_layer.embedding_dim,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=False,  # time-first format
                               dropout=dropout,
                               bidirectional=bidirectional)
            self.rnn2 = nn.RNN(input_size=hidden_size * (2 if bidirectional else 1),
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=False,
                               dropout=dropout,
                               bidirectional=bidirectional)
        elif cell_type == 'gru':
            self.gru1 = nn.GRU(input_size=embedding_layer.embedding_dim,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=False,  # time-first format
                               dropout=dropout,
                               bidirectional=bidirectional)
            self.gru2 = nn.GRU(input_size=hidden_size * (2 if bidirectional else 1),
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=False,
                               dropout=dropout,
                               bidirectional=bidirectional)
        elif cell_type == 'lstm':
            self.lstm1 = nn.LSTM(input_size=embedding_layer.embedding_dim,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                batch_first=False,  # time-first format
                                dropout=dropout,
                                bidirectional=bidirectional)
            self.lstm2 = nn.LSTM(input_size=hidden_size * (2 if bidirectional else 1),
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                batch_first=False,
                                dropout=dropout,
                                bidirectional=bidirectional)
        
        direction_multiplier = 2 if bidirectional else 1
        self.fc1 = nn.Linear(hidden_size * direction_multiplier, 150)
        self.fc2 = nn.Linear(150, 1)

        if self.attention:
            self.W1 = nn.Linear(hidden_size * direction_multiplier, hidden_size // 2)
            self.W2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        embedded = self.embedding_layer(x)
        embedded = embedded.transpose(0, 1)
        
        if hasattr(self, 'rnn1'):
            output, hidden = self.rnn1(embedded)
            output, hidden = self.rnn2(output)
        elif hasattr(self, 'gru1'):
            output, hidden = self.gru1(embedded)
            output, hidden = self.gru2(output)
        elif hasattr(self, 'lstm1'):
            output, (hidden, _) = self.lstm1(embedded)
            output, (hidden, _) = self.lstm2(output)
        
        if isinstance(hidden, tuple):  # LSTM returns (hidden_state, cell_state)
            hidden = hidden[0]
        
        if self.attention:
            score = self.W2(torch.tanh(self.W1(output)))  # (seq_len, batch_size, 1)
            attention_weights = F.softmax(score, dim=0)  # (seq_len, batch_size, 1)
            context_vector = torch.sum(attention_weights * output, dim=0)  # (batch_size, hidden_size * direction_multiplier)
            hidden = context_vector
        else:
            if self.bidirectional:
                hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                hidden = hidden[-1]
        
        h1 = self.fc1(hidden)
        h2 = F.relu(h1)
        #h2 = F.leaky_relu(h1, negative_slope=0.01)
        #h2 = torch.tanh(h1)
        #h2 = F.elu(h1)
        h3 = self.fc2(h2)
        return h3

def main_rnn(seed, model_name, epochs, batch_size, hidden_size, num_layers, dropout, optimizer='adam', lr=1e-4, bidirectional=False, attention=False):
    print("Seed {}:".format(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size[0], shuffle=True, collate_fn=pad_collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size[1], shuffle=False, collate_fn=pad_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size[2], shuffle=False, collate_fn=pad_collate_fn)

    model = RNNModel(model_name, embedding_layer, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, attention=attention)

    criterion = nn.BCEWithLogitsLoss()
    if optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=1e-4)
    elif optimizer == 'sgd':
        optimizer = SGD(model.parameters(), lr=1e-4, momentum=0.9)
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(model.parameters(), lr=1e-4, alpha=0.99, eps=1e-08)
    elif optimizer == 'adagrad':
        optimizer = Adagrad(model.parameters(), lr=1e-4, lr_decay=0, weight_decay=0, initial_accumulator_value=0)

    for epoch in range(epochs):
        train(model, train_dataloader, optimizer, criterion, clip=0.25)
        avg_loss, accuracy, f1, conf_matrix = evaluate(model, valid_dataloader, criterion)
        print("Epoch {}: valid accuracy = {}".format(epoch, accuracy))

    avg_loss, accuracy, f1, conf_matrix = evaluate(model, test_dataloader, criterion)
    print("Avg_loss: ", avg_loss)
    print("f1:", f1) 
    print("Confusion matrix:\n", conf_matrix)
    print("Test accuracy = {}".format(accuracy))

if __name__ == '__main__':
    epochs, batch_size = 5, [32, 32, 32]
    #for model_name in ['vanilla', 'gru', 'lstm']:
    #    print(model_name)
    #    for hidden_size in [40, 150, 400]:
    #        print(hidden_size)
    #        main_rnn(7052020, model_name, epochs, batch_size, hidden_size, 2, 0.0, False)
    #    for num_layers in [2, 4, 7]:
    #        print(num_layers)
    #        main_rnn(7052020, model_name, epochs, batch_size, 150, num_layers, 0.0, False)
    #    for dropout in [0.0, 0.25, 0.5]:
    #        print(dropout)
    #        main_rnn(7052020, model_name, epochs, batch_size, 150, 2, dropout, False)
    #    for bidirectional in [True, False]:
    #        print(bidirectional)
    #        main_rnn(7052020, model_name, epochs, batch_size, 150, 2, 0.0, bidirectional)

    for i in range(1, 6):
        main_rnn(i, 'lstm', epochs, batch_size, 150, 4, 0.25, optimizer='adam', lr=1e-4, bidirectional=True, attention=True)

    #for lr in [5 * 1e-3, 2 * 1e-4, 5 * 1e-4]:
    #    print("lr:", lr)
    #    main_rnn(7052020, 'lstm', epochs, batch_size, 150, 4, 0.25, lr, True)

    #for optimizer in ['sgd', 'rmsprop', 'adagrad']:
    #    main_rnn(7052020, 'lstm', epochs, batch_size, 150, 4, 0.25, optimizer, 1e-4, True)

