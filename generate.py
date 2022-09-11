import sys

import torch
import torch.nn as nn

from train import evaluate
from train import char_to_idx, idx_to_char

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class TextRNN(nn.Module):

    def __init__(self, input_size, hidden_size, embedding_size, n_layers=1):
        super(TextRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(self.input_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.n_layers)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, x, hidden):
        x = self.encoder(x).squeeze(2)
        out, (ht1, ct1) = self.lstm(x, hidden)
        out = self.dropout(out)
        x = self.fc(out)
        return x, (ht1, ct1)

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device))


if __name__ == '__main__':
    n = len(sys.argv)
    start_text = ' '
    prediction_len = 1000

    if n == 5:
        param_name_1 = sys.argv[1]
        param_value_1 = sys.argv[2]
        param_name_2 = sys.argv[3]
        param_value_2 = sys.argv[4]

        if param_name_1 == '--model':
            path_model = param_value_1

        elif param_name_2 == '--model':
            path_model = param_value_2

        if param_name_1 == '--length':
            prediction_len = param_value_1

        elif param_name_2 == '--length':
            prediction_len = param_value_2

    elif n == 7:
        param_name_1 = sys.argv[1]
        param_value_1 = sys.argv[2]
        param_name_2 = sys.argv[3]
        param_value_2 = sys.argv[4]
        param_name_3 = sys.argv[5]
        param_value_3 = sys.argv[6]

        if param_name_1 == '--model':
            path_model = param_value_1

        elif param_name_2 == '--model':
            path_model = param_value_2

        elif param_name_3 == '--model':
            path_model = param_value_3

        if param_name_1 == '--length':
            prediction_len = param_value_1

        elif param_name_2 == '--length':
            prediction_len = param_value_2

        elif param_name_3 == '--length':
            prediction_len = param_value_3

        if param_name_1 == '--prefix':
            start_text = param_value_1

        elif param_name_2 == '--prefix':
            start_text = param_value_2

        elif param_name_3 == '--prefix':
            start_text = param_value_3

    model = TextRNN(input_size=len(idx_to_char), hidden_size=64, embedding_size=64, n_layers=2)
    model.load_state_dict(torch.load(path_model))

    model.eval()
    print(evaluate(model,
                   char_to_idx,
                   idx_to_char,
                   prediction_len,
                   start_text,
                   temp=0.1))