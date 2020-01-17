import torch.nn as nn
import torch
import random


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [src_len, batch_size]
        # embedding: [src_len, batch_size, emb_dim]
        embedding = self.dropout(self.embedding(src))

        # outputs: [src_len, batch_size, hid_dim*n_directions]
        # hidden: [n_layers*n_directions, batch_size, hid_dim]
        # cell: [n_layers*n_directions, batch_size, hid_dim]
        outputs, (hidden, cell) = self.rnn(embedding)   # outputs are always from the top hidden layer

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input: [batch_size]
        # hidden: [n_layers*n_directions, batch_size, hid_dim]
        # cell: [n_layers*n_directions, batch_size, hid_dim]

        # n_directions in the decoder will both always be 1, therefore:
        # hidden: [n_layers, batch_size, hid_dim]
        # context: [n_layers, batch_size, hid_dim]

        input = input.unsqueeze(0)  # 1, batch_size
        embedding = self.dropout(self.embedding(input))  # [1, batchsize, emb_dim]

        # output: [seq_len, batch_size, hid_dim*n_directions]
        # hidden: [n_layers*n_directions, batch_size, hid_dim]
        # cell: [n_layers*n_directions, batch_size, hid_dim]

        # seq_len and n_directions in the decoder will both always be 1, therefore:
        # output: [1, batch_size, hid_dim]
        # hidden: [n_layers, batch_size, hid_dim]
        # cell: [n_layers, batch_size, hid_dim]
        output, (hidden, cell) = self.rnn(embedding, (hidden, cell))

        prediction = self.fc_out(output.squeeze(0))  # [batch_size, output_dim]

        return prediction, hidden, cell


def init_weights(model):
    for name, param in model.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, teacher_forcing_ratio=0.5):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio
        assert encoder.hid_dim == decoder.hid_dim, "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, "Encoder and decoder must have equal number of layers!"
        init_weights(self)

    def forward(self, src, trg):
        # src: [src_len, batch_size]
        # trg: [trg_len, batch_size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # eg, if teacher_forcing_ratio is 0.75, we use groud-truth inputs 75% of the time

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(size=(trg_len, batch_size, trg_vocab_size), device=self.device)

        # last hidden state of the encoder is used as the initial hidden state
        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < (self.teacher_forcing_ratio if self.training else 0)

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use acutal next tokens as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs
