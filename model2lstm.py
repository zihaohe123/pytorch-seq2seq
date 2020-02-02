import numpy as np

import pdb

import torch
from torch import nn
from torch.nn import functional as F


class Seq2SeqWithMainstreamImprovements(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size):
        super(Seq2SeqWithMainstreamImprovements, self).__init__()
        
        self.input_embedding = nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=512)
        self.output_embedding = nn.Embedding(num_embeddings=output_vocab_size, embedding_dim=1024)

        # add dropout to lstm
        self.encoder = nn.LSTM(input_size=512, hidden_size=1024, num_layers=3, dropout=0.5)

        self.decoder = nn.LSTM(input_size=1024, hidden_size=1024, num_layers=3, dropout=0.5)
        #self.linear = nn.Linear(in_features=256, out_features=output_vocab_size, bias=True)
        # tie embedding weights instead

        self.dropout = nn.Dropout(0.5)

    def forward(self, input_seq, output_seq, training=True, sos_tok=0, max_length=0, device=None):
        input_emb = self.input_embedding(input_seq)

        # add dropout
        input_emb = self.dropout(input_emb)
        _, (hidden, cell) = self.encoder(input_emb)   #(h_0 = _0_, c_0 = _0_)

        if training:
            # full teacher forcing
            output_emb = self.output_embedding(output_seq)
            output_emb = self.dropout(output_emb)

            hidden_states, (last_hidden, last_cell) = self.decoder(output_emb[:-1], (hidden, cell))
            logits_seq = F.linear(hidden_states, self.output_embedding.weight)
            return logits_seq
        else:
            # decode
            logits_seq = []
            outputs = []

            batch_size = input_seq.shape[1]
            last_output_seq = torch.LongTensor([sos_tok]).to(device).repeat(batch_size).view(1, batch_size)
            last_output_emb = self.output_embedding(last_output_seq)

            for t in range(0, max_length):
                # last_hidden and last_cell comes from encoder
                hidden_state, (last_hidden, last_cell) = self.decoder(last_output_emb, (hidden, cell))
                logits = F.linear(hidden_state, self.output_embedding.weight)

                logits_seq.append(logits)
                
                last_output = logits.argmax(2)
                outputs.append(last_output)

                last_output_emb = self.output_embedding(last_output)

            logits_seq = torch.cat(logits_seq, dim=0)
            outputs = np.array([ i.tolist()[0] for i in outputs ])
            return outputs, logits_seq 

    def loss(self, logits_seq, output_seq, criterion):
        # remove <sos> and shift output seq by 1
        shape = output_seq.shape
        chain_length = (shape[0] - 1) * shape[1]        # (seq_len - 1) * batch_size
        chained_output_seq = output_seq[1:].permute(1, 0).reshape(chain_length)

        shape = logits_seq.shape
        chained_logits_seq = logits_seq.permute(1, 0, 2).reshape(chain_length, shape[2])

        return criterion(chained_logits_seq, chained_output_seq)


class Seq2Seq(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size):
        super(Seq2Seq, self).__init__()
        
        self.input_embedding = nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=256)
        self.output_embedding = nn.Embedding(num_embeddings=output_vocab_size, embedding_dim=256)

        self.encoder = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, dropout=0.5)

        self.decoder = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, dropout=0.5)
        self.linear = nn.Linear(in_features=256, out_features=output_vocab_size, bias=True)

    def forward(self, input_seq, output_seq, training=True, sos_tok=0, max_length=0, device=None):
        input_emb = self.input_embedding(input_seq)
        _, (hidden, cell) = self.encoder(input_emb)   # (h_0 = _0_, c_0 = _0_)

        if training:
            # full teacher forcing
            output_emb = self.output_embedding(output_seq)

            hidden_states, (last_hidden, last_cell) = self.decoder(output_emb[:-1], (hidden, cell))
            logits_seq = self.linear(hidden_states)
            return logits_seq
        else:
            # decode
            logits_seq = []
            outputs = []

            batch_size = input_seq.shape[1]
            last_output_seq = torch.LongTensor([sos_tok]).to(device).repeat(batch_size).view(1, batch_size)
            last_output_emb = self.output_embedding(last_output_seq)

            for t in range(0, max_length):
                # last_hidden and last_cell comes from encoder
                hidden_state, (last_hidden, last_cell) = self.decoder(last_output_emb, (hidden, cell))
                logits = self.linear(hidden_state)
                logits_seq.append(logits)
                
                last_output = logits.argmax(2)
                outputs.append(last_output)

                last_output_emb = self.output_embedding(last_output)

            logits_seq = torch.cat(logits_seq, dim=0)
            outputs = np.array([i.tolist()[0] for i in outputs])
            return outputs, logits_seq 

    def loss(self, logits_seq, output_seq, criterion):
        # remove <sos> and shift output seq by 1
        shape = output_seq.shape
        chain_length = (shape[0] - 1) * shape[1]        # (seq_len - 1) * batch_size
        chained_output_seq = output_seq[1:].permute(1,0).reshape(chain_length)

        shape = logits_seq.shape
        chained_logits_seq = logits_seq.permute(1, 0, 2).reshape(chain_length, shape[2])

        return criterion(chained_logits_seq, chained_output_seq)
