import numpy as np

import pdb

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


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
        _, (last_hidden, last_cell) = self.encoder(input_emb)   #(h_0 = _0_, c_0 = _0_)

        if training:
            # full teacher forcing
            output_emb = self.output_embedding(output_seq)
            output_emb = self.dropout(output_emb)

            hidden_states, (last_hidden, last_cell) = self.decoder(output_emb[:-1], (last_hidden, last_cell))
            logits_seq = F.linear(hidden_states, self.output_embedding.weight)
            return logits_seq
        else:
            # decode
            logits_seq = []
            outputs = []

            batch_size = input_seq.shape[1]
            #last_output_seq = torch.LongTensor([sos_tok]).to(device).repeat(batch_size).view(1, batch_size)
            # constantly moving tensors between cpu and cuda is a bad idea which takes a lot of cpu utilization
            last_output_seq = torch.zeros(1, batch_size, dtype=torch.long, device=device).fill_(sos_tok)
            last_output_emb = self.output_embedding(last_output_seq)

            for t in range(0, max_length):
                # last_hidden and last_cell comes from encoder
                hidden_state, (last_hidden, last_cell) = self.decoder(last_output_emb, (last_hidden, last_cell))
                logits = F.linear(hidden_state, self.output_embedding.weight)

                logits_seq.append(logits)
                
                last_output = logits.argmax(2)
                outputs.append(last_output)

                last_output_emb = self.output_embedding(last_output)

            logits_seq = torch.cat(logits_seq, dim=0)
            outputs = np.array([i.tolist()[0] for i in outputs])
            return outputs, logits_seq 

    @staticmethod
    def loss(logits_seq, output_seq, criterion):
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
        _, (last_hidden, last_cell) = self.encoder(input_emb)   # (h_0 = _0_, c_0 = _0_)

        if training:
            # full teacher forcing
            output_emb = self.output_embedding(output_seq)  # [seq_len, batch_size, emb_dim]

            hidden_states, (last_hidden, last_cell) = self.decoder(output_emb[:-1], (last_hidden, last_cell))
            logits_seq = self.linear(hidden_states)
            return logits_seq
        else:
            # decode
            logits_seq = []
            outputs = []

            # input_seq: [seq_len, batch_size]
            batch_size = input_seq.shape[1]
            #last_output_seq = torch.LongTensor([sos_tok]).to(device).repeat(batch_size).view(1, batch_size)
            # constantly moving tensors between cpu and cuda is a bad idea which takes a lot of cpu utilization
            last_output_seq = torch.zeros(1, batch_size, dtype=torch.long, device=device).fill_(sos_tok)
            last_output_emb = self.output_embedding(last_output_seq)    # [1, batch_size, emb_dim]

            for t in range(0, max_length):
                # last_hidden and last_cell comes from encoder
                hidden_state, (last_hidden, last_cell) = self.decoder(last_output_emb, (last_hidden, last_cell))  # [1, batch_size, hid_dim]
                logits = self.linear(hidden_state)  # [1, batch_size, output_dim]
                logits_seq.append(logits)
                
                last_output = logits.argmax(2)  # [1, batch_size]
                outputs.append(last_output)  # [seq_len, batch_size]

                last_output_emb = self.output_embedding(last_output)

            logits_seq = torch.cat(logits_seq, dim=0)   # [seq_len, batch_size]
            outputs = np.array([i.tolist()[0] for i in outputs])    # [seq_len, 1, batch_size] --> [seq_len, batch_size]
            return outputs, logits_seq 

    @staticmethod
    def loss(logits_seq, output_seq, criterion):
        # remove <sos> and shift output seq by 1
        shape = output_seq.shape    # [seq_len, batch_size]
        chain_length = (shape[0] - 1) * shape[1]        # (seq_len - 1) * batch_size
        chained_output_seq = output_seq[1:].permute(1, 0).reshape(chain_length)

        shape = logits_seq.shape    # [seq_len-1, batch_size, output_size]
        chained_logits_seq = logits_seq.permute(1, 0, 2).reshape(chain_length, shape[2])

        return criterion(chained_logits_seq, chained_output_seq)


class BERT2LSTM(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size):
        super(BERT2LSTM, self).__init__()
        from transformers import BertModel

        # self.input_embedding = nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=256)
        self.output_embedding = nn.Embedding(num_embeddings=output_vocab_size, embedding_dim=256)

        self.encoder = BertModel.from_pretrained('bert-base-german-cased')

        self.decoder = nn.LSTM(input_size=256, hidden_size=768, num_layers=1)
        self.linear = nn.Linear(in_features=768, out_features=output_vocab_size, bias=True)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, input_seq, input_lens, output_seq, training=True, sos_tok=0, max_length=0, device=None):

        # _, (hidden, cell) = self.encoder(input_emb)  # (h_0 = _0_, c_0 = _0_)
        seq_len, batch_size = input_seq.shape
        input_seq = input_seq.transpose(0, 1)  # [seq_len, batch_size] --> [batch_size, seq_len]
        mask_ids = pad_sequence([torch.ones(l.item(), dtype=torch.long, device=device) for l in input_lens], batch_first=True)

        hidden = self.encoder(input_seq, attention_mask=mask_ids)[0]    # [batch_size, seq_len, 768]
        hidden = hidden[:, 0]   # [batch_size, 768] we only need the representation of the first token to represent the entire sequence
        hidden = hidden.unsqueeze(0)    # [1, batch_size, 768]  to match the input shape of decoder
        # cell = torch.randn(1, batch_size, 768, dtype=torch.float, device=device)
        last_hidden = hidden.contiguous()
        last_cell = last_hidden.detach().clone()

        if training:
            # full teacher forcing
            output_emb = self.output_embedding(output_seq)  # [seq_len, batch_size, emb_dim]

            hidden_states, (last_hidden, last_cell) = self.decoder(output_emb[:-1], (last_hidden, last_cell))
            logits_seq = self.linear(hidden_states)
            return logits_seq
        else:
            # decode
            logits_seq = []
            outputs = []
            # last_output_seq = torch.LongTensor([sos_tok]).to(device).repeat(batch_size).view(1, batch_size)
            # constantly moving tensors between cpu and cuda is a bad idea which takes a lot of cpu utilization
            last_output_seq = torch.zeros(1, batch_size, dtype=torch.long, device=device).fill_(sos_tok)
            last_output_emb = self.output_embedding(last_output_seq)  # [1, batch_size, emb_dim]

            for t in range(0, max_length):
                # last_hidden and last_cell comes from encoder
                hidden_state, (last_hidden, last_cell) = self.decoder(last_output_emb, (last_hidden, last_cell))  # [1, batch_size, hid_dim]
                logits = self.linear(hidden_state)  # [1, batch_size, output_dim]
                logits_seq.append(logits)

                last_output = logits.argmax(2)  # [1, batch_size]
                outputs.append(last_output)  # [seq_len, batch_size]

                last_output_emb = self.output_embedding(last_output)

            logits_seq = torch.cat(logits_seq, dim=0)  # [seq_len, batch_size]
            outputs = np.array([i.tolist()[0] for i in outputs])  # [seq_len, 1, batch_size] --> [seq_len, batch_size]
            return outputs, logits_seq

    @staticmethod
    def loss(logits_seq, output_seq, criterion):
        # remove <sos> and shift output seq by 1
        shape = output_seq.shape  # [seq_len, batch_size]
        chain_length = (shape[0] - 1) * shape[1]  # (seq_len - 1) * batch_size
        chained_output_seq = output_seq[1:].permute(1, 0).reshape(chain_length)

        shape = logits_seq.shape  # [seq_len-1, batch_size, output_size]
        chained_logits_seq = logits_seq.permute(1, 0, 2).reshape(chain_length, shape[2])

        return criterion(chained_logits_seq, chained_output_seq)