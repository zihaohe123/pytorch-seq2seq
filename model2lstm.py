import numpy as np

import pdb
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Decoder_attention(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,n_layers=1, dropout=0.2):
        super(Decoder_attention, self).__init__()
        self.embed_size = embed_size
        self.hidden_size =hidden_size
        self.output_size =output_size#target voc size
        self.n_layers   = n_layers

        self.embed = nn.Embedding(output_size, embed_size) #一样的
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(input_size=256*2, hidden_size=256, num_layers=n_layers, dropout=0.5)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden,last_cell,encoder_outputs):
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
        embedded = self.dropout(embedded)
        # focuse on the shape of last_hidden
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        lstm_input = torch.cat([embedded, context], 2)
        output, (hidden,cell) = self.lstm(lstm_input, (last_hidden,last_cell))
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)
        return output, (hidden, cell),attn_weights

class Attention(nn.Module):
    # 这边只需要输入一个hidden size就可以了，还有encoder还有hidden
    # hidden的这个可能输入对不上。
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # from decoder to input
        self.hidden_size = hidden_size   #256
        self.attn = nn.Linear(self.hidden_size*2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)#初始化一些东西哦。

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        # hidden shape is 32 256
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        # transpose for easily computation
        # first sentence   2 335 5 654  654 98
        # second sen
        # tence  2 65  65 5 6 654 68
        attn_energies = self.score(h, encoder_outputs)

        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*H]->[B*T*2H]->[B*T*H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        # 实际上不是一个一个算的，而是直接弄成了一个大的去计算。
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]

class Seq2Seq(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size):
        super(Seq2Seq, self).__init__()
        self.input_embedding = nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=256)
        self.output_embedding = nn.Embedding(num_embeddings=output_vocab_size, embedding_dim=256)

        self.encoder = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, dropout=0.5)

        self.decoder = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, dropout=0.5)
        self.linear = nn.Linear(in_features=256, out_features=output_vocab_size, bias=True)

    def forward(self, input_seq, input_len, output_seq, output_len, training=True, sos_tok=0, max_length=0, device='cpu'):
        input_emb = self.input_embedding(input_seq)
        packed_input = pack_padded_sequence(input_emb, input_len.cpu().numpy(),enforce_sorted=False)
        _, (last_hidden, last_cell) = self.encoder(packed_input)   # (h_0 = _0_, c_0 = _0_)

        if training:
            # full teacher forcing
            output_emb = self.output_embedding(output_seq)  # [seq_len, batch_size, emb_dim]
            packed_output = pack_padded_sequence(output_emb[:-1], output_len.cpu().numpy() - 1,
                                                 enforce_sorted=False)
            # hidden_states_normal, (last_hidden, last_cell) = self.decoder(output_emb[:-1], (last_hidden, last_cell))
            hidden_states, (last_hidden, last_cell) = self.decoder(packed_output, (last_hidden, last_cell))
            # hidden_states, seq_sum * 256
            unpacked_hidden_states_output, input_sizes = pad_packed_sequence(hidden_states,
                                                                             padding_value=int(1))
            logits_seq = self.linear(unpacked_hidden_states_output)

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

class Seq2Seq_attention(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size):
        super(Seq2Seq_attention, self).__init__()

        self.input_embedding = nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=256)
        self.encoder = nn.LSTM(input_size=256, hidden_size=256, num_layers=1, dropout=0.5)
        # different from previous
        self.decoder_attention = Decoder_attention(embed_size=256, hidden_size=256, output_size = output_vocab_size, n_layers=1, dropout=0.5)
        self.linear = nn.Linear(in_features=256, out_features=output_vocab_size, bias=True)
        self.vocab_size = output_vocab_size

    def forward(self, input_seq, input_len, output_seq, output_len, training=True, sos_tok=0, max_length=0, device='cpu'):
        input_emb = self.input_embedding(input_seq)
        batch_size = input_seq.size(1)
        max_len = output_seq.size(0)#this one is only for the training, different from max_length
        vocab_size = self.vocab_size
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()
        # add padding
        encoder_hidden_states, (last_hidden, last_cell) = self.encoder(input_emb)   # (h_0 = _0_, c_0 = _0_)
        output = Variable(output_seq.data[0, :])  # BOS the first one 222222
        if training:
            for t in range(1, max_len):
                # 输入好像是有问题。
                output, (last_hidden, last_cell),attn_weights = self.decoder_attention(output, last_hidden, last_cell,encoder_hidden_states)
                # shape of output is B V
                outputs[t] = output
                top1 = output.data.max(1)[1] #select the top1 word
                output = Variable(top1).cuda()
            # same shape outputs and hidden_states 49 32 10215
            # do not use the first one, it is <BOS>
            logits_seq = outputs[1:]
            return logits_seq

        else:
            result = []
            for t in range(1, max_len):
                output, (last_hidden, last_cell),attn_weights = self.decoder_attention(output, last_hidden, last_cell,encoder_hidden_states)
                # shape of output is B V
                outputs[t] = output
                top1 = output.data.max(1)[1] #select the top1 word
                output = Variable(top1).cuda()
                # shape of output is 32
                result.append(output.unsqueeze(0))
            # same shape outputs and hidden_states outputs.shape = S B
            # do not use the first one, it is <BOS>
            logits_seq = outputs[1:]
            result = np.array([i.tolist()[0] for i in result])    # [seq_len, 1, batch_size] --> [seq_len, batch_size]
            return result, logits_seq

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

    def forward(self, input_seq, input_lens, output_seq, output_len, training=True, sos_tok=0, max_length=0, device='cpu'):

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
