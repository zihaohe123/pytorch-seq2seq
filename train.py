import os
import tqdm
import numpy as np
import random 

import pdb

import torch
from torch import nn, optim

from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

print('Setting CUDA_VISIBLE_DEVICES...')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Done.')

print('Loading data with torchtext...')
source = Field(
        sequential=True,
        use_vocab=True,
        init_token='<SOS>',
        eos_token='<EOS>',
        lower=True,
        tokenize=str.split,
        include_lengths=True,
        batch_first=False,
        pad_token='<pad>',
        unk_token='<unk>'
    )

target = Field(
        sequential=True,
        use_vocab=True,
        init_token='<SOS>',
        eos_token='<EOS>',
        lower=True,
        tokenize=str.split,
        include_lengths=True,
        batch_first=False,
        pad_token='<pad>',
        unk_token='<unk>'
    )
train_dataset, test_dataset, val_dataset = Multi30k.splits(exts=('.de','.en'), fields=(source, target))
print('Done.')
print('Building vocabulary...')
target.build_vocab(train_dataset, min_freq=20)
source.build_vocab(train_dataset, min_freq=20)
print('Done.')

train_iterator, val_iterator = BucketIterator.splits(
        datasets=(train_dataset, val_dataset),
        batch_size=32,
        device=device
    )

print('Creating model...')
class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.input_embedding = nn.Embedding(num_embeddings=len(source.vocab), embedding_dim=256)
        self.output_embedding = nn.Embedding(num_embeddings=len(target.vocab), embedding_dim=256)

        self.encoder = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, dropout=0.5)

        self.decoder = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, dropout=0.5)
        self.linear = nn.Linear(in_features=256, out_features=len(target.vocab), bias=True)

    def forward(self, input_seq, output_seq, input_seq_length,output_seq_length,training=True, sos_tok=0, max_length=0, device=None):
        #26 32 256 the shape of this input_emb
        input_emb  = self.input_embedding(input_seq)
        output_emb = self.output_embedding(output_seq)

        packed_input = pack_padded_sequence(input_emb, input_seq_length.cpu().numpy(),enforce_sorted=False)
        hidden_states_1, (last_hidden, last_cell) = self.encoder(packed_input)   #(h_0 = _0_, c_0 = _0_)
        if training:
            packed_output = pack_padded_sequence(output_emb[:-1], output_seq_length.cpu().numpy()-1, enforce_sorted=False)
            # hidden_states_normal, (last_hidden, last_cell) = self.decoder(output_emb[:-1], (last_hidden, last_cell))
            # I think this place is right, since it is training process, so that we should give the target a fixed
            # sentence length, so after unpacked, it will back to normal
            hidden_states, (last_hidden, last_cell) = self.decoder(packed_output, (last_hidden, last_cell))
            #hidden_states, seq_sum * 256
            unpacked_hidden_states_output, input_sizes = pad_packed_sequence(hidden_states, padding_value=target_pad_idx)
            # in this way, it will have an unpacked output, where pad is target_pad_idx and this target_pad_idx will
            # be removed in creteria process.
            logits_seq = self.linear(unpacked_hidden_states_output)

            return logits_seq
        else:
            # decode
            # there is no need to change this validation part, since it is going to predict each word, and the
            # out_put_hidden_state's shape is B*256, so cannot use the unpack function. Since unpack function
            # require Sequence_sum * B * 256 shape.
            logits_seq = []
            outputs = []

            batch_size = output_seq.shape[1]
            last_output_seq = torch.LongTensor([sos_tok]).to(device).repeat(batch_size).view(1, batch_size)
            last_output_emb = self.output_embedding(last_output_seq)

            for t in range(0, max_length):
                # last_hidden and last_cell comes from encoder(packed embedding)
                hidden_state, (last_hidden, last_cell) = self.decoder(last_output_emb, (last_hidden, last_cell))
                logits = self.linear(hidden_state)
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
        chain_length = (shape[0] - 1) * shape[1]  # (seq_len - 1) * batch_size
        chained_output_seq = output_seq[1:].permute(1, 0).reshape(chain_length)

        shape = logits_seq.shape
        chained_logits_seq = logits_seq.permute(1, 0, 2).reshape(chain_length, shape[2])

        return criterion(chained_logits_seq, chained_output_seq)

model = Seq2Seq()
model.to(device)
print('Done.')

print('Training:')
optimizer = optim.Adam(model.parameters())
target_pad_idx = target.vocab.stoi[target.pad_token]
target_sos_idx = target.vocab.stoi['<SOS>']
criterion = nn.CrossEntropyLoss(ignore_index=target_pad_idx, reduction='sum')

patience = 1
best_val_loss = float('inf')

for epoch in range(1, 10+1):
    model.train()
    acc_loss, total_toks, total_seqs = 0., 0, 0
    tqdm_iterator = tqdm.tqdm(train_iterator)
    for batch in tqdm_iterator:
        batch_input_seq, batch_input_len = batch.src
        #shape = 32 for the batch input_len, inside is the length of each sentence
        #input_seq is tokenized(###,###,###) start is 2, end is 1
        batch_output_seq, batch_output_len = batch.trg

        logits_seq = model(batch_input_seq, batch_output_seq, batch_input_len,batch_output_len,training=True)
        loss = model.loss(logits_seq, batch_output_seq, criterion)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_toks += int(batch_output_len.sum())
        total_seqs += batch_output_len.shape[0]
        acc_loss += loss.item()
        avg_tok_loss = acc_loss / total_toks
        avg_seq_loss = acc_loss / total_seqs
        tqdm_iterator.set_description('avg_tok_loss=%f, avg_seq_loss=%f' % (avg_tok_loss, avg_seq_loss))


    # validation
    model.eval()
    val_loss = 0.
    total_toks, total_seqs = 0, 0
    random_batch = random.randint(1, len(val_iterator))
    for i, batch in enumerate(tqdm.tqdm(val_iterator)):
        with torch.no_grad():
            batch_input_seq, batch_input_len = batch.src
            batch_output_seq, batch_output_len = batch.trg
            outputs, logits_seq = model(batch_input_seq, batch_output_seq, batch_input_len,batch_output_len,training=False,
                    sos_tok=target_sos_idx, max_length=batch_output_seq.shape[0]-1, device=device)

            loss = model.loss(logits_seq, batch_output_seq, criterion)
            val_loss += loss.item()

            total_toks += int(batch_output_len.sum())
            total_seqs += batch_output_len.shape[0]

            # generate the sequences
            if i == random_batch:
                saved_outputs = outputs
                saved_batch_output_seq = np.array(batch_output_seq.tolist())

    # print a random batch
    itos = lambda x: target.vocab.itos[x]
    print('Randomly sampling some output...')
    print('=== Ground truth ===')
    print(np.vectorize(itos)(saved_batch_output_seq.T)[0:5])
    print('=== Predictions ===')
    print(np.vectorize(itos)(saved_outputs.T)[0:5])
    print('Done.')

    print('Total validation loss: %f' % val_loss)
    print('Average token loss: %f' % (val_loss / total_toks))
    print('Average sequence loss: %f' % (val_loss / total_seqs))
