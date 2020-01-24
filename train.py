import os
import tqdm
import numpy as np
import random 

import torch
from torch import nn, optim

import datasets

print('Setting CUDA_VISIBLE_DEVICES...')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Done.')

print('Loading data...')
dataset = 'multi30k'    # replace with argparse
func = getattr(datasets, dataset)
(source, target), (train_iterator, val_iterator), (tokenizer, detokenizer) = func(device)
print('Done.')

print('Creating model...')
class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        
        self.input_embedding = nn.Embedding(num_embeddings=len(source.vocab), embedding_dim=256)
        self.output_embedding = nn.Embedding(num_embeddings=len(target.vocab), embedding_dim=256)

        self.encoder = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, dropout=0.5)

        self.decoder = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, dropout=0.5)
        self.linear = nn.Linear(in_features=256, out_features=len(target.vocab), bias=True)

    def forward(self, input_seq, output_seq, training=True, sos_tok=0, max_length=0, device=None):
        input_emb = self.input_embedding(input_seq)
        output_emb = self.output_embedding(output_seq)

        hidden_states, (last_hidden, last_cell) = self.encoder(input_emb)   #(h_0 = _0_, c_0 = _0_)

        if training:
            # full teacher forcing
            hidden_states, (last_hidden, last_cell) = self.decoder(output_emb[:-1], (last_hidden, last_cell))
            logits_seq = self.linear(hidden_states)
            return logits_seq
        else:
            # decode
            logits_seq = []
            outputs = []

            batch_size = output_seq.shape[1]
            last_output_seq = torch.LongTensor([sos_tok]).to(device).repeat(batch_size).view(1, batch_size)
            last_output_emb = self.output_embedding(last_output_seq)

            for t in range(0, max_length):
                # last_hidden and last_cell comes from encoder
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
        chain_length = (shape[0] - 1) * shape[1]        # (seq_len - 1) * batch_size
        chained_output_seq = output_seq[1:].permute(1,0).reshape(chain_length)

        shape = logits_seq.shape
        chained_logits_seq = logits_seq.permute(1,0,2).reshape(chain_length, shape[2])

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
        batch_output_seq, batch_output_len = batch.trg

        logits_seq = model(batch_input_seq, batch_output_seq, training=True)
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
            outputs, logits_seq = model(batch_input_seq, batch_output_seq, training=False, 
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
