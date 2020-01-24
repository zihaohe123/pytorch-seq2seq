import os
import tqdm
import numpy as np
import random 

import torch
from torch import nn, optim

import datasets
import model2lstm

print('Setting CUDA_VISIBLE_DEVICES...')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Done.')

print('Loading data...')
dataset = 'iwslt'    # replace with argparse
func = getattr(datasets, dataset)
(source, target), (train_iterator, val_iterator), (tokenizer, detokenizer) = func(device)
print('Done.')

print('Creating model...')
name = 'Seq2SeqWithMainstreamImprovements'    # replace with argparse
class_ = getattr(model2lstm, name)
model = class_(input_vocab_size=len(source.vocab), output_vocab_size=len(target.vocab))
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
