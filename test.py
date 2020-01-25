import os
import tqdm
import pickle

import torch
import numpy as np

import datasets

print('Setting CUDA_VISIBLE_DEVICES...')
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Done.')

print('Loading data...')
dataset = 'multi30k'    # replace with argparse
func = getattr(datasets, dataset)
(source, target), (train_iterator, val_iterator) = func(device)
print('Done.')

print('Creating model...')
model = torch.load('experiments/test/model.pkl')
model.to(device)
print('Done.')

print('Loading fields...')
source = pickle.load(open('experiments/test/source.pkl', 'rb'))
target = pickle.load(open('experiments/test/target.pkl', 'rb'))
print('Done.')

target_sos_idx = target.vocab.stoi['<SOS>']
output_file = open('experiments/test/predictions', 'wt')

# validation
model.eval()
test_loss = 0.
for i, batch in enumerate(tqdm.tqdm(val_iterator)):
    with torch.no_grad():
        batch_input_seq, batch_input_len = batch.src
        outputs, logits_seq = model(batch_input_seq, output_seq=None, training=False, 
                sos_tok=target_sos_idx, max_length=100, device=device)

    # convert to tokens
    itos = lambda x: target.vocab.itos[x]
    outputs = np.vectorize(itos)(outputs.T)[0:5]

    # write to file
    for i in outputs:
        output_file.write(' '.join(i) + '\n')
