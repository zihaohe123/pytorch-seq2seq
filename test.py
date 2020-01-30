import os
import tqdm
import pickle
import subprocess

import torch
import numpy as np

import datasets

print('Setting CUDA_VISIBLE_DEVICES...')
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Done.')

print('Loading data...')
dataset = 'iwslt2014'    # replace with argparse
func = getattr(datasets, dataset)
(source, target), (train_iterator, val_iterator, test_iterator) = func(device)
print('Done.')

print('Creating model...')
model = torch.load('experiments/test/model.pkl')
model.to(device)
print('Done.')

target_sos_idx = target.vocab.stoi['<SOS>']
output_file = open('experiments/test/predictions', 'wt')

# validation
model.eval()
test_loss = 0.
for i, batch in enumerate(tqdm.tqdm(test_iterator)):
    with torch.no_grad():
        batch_input_seq, batch_input_len = batch.src
        outputs, logits_seq = model(batch_input_seq, output_seq=None, training=False, 
                sos_tok=target_sos_idx, max_length=100, device=device)

    # convert to tokens
    itos = lambda x: target.vocab.itos[x]
    outputs = np.vectorize(itos)(outputs.T)

    # write to file
    for i in outputs:
        output_file.write(' '.join(i) + '\n')
output_file.close()

# post processing pipeline
subprocess.run('./detokenizer.sh %s %s' % ('./experiments/test/predictions', 'en'), shell=True)
subprocess.run('cat ./experiments/test/predictions.detok | \
        sacrebleu -b %s' % './data/iwslt2014/references.de-en.en', shell=True)
