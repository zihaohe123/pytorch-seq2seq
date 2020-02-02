import os
import tqdm
import pickle
import subprocess

import torch
import numpy as np

import datasets
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Neural Machine Translation with Seq2Seq.')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_path', type=str, default='data/iwslt2014/train.de-en.bpe')
    parser.add_argument('--dev_path', type=str, default='data/iwslt2014/dev.de-en.bpe')
    parser.add_argument('--test_path', type=str, default='data/iwslt2014/test.de-en.bpe')
    args = parser.parse_args()

    print('Setting CUDA_VISIBLE_DEVICES...')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Done.')

    print('Loading config')
    config = torch.load(os.path.join('experiments/test', 'config.pt'))
    dataset = config.dataset

    print('Loading data...')
    func = getattr(datasets, dataset)
    (source, target), (train_iterator, val_iterator, test_iterator) = func(device, args.batch_size, args.train_path,
                                                                           args.dev_path, args.test_path)
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

    print('Detokenizing...')
    subprocess.run('./detokenizer.sh %s %s 2> /dev/null' % ('./experiments/test/predictions', 'en'), shell=True)
    print('Done.')

    print('Calculating BLEU...')
    subprocess.run('cat ./experiments/test/predictions.detok | sacrebleu %s' % './data/iwslt2014/references.de-en.en', shell=True)
    print('Done.')
