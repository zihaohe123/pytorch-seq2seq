from torchtext.datasets import Multi30k, TranslationDataset
from torchtext.data import Field, BucketIterator
import spacy, os, random

def train_data_loader(data_path, src_lang, trg_lang, n_samples=0, batchsize=128, device='cpu'):
    # sampling data
    print('Sampling data...')
    src_data = list(open(os.path.join(data_path, 'europarl-v7.de-en.' + src_lang)))
    trg_data = list(open(os.path.join(data_path, 'europarl-v7.de-en.' + trg_lang)))
    data_len = len(src_data)

    data = list(zip(src_data, trg_data))
    random.seed('2020')
    random.shuffle(data)

    f_src = open(os.path.join(data_path, 'sample_data.{}'.format(src_lang)), 'wt')
    f_trg = open(os.path.join(data_path, 'sample_data.{}'.format(trg_lang)), 'wt')
    if n_samples == 0:
        n_samples = data_len
    for idx, (i, j) in zip(range(0, n_samples), data):
        f_src.write(i)
        f_trg.write(j)
    f_src.close()
    f_trg.close()
    print('Done.')

    src = Field(tokenize=str.split, init_token='<sos>', eos_token='<eos>', lower=True)
    trg = Field(tokenize=str.split, init_token='<sos>', eos_token='<eos>', lower=True)

    dataset = TranslationDataset(path=os.path.join(data_path, 'sample_data.'), exts=(src_lang, trg_lang), fields=(src, trg))
    train_dataset, val_dataset = dataset.split(0.9)

    print('Building vocabulary...')
    src.build_vocab(train_dataset, min_freq=2)
    trg.build_vocab(train_dataset, min_freq=2)
    print('Unique tokens in source ({}) vocabulary: {}'.format(src_lang, len(src.vocab)))
    print('Unique tokens in target ({}) vocabulary: {}'.format(trg_lang, len(trg.vocab)))

    train_iterator, val_iterator = BucketIterator.splits(
        datasets=(train_dataset, val_dataset),
        batch_size=batchsize, device=device
    )

    return src, trg, train_iterator, val_iterator


if __name__ == '__main__':
    train_data_loader(device='cuda:0')
