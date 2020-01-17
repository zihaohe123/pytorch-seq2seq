from torchtext.datasets import Multi30k, TranslationDataset
from torchtext.data import Field, BucketIterator
import spacy, os, random


def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings (tokens) and reverses it
    """
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]


def tokenize_en(text):
    """
    Tokenizes English text from a string to a list of strings (tokens)
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


def get_data_loader(data_path, src_lang, trg_lang, n_samples=0, batchsize=128, device='cpu'):

    # sampling data
    print('Sampling data....')
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

    global spacy_de, spacy_en
    try:
        spacy_de = spacy.load('de')
    except:
        os.system('python -m spacy download de')
        spacy_de = spacy.load('de')
    try:
        spacy_en = spacy.load('en')
    except:
        os.system('python -m spacy download en')
        spacy_en = spacy.load('en')

    tokenizer_dict = {'de': tokenize_de, 'en': tokenize_en}
    src = Field(tokenize=tokenizer_dict[src_lang], init_token='<sos>', eos_token='<eos>', lower=True)
    trg = Field(tokenize=tokenizer_dict[trg_lang], init_token='<sos>', eos_token='<eos>', lower=True)

    print('Splitting train, val, test data....')
    dataset = TranslationDataset(path=os.path.join(data_path, 'sample_data.'), exts=(src_lang, trg_lang), fields=(src, trg))
    train_val_dataset, test_dataset = dataset.split(0.9)
    train_dataset, val_dataset = train_val_dataset.split(8/9)

    print('Building vocabulary....')
    src.build_vocab(train_dataset, min_freq=2)
    trg.build_vocab(train_dataset, min_freq=2)
    print('Unique tokens in source ({}) vocabulary: {}'.format(src_lang, len(src.vocab)))
    print('Unique tokens in target ({}) vocabulary: {}'.format(trg_lang, len(trg.vocab)))

    train_iterator, val_iterator, test_iterator = BucketIterator.splits(
        datasets=(train_dataset, val_dataset, test_dataset),
        batch_size=batchsize, device=device
    )

    return src, trg, train_iterator, val_iterator, test_iterator


if __name__ == '__main__':
    get_data_loader(device='cuda:0')