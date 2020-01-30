from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k, IWSLT, TranslationDataset

def multi30k(device):
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

    train_dataset, val_dataset, test_dataset = Multi30k.splits(exts=('.de','.en'), fields=(source, target))
    print('Done.')

    print('Building vocabulary...')
    target.build_vocab(train_dataset, min_freq=20)
    source.build_vocab(train_dataset, min_freq=20)
    print('Done.')

    train_iterator, val_iterator, test_iterator = BucketIterator.splits(
        datasets=(train_dataset, val_dataset, test_dataset),
        batch_size=32, 
        device=device
    )

    return (source, target), (train_iterator, val_iterator, test_iterator)

def iwslt2014(device, train_path='data/iwslt2014/train.de-en.bpe', dev_path='data/iwslt2014/dev.de-en.bpe', test_path='data/iwslt2014/test.de-en.bpe'):
    print('Loading data with torchtext...')
    source = Field(
            sequential=True,
            use_vocab=True,
            init_token='<SOS>',
            eos_token='<EOS>',
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
            tokenize=str.split,
            include_lengths=True,
            batch_first=False,
            pad_token='<pad>',
            unk_token='<unk>'
        )

    train_dataset = TranslationDataset(path=train_path, exts=('.de','.en'), fields=(source, target))
    val_dataset = TranslationDataset(path=dev_path, exts=('.de','.en'), fields=(source, target))
    test_dataset = TranslationDataset(path=test_path, exts=('.de','.en'), fields=(source, target))
    print('Done.')

    print('Building vocabulary...')
    target.build_vocab(train_dataset, min_freq=20)
    source.build_vocab(train_dataset, min_freq=20)
    print('Done.')

    train_iterator, val_iterator, test_iterator = BucketIterator.splits(
        datasets=(train_dataset, val_dataset, test_dataset),
        batch_size=32, 
        device=device
    )

    return (source, target), (train_iterator, val_iterator, test_iterator)
