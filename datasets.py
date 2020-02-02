from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k, IWSLT, TranslationDataset


def multi30k(device, batch_size):
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

    train_dataset, val_dataset, test_dataset = Multi30k.splits(exts=('.de', '.en'), fields=(source, target))
    print('Done.')

    print('Building vocabulary...')
    target.build_vocab(train_dataset, min_freq=20)
    source.build_vocab(train_dataset, min_freq=20)
    print('Done.')

    train_iterator, val_iterator, test_iterator = BucketIterator.splits(
        datasets=(train_dataset, val_dataset, test_dataset),
        batch_size=batch_size,
        device=device
    )

    return (source, target), (train_iterator, val_iterator, test_iterator)


def iwslt2014(device, batch_size, train_path, dev_path, test_path):
    return bpe_dataset(device, train_path, dev_path, test_path, batch_size)


def bpe_dataset(device, train_path, dev_path, test_path, batch_size):
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

    train_iterator, val_iterator = BucketIterator.splits(
        datasets=(train_dataset, val_dataset),
        batch_size=batch_size,
        device=device
    )

    test_iterator = BucketIterator(
        dataset=test_dataset,
        train=False,
        shuffle=False,
        sort=False,
        batch_size=batch_size,
        device=device
    )

    return (source, target), (train_iterator, val_iterator, test_iterator)
