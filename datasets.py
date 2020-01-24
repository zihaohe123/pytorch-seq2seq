from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k
from torchtext.datasets import IWSLT

def multi30k(device):
    print('Loading data with torchtext...')
    tokenizer = lambda x: x.split()
    detokenizer = lambda x: ' '.join(x)

    source = Field(
            sequential=True,
            use_vocab=True,
            init_token='<SOS>',
            eos_token='<EOS>',
            lower=True,
            tokenize=tokenizer,
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
            tokenize=tokenizer,
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

    return (source, target), (train_iterator, val_iterator), (tokenizer, detokenizer)

def iwslt(device):
    print('Loading data with torchtext...')
    tokenizer = lambda x: x.split()
    detokenizer = lambda x: ' '.join(x)

    source = Field(
            sequential=True,
            use_vocab=True,
            init_token='<SOS>',
            eos_token='<EOS>',
            lower=True,
            tokenize=tokenizer,
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
            tokenize=tokenizer,
            include_lengths=True,
            batch_first=False,
            pad_token='<pad>',
            unk_token='<unk>'
        )

    train_dataset, test_dataset, val_dataset = IWSLT.splits(exts=('.de','.en'), fields=(source, target))
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

    return (source, target), (train_iterator, val_iterator), (tokenizer, detokenizer)
