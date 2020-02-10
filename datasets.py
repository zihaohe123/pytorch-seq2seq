from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k, IWSLT, TranslationDataset

from transformers import BertTokenizer

def multi30k(device, batch_size, train_path, dev_path, test_path, bert = True):
    print('Loading data with torchtext...')
    if bert:
        tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
        source = Field(
                sequential=True,
                use_vocab=False,
                init_token=tokenizer.cls_token_id,
                eos_token=tokenizer.sep_token_id,
                tokenize=tokenizer.encode,
                include_lengths=True,
                batch_first=False,
                pad_token=tokenizer.pad_token_id,
                unk_token=tokenizer.unk_token_id
            )
    else:
        source = Field(
                sequential=True,
                use_vocab=True,
                init_token='<sos>',
                eos_token='<eos>',
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
            init_token='[CLS]',
            eos_token='[SEP]',
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


def iwslt2014(device, batch_size, train_path, dev_path, test_path, bert=True):
    return bpe_dataset(device, bert, train_path, dev_path, test_path, batch_size)


def bpe_dataset(device, bert, train_path, dev_path, test_path, batch_size):
    print('Loading data with torchtext...')
    if bert:
        tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
        source = Field(
                sequential=True,
                use_vocab=False,
                init_token=tokenizer.cls_token_id,
                eos_token=tokenizer.sep_token_id,
                tokenize=tokenizer.encode,
                include_lengths=True,
                batch_first=False,
                pad_token=tokenizer.pad_token_id,
                unk_token=tokenizer.unk_token_id
            )
    else:
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
            init_token='[CLS]',
            eos_token='[SEP]',
            tokenize=str.split,
            include_lengths=True,
            batch_first=False,
            pad_token='<pad>',
            unk_token='<unk>'
        )

    train_dataset = TranslationDataset(path=train_path, exts=('.de', '.en'), fields=(source, target))
    val_dataset = TranslationDataset(path=dev_path, exts=('.de', '.en'), fields=(source, target))
    test_dataset = TranslationDataset(path=test_path, exts=('.de', '.en'), fields=(source, target))
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
