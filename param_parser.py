import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description="Low resource machine translation.")

    # experimental arguments
    parser.add_argument('--ckp_path', type=str, default='ckp', help='saved model path')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--src_lang', type=str, default='de')
    parser.add_argument('--trg_lang', type=str, default='en')
    parser.add_argument('--n_samples', type=int, default=0, help='num of samples to use')
    parser.add_argument('--gpu', type=str, default='')

    # training arguments
    parser.add_argument('--n_epochs', type=int, default=10, help='num of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--clip', type=float, default=1, help='max gradient')

    return parser.parse_args()
