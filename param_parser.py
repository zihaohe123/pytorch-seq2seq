import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run Attention Walk.")

    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--n_epochs', type=int, default=10, help='num of epochs')
    parser.add_argument('--ckp_path', type=str, default='ckp', help='saved model path')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--src_lang', type=str, default='de')
    parser.add_argument('--trg_lang', type=str, default='en')
    parser.add_argument('--n_samples', type=int, default=0, help='num of samples to use')
    parser.add_argument('--clip', type=float, default=1, help='max gradient')
    parser.add_argument('--enc_emb_dim', type=int, default=128, help='encoder embedding dimension')
    parser.add_argument('--dec_emb_dim', type=int, default=128, help='decoder embedding dimension')
    parser.add_argument('--hid_dim', type=int, default=256, help='hidden dimension')
    parser.add_argument('--n_layers', type=int, default=1, help='num of rnn layers')
    parser.add_argument('--enc_dropout', type=float, default=0.5)
    parser.add_argument('--dec_dropout', type=float, default=0.5)
    parser.add_argument('--teacher_force_ratio', type=float, default=0.75)
    parser.add_argument('--gpu', type=str, default='')

    return parser.parse_args()