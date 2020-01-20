import os
import time
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from model2lstm import lstm2lstm_baseline
from data_preprocessing import train_data_loader
from utils import eplased_time_since, count_parameters

import warnings
warnings.filterwarnings("ignore")


class Solver:
    def __init__(self, args):
        self.args = args

        # gpus
        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_iterator = None
        self.val_iterator = None
        self.src = None
        self.trg = None

    def init_training(self):
        if not os.path.exists(self.args.ckp_path):
            os.makedirs(self.args.ckp_path)

        print('Preparing data...')
        src, trg, train_iterator, val_iterator = \
            train_data_loader(self.args.data_path, self.args.src_lang, self.args.trg_lang, self.args.n_samples,
                            self.args.batch_size, self.device)
        input_dim, output_dim = len(src.vocab), len(trg.vocab)
        print('Done.')

        model = lstm2lstm_baseline(self.device, input_dim, output_dim)
        self.model = model
        print('The model has {} trainable parameters'.format(count_parameters(model)))

        device_count = 0
        if self.device == 'cuda':
            device_count = torch.cuda.device_count()
            if device_count > 1:
                model = nn.DataParallel(model, dim=1)
            torch.backends.cudnn.benchmark = True
        print("Let's use {} GPUs!".format(device_count))
        model.to(self.device)

        optimizer = optim.Adam(model.parameters())

        #  passing the index of the <pad> token
        #  as the ignore_index argument we ignore the loss whenever the target token is a padding token
        trg_pad_idx = trg.vocab.stoi[trg.pad_token]
        criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.src = src
        self.trg = trg
    
        # pickle
        print('Saving preprocessing pipeline...')
        model_dict = {}
        model_dict['input_dim'] = input_dim
        model_dict['output_dim'] = output_dim
        model_dict['source_vocab'] = src
        model_dict['target_vocab'] = trg
        pickle.dump(model_dict, open(os.path.join(self.args.ckp_path, 'model_dict.pkl'), 'wb'))

    def train(self):
        best_val_loss = float('inf')
        best_val_loss_epoch = 0

        print('\nStarting Training....')
        for epoch in range(self.args.n_epochs):
            print('*'*20+'Epoch: {}'.format(epoch+1)+'*'*20)
            start_time = time.time()
            train_loss = self.train_epoch()
            val_loss = self.validate(self.val_iterator)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_loss_epoch = epoch
                torch.save(self.model.state_dict(), os.path.join(self.args.ckp_path, 'model.pth'))

            print('Epoch: {}/{}, Time: {}, '
                  'Train Loss: {:.3f}, Val Loss: {:.3f}, '
                  'Best Val Loss: {:.3f}, Best Val Loss Epoch: {}\n'.format(epoch+1, self.args.n_epochs,
                                                                          eplased_time_since(start_time),
                                                                          train_loss, val_loss,
                                                                          best_val_loss, best_val_loss_epoch+1))


    def train_epoch(self):
        self.model.train()
        epoch_loss = 0

        start_time = time.time()
        for idx, batch in enumerate(self.train_iterator):
            src = batch.src
            trg = batch.trg  # trg: [trg_len, batch_size]
            output = self.model(src, trg)  # output: [trg_len, batch_size, output_dim]

            output_dim = output.shape[-1]
            # our decoder loop starts at 1, not 0.
            # This means the 0th element of our outputs tensor remains all zeros
            # so we cut off the first element of each tensor to get
            output = output[1:].view(-1, output_dim)  # [(trg len - 1) * batch size]
            trg = trg[1:].view(-1)  # [(trg len - 1) * batch size, output dim]

            self.optimizer.zero_grad()
            loss = self.criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()
            epoch_loss += loss.item()

            if idx == 0 or (idx+1) % 50 == 0 or (idx+1) == len(self.train_iterator):
                print('Batch: {}/{}, Batch Training Time: {}, Batch Loss: {:.3f}'.format(
                    idx+1, len(self.train_iterator),
                    eplased_time_since(start_time),
                    loss.item()
                ))
                start_time = time.time()

        return epoch_loss / len(self.train_iterator)

    def validate(self, iterator):
        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for idx, batch in enumerate(iterator):
                src = batch.src
                trg = batch.trg  # [trg_len, batch_size]

                output = self.model(src, trg)    # [trg_len, batch_size, output_dim]
                output_dim = output.shape[-1]
                # our decoder loop starts at 1, not 0.
                # This means the 0th element of our outputs tensor remains all zeros
                # so we cut off the first element of each tensor to get
                output = output[1:].view(-1, output_dim)  # [(trg len - 1) * batch size, output dim]
                trg = trg[1:].view(-1)  # [(trg len - 1) * batch size]

                loss = self.criterion(output, trg)
                epoch_loss += loss.item()
        return epoch_loss / len(iterator)

    def translate(self, sentence):
        print('Testing...')

        # preprocessing pipeline
        print('Preprocessing test set...')
        model_dict = pickle.load(open(os.path.join(self.args.ckp_path, 'model_dict.pkl'), 'rb'))

        model = lstm2lstm_baseline(self.device, model_dict['input_dim'], model_dict['output_dim'])
        self.model = model

        ckp = torch.load(os.path.join(self.args.ckp_path, 'model.pth'))
        self.model.load_state_dict(ckp)


        self.model.translate(sentence)
        print('Test Loss: {:.3f}'.format(test_loss))

    # def translate(self, sentence):
    #     ckp = torch.load(os.path.join(self.args.ckp_path, 'model.pth'))
    #     self.model.load_state_dict(ckp)
    #     import spacy
    #     spacy_lang = spacy.load(self.args.src_lang)
    #
    #     sentence = '<sos>' + sentence + '<eos>'
    #     ids = [self.src.vocab.stoi[tok.text] for tok in spacy_lang.tokenizer(sentence)]
    #     ids = torch.tensor(ids, dtype=torch.long).unsqueeze(1).to(self.device)
    #
    #     output_ids = self.model(ids, ids).argmax(2).view(-1).detach().to('cpu').numpy()
    #
    #     output_sentence = []
    #     for each in output_ids:
    #         output_sentence.append(self.trg.vocab.itos[each])
    #
    #     print(output_sentence)
