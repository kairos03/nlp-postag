import copy
import logging
import os
import sys

import dill
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import yaml
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import Sejong
from model import CRFTagger, KMAModel, RNNDecoderPointer, RNNEncoder

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

logger.info("==> Load config")
with open(os.path.join('config', 'config.yml'), 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

device = torch.device('cuda' if torch.cuda.is_available() and config['gpu'] else 'cpu')
logger.info('Device: %s' % device)


def unique_everseen(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


WORD = torchtext.data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True, include_lengths=True)
LEX = torchtext.data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True, is_target=True)
POS_TAG = torchtext.data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True,
                            unk_token=None, is_target=True)


logger.info("==> Load Dataset")
train, valid = Sejong.splits(fields=(WORD, LEX, POS_TAG),
                                    train=config['train_file'],
                                    validation=config['validation_file'],
                                    test=None,
                                    max_token=config['preprocessing']['max_token'])


logger.info('==> Sejong Dataset Loaded: ')
logger.info('Train size: %d' % (len(train)))
logger.info('Validation size: %d' % (len(valid)))

min_freq = config['preprocessing']['min_freq']
WORD.build_vocab(train, specials=["<pad>", "<tok>", "<blank>"], min_freq=min_freq)
LEX.build_vocab(train, specials=["<pad>", "<tok>", "<blank>"], min_freq=min_freq)
POS_TAG.build_vocab(train, specials=["<pad>", "<tok>", "<blank>"])

MERGE = copy.deepcopy(WORD)
for k in LEX.vocab.stoi:
    if k not in MERGE.vocab.stoi:
        MERGE.vocab.stoi[k] = len(MERGE.vocab.stoi)
MERGE.vocab.itos.extend(LEX.vocab.itos)
MERGE.vocab.itos = unique_everseen(MERGE.vocab.itos)
MERGE.is_target = True
MERGE.include_lengths = False
LEX = MERGE

train.fields['lex'] = LEX
valid.fields['lex'] = LEX

logger.info('Size of WORD vocab: %d' % len(WORD.vocab))
logger.info('Size of LEX vocab: %d' % len(LEX.vocab))
logger.info('Size of POS_TAG vocab: %d' % len(POS_TAG.vocab))

vocab = {"WORD": WORD, "LEX": LEX, "POS": POS_TAG}
with open(config['vocab_name'], 'wb') as fout:
    dill.dump(vocab, fout)

encoder = RNNEncoder(vocab_size=len(WORD.vocab), pad_id=WORD.vocab.stoi[WORD.pad_token], **config['encoder'])
if config['encoder']['bidirectional']:
    hidden_size = config['encoder']['hidden_size'] * 2
else:
    config['encoder']['hidden_size']
decoder = RNNDecoderPointer(vocab_size=len(LEX.vocab), hidden_size=hidden_size,
                            sos_id=LEX.vocab.stoi[LEX.init_token], eos_id=LEX.vocab.stoi[LEX.eos_token],
                            pad_id=LEX.vocab.stoi[LEX.pad_token], **config['decoder'])
tagger = CRFTagger(hidden_size=hidden_size, num_tags=len(POS_TAG.vocab))
model = KMAModel(encoder, decoder, tagger).to(device)
logger.info(model)

model_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = getattr(optim, config['optimizer']['optim'])(model_params, lr=config['optimizer']['lr'])
scheduler = ReduceLROnPlateau(optimizer, 'min')
criterion = nn.NLLLoss(ignore_index=LEX.vocab.stoi[LEX.pad_token])

train_iter = torchtext.data.BucketIterator(train, batch_size=config['learning']['batch_size'],
                                        sort_key=lambda x: x.word.__len__(),
                                        sort_within_batch=True,
                                        shuffle=True)

valid_iter = torchtext.data.BucketIterator(valid, batch_size=config['learning']['batch_size'],
                                        sort_key=lambda x: x.word.__len__(),
                                        sort_within_batch=True,
                                        shuffle=True)

for epoch in range(config['learning']['epochs']):
    model.train()
    train_loss = 0

    for data in tqdm(train_iter):
        decoder_outputs, tagger_loss, others = model(data.word[0].to(device), data.lex.to(device), data.pos.to(device),
                                                    input_lengths=None,
                                                    teaching_force_ratio=config['decoder']['teaching_force_ratio'])
        optimizer.zero_grad()
        lex_loss = 0
        for step, step_output in enumerate(decoder_outputs):
            batch_size = data.lex.size(0)
            lex_loss += criterion(step_output.view(batch_size, -1), data.lex[:, step + 1].to(device))

        loss = lex_loss + tagger_loss
        train_loss += loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['learning']['max_grad_norm'])
        optimizer.step()

    model.eval()
    dev_loss = 0
    with torch.no_grad():
        dev_loss = 0
        for val_data in tqdm(valid_iter):
            decoder_outputs, tagger_loss, others = model(val_data.word[0].to(device),
                                                         val_data.lex.to(device),
                                                         val_data.pos.to(device),
                                                         input_lengths=val_data.word[1])

            dev_loss += tagger_loss
            for step, step_output in enumerate(decoder_outputs):
                batch_size = val_data.lex.size(0)
                dev_loss += criterion(step_output.view(batch_size, -1),
                                      val_data.lex[:, step + 1].to(device))

    scheduler.step(dev_loss)
    logger.info("Epoch: %d, Train loss: %f, Dev loss: %f" % (epoch, train_loss, dev_loss))

dict_save = {"model": model.state_dict(), "epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss}
torch.save(dict_save, config['model_name'])
