import classification_transformer

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

# from torchtext import data, datasets, vocab
from torchtext.legacy import data, datasets, vocab

import numpy as np

from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

import random, tqdm, sys, math, gzip

# Used for converting between nats and bits
LOG2E = math.log2(math.e)
TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
LABEL = data.Field(sequential=False)
NUM_CLS = 2


def go(arg):
    """
    Creates and trains a basic transformer for the IMDB sentiment classification task.
    """
    tbw = SummaryWriter(log_dir=arg.tb_dir) # Tensorboard logging 

    # load the IMDB data
    if arg.final:
        train, test = dataset.IMDB.splits(TEXT, LABEL)
        train, test = datasets.IMDB.splits(TEXT, LABEL)
