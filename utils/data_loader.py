import os

from torchtext.data import Field, Iterator
from torchtext.datasets import TranslationDataset

from process_dm_math import DATASET_TARGET_DIR

TRAIN_FILE_NAME = "train"
EVAL_FILE_NAME = "interpolate"

INPUTS_FILE_ENDING = ".x"
TARGETS_FILE_ENDING = ".y"

class DataLoader:
  def __init__(self, module_name, train_bs, eval_bs, device, log):
    self.module_name = module_name

    # split_chars = lambda x: list("".join(x.split()))
    split_chars = lambda x: list(x)  # keeps whitespaces

    source = Field(tokenize=split_chars,
                   init_token='<sos>',
                   eos_token='<eos>',
                   batch_first=True)

    target = Field(tokenize=split_chars,
                   init_token='<sos>',
                   eos_token='<eos>',
                   batch_first=True)

    log("Loading FULL datasets ...")
    folder = os.path.join(DATASET_TARGET_DIR, module_name)
    train_dataset, eval_dataset, _ = TranslationDataset.splits(
      path=folder,
      root=folder,
      exts=(INPUTS_FILE_ENDING, TARGETS_FILE_ENDING),
      fields=(source, target),
      train=TRAIN_FILE_NAME,
      validation=EVAL_FILE_NAME,
      test=EVAL_FILE_NAME)

    log("Building vocab ...")
    source.build_vocab(train_dataset)
    target.vocab = source.vocab

    log("Creating iterators ...")
    train_iterator = Iterator(dataset=train_dataset,
                              batch_size=train_bs,
                              train=True,
                              repeat=True,
                              shuffle=True,
                              device=device)

    eval_iterator = Iterator(dataset=eval_dataset,
                             batch_size=eval_bs,
                             train=False,
                             repeat=False,
                             shuffle=False,
                             device=device)

    self.train_dataset = train_dataset
    self.eval_dataset = eval_dataset
    self.train_iterator = train_iterator
    self.eval_iterator = eval_iterator
    self.source = source
    self.target = target

