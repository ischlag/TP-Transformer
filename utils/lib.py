import logging
import os
import random
import shutil

import torch


def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def assert_params_exist(p, keys):
  for k in keys:
    if not k in p.__dict__.keys():
      raise AttributeError("Necessary parameter {} is missing!"
                           .format(k))


def pick_new_seed(log_folder):
  if not os.path.exists(log_folder):
    return random.randint(1,99999)

  existing_seeds = os.listdir(log_folder)
  while True:
    seed = random.randint(1,99999)
    if seed not in existing_seeds:
      break
  return seed


def tensor_find(t, value):
  result = (t == value).nonzero().cpu().numpy()
  index = result[0][0] if len(result) else -1
  return index


def calc_string_acc(y, y_hat, pad_value):
  """
  Ignores first symbol (SOS) and returns 0 or 1 for full sequence match.
  y_hat has to also predict the EOS symbol.
  :param y: single target sequence
  :param y_hat: single predicted sequence
  :param eos_idx: EOS idx
  :return: 1 if all sequence elements match else 0
  """
  # single sample
  # print("calc_string_acc: y.shape=", y.shape, ", y_hat.shape=", y_hat.shape)

  y_shifted = y[1:]              # drop the SOS from y
  exact_match = (torch.eq(y_shifted,y_hat) | (y_shifted==pad_value)).all().item()
  return exact_match


def compute_accuracy(logits, targets, pad_value):
  """
  Compute full sequence accuracy of a batch.
  :param logits: the model logits (batch_size, seq_len, out_dim)
  :param targets: the true targets (batch_size, seq_len)
  :param pad_value: PAD value used to fill end of target seqs
  :return: continous accuracy between 0.0 and 1.0
  """
  trg_shifted = targets[:, 1:]              # drop the SOS from targets
  y_hat = torch.argmax(logits, dim=-1)      # get index predictions from logits

  # count matches in batch, masking out pad values in each target
  matches = (torch.eq(trg_shifted,y_hat) | (trg_shifted==pad_value)).all(1).sum().item()
  
  acc_percent = matches / len(logits)
  return acc_percent

def compute_sample_accuracy(logits, targets, pad_value):
  """
  Compute full sequence accuracy of a batch.
  :param logits: the model logits (batch_size, seq_len, out_dim)
  :param targets: the true targets (batch_size, seq_len)
  :param pad_value: PAD value used to fill end of target seqs
  :return: continous accuracy between 0.0 and 1.0
  """
  trg_shifted = targets[:, 1:]              # drop the SOS from targets
  y_hat = torch.argmax(logits, dim=-1)      # get index predictions from logits

  # count matches in batch, masking out pad values in each target
  matches = (torch.eq(trg_shifted,y_hat) | (trg_shifted==pad_value)).all(1)
  return matches


def get_dynamic_matches(y_hat, targets, eos_value):
  if targets.shape[1] > y_hat.shape[1]:
    tmp = torch.ones_like(targets)
    tmp[:, :y_hat.shape[1]] = y_hat
    y_hat = tmp
  elif targets.shape[1] < y_hat.shape[1]:
    tmp = y_hat[:, :targets.shape[1]]
    y_hat = tmp

  correct = torch.ones((y_hat.shape[0])).type(torch.uint8)
  done = torch.zeros((y_hat.shape[0])).type(torch.uint8)

  for i in range(targets.shape[1]):
    #print(i)
    match = (y_hat[:, i] == targets[:, i]).cpu().type(torch.uint8)
    #print('match\t', match)
    relevant_match = match | done  # OR
    #print("r_match\t", relevant_match)
    correct = correct & relevant_match  # AND
    #print("correct\t", correct)
    eos_match = (targets[:, i] == eos_value).cpu().type(torch.uint8)
    #print("eos?\t", eos_match)
    done = done | eos_match  # OR
    #print("done\t", done)
  return correct

def setup_log_folder(log_folder, force_remove=False, force_reload=False):
  if os.path.exists(log_folder) and not force_remove:
    print("WARNING: The results directory (%s) already exists. "
          "Delete previous results directory [y/N]? " % log_folder, end="")
    if not force_reload:
      choice = input()
    else:
      choice = "n"
    if choice is "y" or choice is "Y":
      print("removing directory ...")
      shutil.rmtree(log_folder)
      os.makedirs(log_folder)
    else:
      print("WARNING: The results directory already exists: %s" % log_folder)
      #sys.exit(1)
  elif os.path.exists(log_folder) and force_remove:
    print("removing directory ...")
    shutil.rmtree(log_folder)
    os.makedirs(log_folder)
  else:
    print("creating new log directory ...")
    os.makedirs(log_folder)

def save_current_script(log_folder, log):
  # source folder
  current_folder = os.getcwd()
  log("Taking scripts from {}".format(current_folder))

  # target folder
  copy_to_folder = os.path.join(log_folder, "source")
  log("... and saving them in {}".format(copy_to_folder))

  # create folder
  if not os.path.exists(copy_to_folder):
    os.makedirs(copy_to_folder)
  else:
    # do not copy scripts if folder alreay exists
    return

  # get file names
  files = [f for f in os.listdir(current_folder) if f[-3:] == ".py"]

  # copy files
  for f in files:
    source_file = os.path.join(current_folder, f)
    target_file = os.path.join(copy_to_folder, f)
    shutil.copyfile(source_file, target_file)

  # also copy models and utils folder
  shutil.copytree("models", os.path.join(copy_to_folder, "models"))
  shutil.copytree("utils", os.path.join(copy_to_folder, "utils"))

def setup_logger(log_folder, file_name="output.log", write_to_file=True):
  logger = logging.getLogger("my_logger")
  logger.setLevel(logging.DEBUG)
  logger.addHandler(logging.StreamHandler())
  if write_to_file:
    logger.addHandler(logging.FileHandler(os.path.join(log_folder, file_name)))

  return lambda *x: logger.debug((x[0]
                               .replace('{','{{')
                               .replace('}','}}') + "{} " * (len(x)-1))
                              .format(*x[1:]))


def pretty_args(args):
  keys = args.__dict__.keys()
  items = ("{}={!r}".format(k, args.__dict__[k]) for k in keys)
  return "{}:\n\t{}".format("Arguments", "\n\t".join(items))

class HyperParams:
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

  def keys(self):
    return self.__dict__.keys()

  def __repr__(self):
    keys = self.__dict__.keys()
    items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
    return "{}:\n\t{}".format(type(self).__name__, "\n\t".join(items))

  def __eq__(self, other):
    return self.__dict__ == other.__dict__

class Terminal:
  def __init__(self, model, module, p):
    self.module = module
    self.model = model
    self.p = p

  def enter(self, str_sentence):
    """
    E.g. terminal.enter("Evaluate: 1 + 2")
    :param string input
    :return: string output
    """
    for idx, c in enumerate(str_sentence):
      if not c in self.module.source.vocab.itos:
        print("invalid character \"{}\" at pos {}. The only valid characters"
              "are {}".format(c, idx, "".join(self.module.source.vocab.itos)))
        return None

    encoded_sequence = self.module.encode([str_sentence])
    prediction = self.model.greedy_inference(src=encoded_sequence,
                                             sos_idx=self.p.SOS,
                                             eos_idx=self.p.EOS,
                                             max_length=50,
                                             device=self.p.device)
    return self.module.decode(prediction)

def pretty(attention, text):
  block = ""
  for head in attention:
    line = ""
    for prob in head:
      line += "{:.2f} ".format(prob)
    block += line + "\n"

  line = ""
  for character in text:
    line += "{:5}".format(character)
  block += line

  print(block)