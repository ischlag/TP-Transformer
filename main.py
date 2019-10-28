import argparse
import importlib
import sys

import torch.nn as nn

from utils.data_loader import *
from utils.jit_data_loader import *
from utils.lib import *
from utils.trainer import BasicSeq2SeqTrainer
from process_dm_math import DATASET_TARGET_DIR

# parse arguments
parser = argparse.ArgumentParser(
  description='TP-Transformer and the Deepmind Mathematics Dataset')
# setup parameters
parser.add_argument('--seed', type=int,
                    default=0xBADB1A5, metavar='SEED',
                    help='random seed (default: 0xBADB1A5)')
# experiment parameters
parser.add_argument('--model_name', type=str, default=None, metavar='NAME',
                    help='model name (default: transformer_official)')
parser.add_argument('--module_name', type=str,
                    default="numbers__place_value", metavar='NAME',
                    help='module name (default: numbers__place_value)')
parser.add_argument('--load_model', type=str, default="", metavar='S',
                    help='Model to load (default: "")')
parser.add_argument('--eval_mode', action='store_true',
                    help="Don't write logs. (Default: False)")
parser.add_argument('--n_steps', type=int,
                    default=10000, metavar='N',
                    help='maximum number of steps to train (default: 10000)')
parser.add_argument('--max_strikes', type=int,
                    default=1000, metavar='N',
                    help='number of steps without eval loss improvement '
                         'before exiting (default: 1000)')
parser.add_argument('--log_every', type=int,
                    default=50, metavar='N',
                    help='after how many steps to log to terminal '
                         'and tensorboard (default: 50)')
parser.add_argument('--eval_every', type=int,
                    default=200, metavar='N',
                    help='after how many steps to evaluate the model on the '
                         'held-out data (default: 200)')
parser.add_argument('--full_loader', action='store_true', 
                    help="Use full data loader instead of JIT loader "
                         "(default: False)")
parser.add_argument('--force_remove', action='store_true',
                    help="Removes pre-existing log folders (default: False)")
parser.add_argument('--force_reload', action='store_true',
                    help="Load previous model if available. (Default: False)")
parser.add_argument('--no_train', action='store_true',
                    help="Don't start training. (Default: False)")
parser.add_argument('--log_folder', type=str, default="log", metavar='S',
                    help='Log folder (default: "")')
parser.add_argument('-s', '--log_suffix', type=str,
                    default="", metavar='S',
                    help='Additional log suffix (default: "")')
# optimizer parameters
parser.add_argument('-opt', '--optimizer', type=str,
                    default="Adam", metavar='S',
                    help='the sgd optimizer (default: "Adam")')
parser.add_argument('-lr', '--learning_rate', type=float,
                    default=1e-4, metavar='F',
                    help='adam learning rate (default: 6e-4)')
parser.add_argument('--beta1', type=float,
                    default=0.9, metavar='F',
                    help='adam beta1 (default: 0.9)')
parser.add_argument('--beta2', type=float,
                    default=0.995, metavar='F',
                    help='adam beta2 (default: 0.995)')
parser.add_argument('-bs', '--batch_size', type=int,
                    default=256, metavar='N',
                    help='batch size for train and test (default: 256)')
parser.add_argument('--max_abs_grad_norm', type=float,
                    default=0.1, metavar='F',
                    help='max absolute gradient norm clip (default: 0.1)')
parser.add_argument('--grad_accum_steps', type=int,
                    default=1, metavar='N',
                    help='gradient accumulation steps (default: 1)')
# model parameters
parser.add_argument('--dropout', type=float,
                    default=0.0, metavar='PROB',
                    help='dropout (default: 0.0)')
parser.add_argument('--hidden', type=int,
                    default=512, metavar='N',
                    help='hidden size (default: 512)')
parser.add_argument('-l', '--n_layers', type=int,
                    default=6, metavar='N',
                    help='number of transformer layers (default: 6)')
parser.add_argument('-nh', '--n_heads', type=int,
                    default=8, metavar='N',
                    help='number of attention heads (default: 8)')
parser.add_argument('-f', '--filter', type=int,
                    default=2048, metavar='N',
                    help='filter size (default: 2048)')
parser.add_argument('-d_r', type=int,
                    default=0, metavar='N',
                    help='role size (default: 0)')
p = parser.parse_args()

# further refinements
if torch.cuda.is_available():
  p.device = torch.device('cuda')
  n_gpus = torch.cuda.device_count()
else:
  print("ERROR: No CUDA device available.")
  sys.exit()

if p.d_r:
  p.log_suffix = "dr={}_".format(p.d_r) + p.log_suffix

if n_gpus > 1:
  p.log_suffix = "GPU{}_".format(n_gpus) + p.log_suffix

# custom log folder
p.log_folder = "{}/{}/{}/lr={}_bs={}{}_h={}_f={}_nl={}_nh={}_d={}_{}_{}"\
  .format(p.log_folder,
          p.module_name,
          p.model_name,
          p.learning_rate,
          p.batch_size,
          "" if p.grad_accum_steps == 1 else "({})".format(p.grad_accum_steps),
          p.hidden,
          p.filter,
          p.n_layers,
          p.n_heads,
          p.dropout,
          p.optimizer,
          p.log_suffix)

if p.seed is None:
  p.seed = pick_new_seed(p.log_folder)
p.log_folder = os.path.join(p.log_folder, str(p.seed))

# setup the experiment
if p.eval_mode:
  log = setup_logger(p.log_folder, write_to_file=False)
else:
  setup_log_folder(p.log_folder,
                   force_remove=p.force_remove,
                   force_reload=p.force_reload)
  log = setup_logger(p.log_folder)
  save_current_script(p.log_folder, log)
log(pretty_args(p))

random.seed(p.seed)
torch.manual_seed(p.seed)
torch.backends.cudnn.deterministic = True


# load the dataset
if p.full_loader:
  # load FULL data (this will copy the whole dataset into memory)
  log("using full data loader")
  module = DataLoader(module_name=p.module_name,
                      train_bs=p.batch_size,
                      eval_bs=p.batch_size,
                      device=p.device,
                      log=log)
  train_iterator = module.train_iterator
  eval_iterator = module.eval_iterator
  p.input_dim = len(module.source.vocab)
  p.output_dim = p.input_dim
  p.SOS = module.source.vocab.stoi['<sos>']  # start of sentence token
  p.EOS = module.source.vocab.stoi['<eos>']  # end of sentence token
  p.PAD = module.source.vocab.stoi['<pad>']  # padding token

else:
  # Just-in-time (JIT) version loads only the indecies into memory
  log("using JIT loader")
  vocab = torch.load(os.path.join(DATASET_TARGET_DIR, "vocab.pt"))
  train_module = JitDataLoader(module_name=p.module_name,
                               file_name=TRAIN_FILE_NAME,
                               batch_size=p.batch_size,
                               is_train=True,
                               device=p.device,
                               log=log,
                               vocab=vocab)

  eval_module = JitDataLoader(module_name=p.module_name,
                              file_name=EVAL_FILE_NAME,
                              batch_size=p.batch_size,
                              is_train=False,
                              device=p.device,
                              log=log,
                              vocab=vocab)

  train_iterator = train_module.iterator
  eval_iterator = eval_module.iterator
  p.input_dim = len(train_module.source.vocab)
  p.output_dim = p.input_dim
  p.SOS = train_module.source.vocab.stoi['<sos>']  # start of sentence token
  p.EOS = train_module.source.vocab.stoi['<eos>']  # end of sentence token
  p.PAD = train_module.source.vocab.stoi['<pad>']  # padding token


# build model by importing the python module
log("Building model ...")
try:
  imp_module = importlib.import_module("models.{}".format(p.model_name))
  model = imp_module.build_transformer(params=p, pad_idx=p.PAD).to(p.device)
except ModuleNotFoundError:
  log("{} is not a valid model name.".format(p.model_name))
  sys.exit(1)
log("done. {} trainable parameters.".format(count_parameters(model)))

# Optimzier (Adam is the only option for now)
if p.optimizer == "Adam":
  optimizer = torch.optim.Adam(params=model.parameters(),
                               lr=p.learning_rate,
                               betas=(p.beta1, p.beta2))
else:
  raise NotImplementedError()
criterion = nn.CrossEntropyLoss(ignore_index=p.PAD)

# DataParallel over multiple GPUs
if n_gpus > 1:
  log("{} GPUs detected. Using nn.DataParallel. Batch-size per GPU: {}"
        .format(n_gpus, p.batch_size // n_gpus))
  model = nn.DataParallel(model)
model = model.to(p.device)

# create a trainer object that will deal with all the training
trainer = BasicSeq2SeqTrainer(model=model,
                              params=p,
                              train_iterator=train_iterator,
                              eval_iterator=eval_iterator,
                              optimizer=optimizer,
                              criterion=criterion,
                              log=log)

# load a previous model if a path is given
if p.load_model != "":
  if n_gpus == 1:
    # renaming is necessary for single gpu setting since it is assumed that
    # the model to be loaded was trained on a multi-gpu setting.
    state = torch.load(p.load_model)
    fixed_state = {key[7:]: state["model"][key]
                   for key in state["model"].keys()}
    trainer.model.load_state_dict(fixed_state)
    trainer.optimizer.load_state_dict(state["optimizer"])
    trainer.global_step = state["global_step"]
    trainer.best_eval_loss = state["best_eval_loss"]
    trainer.best_eval_acc = state["best_eval_acc"]
    trainer.best_step = state["best_step"]
  else:
    trainer.load_state(p.load_model)

print("doing initial evaluate()...")
trainer.evaluate()

if not p.no_train and not p.eval_mode:
  trainer.train(steps=p.n_steps)