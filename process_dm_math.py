# Simple preprocessing Script for the Deepmind Mathematics Dataset
#
# 1. Merges train_easy, train_medium, and train_hard per task
# 2. Splits train into train_inputs and train_targets.
# 3. Splits interpolation into eval_inputs and eval targets
#
# This program now generates data in both formats:
#   original: - .x and .y files
#   JIT:      - .xy and .index files

import os
import torch

# The path to the extracted Deepmind dataset 
DATA_SOURCE_DIR = "raw_data/deepmind_mathematics/v1.0/"

# Path to write the new modules into
DATASET_TARGET_DIR = "data/dm_math/"

TRAIN_SUB_DIRS = ["train-easy", "train-medium", "train-hard"]
INTER_SUB_DIRS = ["interpolate"]
EXTRA_SUB_DIRS = ["extrapolate"]

def read_files(subdirs, module_file):
  all_lines = []
  for subdir in subdirs:
    with open(os.path.join(DATA_SOURCE_DIR, subdir, module_file), "r") as f:
      lines = f.readlines()
    print("... read {} lines from {}".format(len(lines), subdir))
    all_lines += lines
  return all_lines

def split_into_x_y(lines):
  x, y = [], []
  for idx in range(0,len(lines),2):
    x.append(lines[idx])
    y.append(lines[idx+1])
  return x, y

def make_jit_pairs_and_indexes(x, y):
    xy_list = []
    indexes = []
    file_offset = 0
    vocab = set()

    for xx, yy in zip(x, y):
        # remove NEWLINEs
        x_line = xx.replace("\n", "")
        y_line = yy.replace("\n", "")

        # update vocab with both x and y
        vocab.update(list(x_line))
        vocab.update(list(y_line))

        assert not "\t" in x_line
        assert not "\t" in y_line

        # merge x,y into same line
        xy_line = "{}\t{}\n".format(x_line, y_line)

        # accumulate xy and index for JIT
        xy_list.append(xy_line)
        indexes.append(file_offset)

        file_offset += len(xy_line)

    vocab = "".join(list(vocab))
    return xy_list, indexes, vocab

def write_file(path, file, lines):
  # specify 'newline' here to avoid adding CR to \n on windows
  with open(os.path.join(path, file), "w", newline="") as f:
    f.writelines(lines)

def process_module_group(group_name, subdirs, module_name):
  module_file = module_name + ".txt"

  # read the group data of this module
  lines = read_files(subdirs=subdirs, module_file=module_file)
  print("total {} lines read={}".format(group_name, len(lines)))

  # split data into input lines and output lines
  inputs, targets = split_into_x_y(lines)
  print("total {} samples input={} targets={}".format(group_name, 
        len(inputs), len(targets)))

  # generate JIT data
  xy_list, indexes, vocab = make_jit_pairs_and_indexes(inputs, targets)
  print("{} vocab=".format(group_name), vocab)
  
  # ensure directory exists
  target_dir = os.path.join(DATASET_TARGET_DIR, module_name)
  if not os.path.exists(target_dir):
    os.makedirs(target_dir)

  # write NORMAL format files
  print("Writing files into {} ...".format(target_dir), end="")
  write_file(path=target_dir, file="{}.x".format(group_name), lines=inputs)
  write_file(path=target_dir, file="{}.y".format(group_name), lines=targets)
  
  # write JIT format files
  write_file(path=target_dir, file="{}.xy".format(group_name), lines=xy_list)
  write_file(path=target_dir, file="{}.vocab".format(group_name), lines=[vocab])
  torch.save(indexes, os.path.join(target_dir, group_name + ".indexes_pt"))
  print()

def process_all_modules():
  modules = os.listdir(os.path.join(DATA_SOURCE_DIR, "interpolate"))
  print("Starting to process {} modules".format(len(modules)))
  print()

  for idx, module_file in enumerate(modules):
    module_name = module_file[:-4]

    print("{}.) Processing {} ...".format(idx, module_name))
    process_module_group("train", TRAIN_SUB_DIRS, module_name)
    process_module_group("interpolate", INTER_SUB_DIRS, module_name)
    print()

  modules = os.listdir(os.path.join(DATA_SOURCE_DIR, "extrapolate"))
  print("what")
  for idx, module_file in enumerate(modules):
    module_name = module_file[:-4]

    print("{}.) Processing {} ...".format(idx, module_name))
    process_module_group("extrapolate", EXTRA_SUB_DIRS, module_name)
    print()

print(" Done.")

if __name__ == "__main__":
    process_all_modules()