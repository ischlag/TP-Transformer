# Merges all preprocessed files together into one big all_module. Modules
# need to be generated first using process_dm_math.py.
#
# 1. the generated modules are already randomized.
# 2. randomly picks a module and reads 1 line until all modules are read
#
# This program now generates data both formats:
#   original: - .x and .y files
#   JIT:      - .xy and .indexes_pt, and .vocab files


import os
import sys
import random
import torch

# path to the generated modules
DATASET_DIR = "data/dm_math/"
ALL_MODULE_NAME = "all_modules"
SAMPLES_PER_MODULE = 2000000

def get_read_streams(name, modules):
  files = []
  for module in modules:
    x_file_name = os.path.join(DATASET_DIR, module, name + ".x")
    y_file_name = os.path.join(DATASET_DIR, module, name + ".y")

    if os.path.exists(x_file_name) and os.path.exists(y_file_name):
      x_file = open(x_file_name, "r")
      y_file = open(y_file_name, "r")
      files.append((x_file, y_file))

  return files

def get_write_stream(name):
  # specify 'newline' here to avoid adding CR to \n on windows
  x_file = open(os.path.join(DATASET_DIR, ALL_MODULE_NAME, name + ".x"),
                'w', newline='')
  y_file = open(os.path.join(DATASET_DIR, ALL_MODULE_NAME, name + ".y"),
                'w', newline='')
  xy_file = open(os.path.join(DATASET_DIR, ALL_MODULE_NAME, name + ".xy"),
                 'w', newline='')
  vocab_file = open(os.path.join(DATASET_DIR, ALL_MODULE_NAME, name + ".vocab"),
                    "w")
  fn_indexes_pt = os.path.join(DATASET_DIR,
                               ALL_MODULE_NAME,
                               name + ".indexes_pt")
  
  return x_file, y_file, xy_file, vocab_file, fn_indexes_pt

def random_merge(read_file_list, write_file_list):
  total_lines = len(read_file_list) * SAMPLES_PER_MODULE
  file_write_counter = 0
  file_offset = 0
  vocab = set()
  indexes = []

  x_file, y_file, xy_file, vocab_file, fn_indexes_pt = write_file_list

  while True:
    if len(read_file_list) == 0:
      break

    # draw random module files
    rf = random.choice(read_file_list)

    # read one line each, or close and remove file if empty
    x_line = rf[0].readline()
    y_line = rf[1].readline()

    if len(x_line) == 0:
      rf[0].close()
      rf[1].close()
      read_file_list.remove(rf)
      continue

    # merge x,y into same line
    x_line = x_line.replace("\n", "")
    y_line = y_line.replace("\n", "")
    vocab.update(list(x_line))
    vocab.update(list(y_line))

    assert not "\t" in x_line
    assert not "\t" in y_line

    xy_line = "{}\t{}\n".format(x_line, y_line)

    # write X, Y (for now, support both FULL and JIT formats)
    x_file.write(x_line + "\n")
    y_file.write(y_line + "\n")
    xy_file.write(xy_line)
    indexes.append(file_offset)

    file_write_counter += 1
    file_offset += len(xy_line)

    # log
    if file_write_counter % 1000000 == 0:
      p = (file_write_counter / total_lines) * 100.
      mil = file_write_counter / 1000000
      print("... written {:.1f}% ({} millions)".format(p, mil))

  # write VOCAB
  vocab = "".join(list(vocab))
  print("vocab=", vocab)
  vocab_file.write(vocab)

  # write INDEXES
  torch.save(indexes, fn_indexes_pt)
  
  x_file.close()
  y_file.close()
  xy_file.close()
  vocab_file.close()
  
  print("done. ({} lines written)".format(file_write_counter))

def merge_all_modules():
  modules = os.listdir(DATASET_DIR)

  print("Starting to merge {} modules.".format(len(modules)))
  target_dir = os.path.join(DATASET_DIR, ALL_MODULE_NAME)
  if os.path.exists(target_dir):
    print("Merge module {} already exists?!".format(target_dir))
    print("Exiting.")
    sys.exit(1)

  os.makedirs(target_dir)

  print("Merging extrapolate ...")
  read = get_read_streams(name="extrapolate", modules=modules)
  write = get_write_stream(name="extrapolate")
  random_merge(read, write)

  print("Merging interpolate ...")
  read = get_read_streams(name="interpolate", modules=modules)
  write = get_write_stream(name="interpolate")
  random_merge(read, write)

  print("Merging train ...")
  read = get_read_streams(name="train", modules=modules)
  write = get_write_stream(name="train")
  random_merge(read, write)

if __name__ == "__main__":
    merge_all_modules()