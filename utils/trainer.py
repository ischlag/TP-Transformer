import os
import random
import time

import torch
from tensorboardX import SummaryWriter

from utils.lib import compute_accuracy, assert_params_exist, calc_string_acc


class BasicSeq2SeqTrainer:
  def __init__(self, model,
               params,
               train_iterator,
               eval_iterator,
               optimizer,
               criterion,
               log):
    assert_params_exist(params, ["EOS",
                                 "SOS",
                                 "PAD",
                                 "max_abs_grad_norm",
                                 "log_every",
                                 "eval_every",
                                 "log_folder",
                                 "grad_accum_steps",
                                 "max_strikes",
                                 "eval_mode"])
    self.p = params
    self.model = model
    self.train_iterator = train_iterator
    self.eval_iterator = eval_iterator
    self.optimizer = optimizer
    self.criterion = criterion
    self.log = log
    if not params.eval_mode:
      self.train_writer = SummaryWriter(os.path.join(self.p.log_folder, "train"))
      self.eval_writer = SummaryWriter(os.path.join(self.p.log_folder, "eval"))
      self.model_save_path = os.path.join(self.p.log_folder, "best_eval_model.pt")
    else:
      self.train_writer = None
      self.eval_writer = None
      self.model_save_path = ""
    # train stats
    self.global_step = 0
    self.begin_time = time.time()
    # best stats
    self.best_eval_loss = float('inf')
    self.best_eval_acc = 0
    self.best_time = 0
    self.best_step = 0
    # gradient accum counter
    self.grad_accum_step = 0

    # reload model if necessary
    self.reload_existing_model()

  def reload_existing_model(self):
    if os.path.exists(self.model_save_path):
      self.log("best_eval_model.pt found. Do you want to load model? [Y/n]")

      if not self.p.force_reload:
        choice = input()
      else:
        choice = "y"

      if choice is "y" or choice is "Y":
        self.load_state()
        self.log("model loaded!")
      else:
        self.log("no model loaded.")

  def train(self, steps):
    self.log("training for {} steps ...".format(steps))
    self.model.train()
    self.begin_time = start_time = time.time()

    loss_sum, acc_sum, loss_counter = 0, 0, 0
    accum_loss, accum_acc = 0, 0
    self.optimizer.zero_grad()

    for idx, batch in enumerate(self.train_iterator):
      if idx >= steps * self.p.grad_accum_steps:
        break
      src = batch.src  # [batch_size, src_seq_len]
      trg = batch.trg  # [batch_size, trg_seq_len]

      src = src.to(self.p.device)
      trg = trg.to(self.p.device)

      logits = self.model(src, trg[:, :-1])
      # [batch_size, trg_seq_len-1, output_dim]

      flat_logits = logits.contiguous().view(-1, logits.shape[-1])
      # [batch_size * (trg_seq_len-1), output_dim]

      # ignore SOS symbol (skip first)
      flat_trg = trg[:, 1:].contiguous().view(-1)
      # [batch_size * (trg_seq_len-1)]

      # compute loss
      loss = self.criterion(flat_logits, flat_trg) / self.p.grad_accum_steps
      loss.backward()

      # compute acc
      acc = compute_accuracy(logits=logits,
                             targets=trg,
                             pad_value=self.p.PAD) / self.p.grad_accum_steps

      # store accumulation until it is time to step
      accum_loss += loss
      accum_acc += acc
      self.grad_accum_step += 1

      if self.grad_accum_step % self.p.grad_accum_steps == 0:
        # perform step
        torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
                                       max_norm=self.p.max_abs_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.grad_accum_step = 0

        # update loss and acc sum and counter
        loss_counter += 1
        loss_sum += accum_loss
        acc_sum += accum_acc

        # reset accum values
        accum_loss = 0
        accum_acc = 0

      else:
        # only grad accum step, skip global_step increment at the end
        continue

      if self.global_step % self.p.log_every == 0 and idx != 0 \
              and self.global_step != 0:
        # compute loggin metrics
        elapsed = time.time() - start_time
        steps_per_sec = loss_counter / elapsed
        start_time = time.time()
        avg_loss = loss_sum / loss_counter
        avg_acc = acc_sum / loss_counter
        loss_sum, acc_sum, loss_counter = 0, 0, 0

        # log on terminal
        self.log("{:4}: loss={:.4f} acc={:.4f} steps/sec={:.2f}"
              .format(self.global_step, avg_loss, avg_acc, steps_per_sec))

        # log in tensorboard
        if self.train_writer:
          self.train_writer.add_scalar("loss", avg_loss,
                                       global_step=self.global_step)
          self.train_writer.add_scalar("acc", avg_acc,
                                       global_step=self.global_step)
          self.train_writer.add_scalar("steps_per_sec", steps_per_sec,
                                       global_step=self.global_step)

      if self.global_step % self.p.eval_every == 0 and idx != 0 \
              and self.global_step != 0:
        self.evaluate(save_best_model=True)
        self.model.train()

      # kill experiment if there was no improvement
      if self.global_step - self.best_step > self.p.max_strikes:
        self.log("{} steps without improvement. Exiting."
            .format(self.p.max_strikes))
        break

      # kill experiment if there are NaNs
      if torch.isnan(loss):
        self.log("Loss has NaN values! Breaking training.")
        break

      self.global_step += 1

  def evaluate(self, save_best_model=False, iterator=None, write=True):
    if iterator is None:
      iterator = self.eval_iterator

    self.model.eval()
    loss_sum, acc_sum, count = 0, 0, 0
    start_time = time.time()

    with torch.no_grad():
      for idx, batch in enumerate(iterator):
        src = batch.src  # [batch_size, src_seq_len]
        trg = batch.trg  # [batch_size, trg_seq_len]

        src = src.to(self.p.device)
        trg = trg.to(self.p.device)

        logits = self.model(src, trg[:, :-1])
        # [batch_size, trg_seq_len-1, output_dim]

        flat_logits = logits.contiguous().view(-1, logits.shape[-1])
        # [batch_size * (trg_seq_len-1), output_dim]

        # ignore SOS symbol (skip first)
        flat_trg = trg[:, 1:].contiguous().view(-1)
        # [batch_size * (trg_seq_len-1)]

        loss_sum += self.criterion(flat_logits, flat_trg)

        acc_sum += compute_accuracy(logits=logits,
                                    targets=trg,
                                    pad_value=self.p.PAD)
        count += 1

      # compute loggin metrics
      elapsed = time.time() - start_time
      run_time = (time.time() - self.begin_time) / 60.
      batches_per_sec = elapsed / count
      avg_loss = loss_sum / count
      avg_acc = acc_sum / count

      # track best results
      if avg_loss < self.best_eval_loss:
        self.best_eval_loss = avg_loss
        self.best_eval_acc = avg_acc
        self.best_time = time.time()
        self.best_step = self.global_step
        if save_best_model:
          self.save_state()

      # log on terminal
      self.log("") if write else None
      self.log("eval: loss={:.4f} acc={:.4f} b/sec={:.2f} time={:.1f} min"
            .format(avg_loss, avg_acc, batches_per_sec, run_time))
      if write:
        self.log("run:  {}".format(self.p.log_folder))
        self.log("best: loss={:.4f} acc={:.4f} best since {} steps and {:.1f} min"
              .format(self.best_eval_loss,
                      self.best_eval_acc,
                      self.global_step - self.best_step,
                      (time.time() - self.best_time) / 60.))
        self.log("")

        if self.eval_writer:
          # log in tensorboard
          self.eval_writer.add_scalar("loss", avg_loss,
                                      global_step=self.global_step)
          self.eval_writer.add_scalar("acc", avg_acc,
                                      global_step=self.global_step)
    return avg_loss, avg_acc

  def save_state(self):
    state = {
      "global_step": self.global_step,
      "model": self.model.state_dict(),
      "optimizer": self.optimizer.state_dict(),
      "best_eval_loss": self.best_eval_loss,
      "best_eval_acc": self.best_eval_acc,
      "best_step": self.best_step
    }
    torch.save(obj=state, f=self.model_save_path)

  def load_state(self, save_path=None):
    if save_path is None:
      save_path = self.model_save_path
    state = torch.load(save_path)
    self.model.load_state_dict(state["model"])
    self.optimizer.load_state_dict(state["optimizer"])
    self.global_step = state["global_step"]
    self.best_eval_loss = state["best_eval_loss"]
    self.best_eval_acc = state["best_eval_acc"]
    self.best_step = state["best_step"]


  def evaluate_greedy(self):

    # put model into eval mode
    self.model.eval()

    infer_raw_match = 0
    infer_raw_count = 0
    start_time = time.time()
    infer_count = int(len(self.eval_iterator)*self.p.infer_percent)

    with torch.no_grad():
      for idx, batch in enumerate(self.eval_iterator):
        src = batch.src  # [batch_size, src_seq_len]
        trg = batch.trg  # [batch_size, trg_seq_len]

        if idx >= infer_count:
            break

        # run inference on a specified percent of eval data (costly)
        for i, (x, y) in enumerate(zip(src, trg)):
            y_hat = self.model.greedy_inference(x, max_len=len(y),
                                                sos_idx=self.p.SOS,
                                                eos_idx=self.p.EOS)
            match = calc_string_acc(y, y_hat, self.p.PAD)

            # # ensure we have same results with teacher_forcing eval
            # # ensure we have same results with teacher_forcing eval
            # xx = x.unsqueeze(0)
            # yy = y.unsqueeze(0)
            # logits = self.model(xx, yy[:, :-1])
            # yy_hat = torch.argmax(logits, dim=-1)      # get index predictions from logits
            # match2 = calc_string_acc(y, yy_hat[0], self.p.PAD)
            # assert(match == match2)

            infer_raw_match += match
            infer_raw_count += 1

      # compute metrics for this call
      elapsed = time.time() - start_time
      greedy_acc = infer_raw_match / infer_raw_count

      # log on terminal
      self.log("")
      self.log(
        "greedy-acc: {:.4f}, greedy-percent{}, infer-elapsed={:.2f} min" \
            .format( greedy_acc, self.p.infer_percent, elapsed / 60))
      self.log("")

      # log in tensorboard
      self.eval_writer.add_scalar("greedy-acc", greedy_acc,
                                  global_step=self.global_step)

  def show_sample(self, vocab, source="train"):
    if source is "train":
      it = iter(self.train_iterator)
    elif source is "eval":
      it = iter(self.eval_iterator)
    else:
      raise AttributeError("Invalid source string {}".format(source))
    batches = [next(it) for _ in range(10)]

    batch = random.sample(batches, k=1)[0]
    idx = random.randint(0, batch.src.shape[0] - 1)

    src = batch.src  # [batch_size, src_seq_len]
    trg = batch.trg  # [batch_size, trg_seq_len]

    logits = self.model(src, trg[:, :-1])
    # [batch_size, trg_seq_len-1, output_dim]

    preds = torch.argmax(logits, dim=-1)
    # [batch_size, trg_seq_len-1]

    def arr_to_str(arr):
      arr = arr.cpu().numpy()
      return "".join([vocab.itos[j] for j in arr])\
               .replace("<pad>", "") \
               .replace("<sos>", "")

    correct = calc_string_acc(y=trg[idx],
                              y_hat=preds[idx],
                              pad_value=self.p.PAD)

    text = "Evaluation Sample:\n" \
           "  input:  {}\n" \
           "  target: {}\n" \
           "  pred:   {}\n" \
           "  [{}]".format(arr_to_str(src[idx]),
                           arr_to_str(trg[idx]),
                           arr_to_str(preds[idx]),
                           "CORRECT" if correct else "WRONG")
    self.log(text)

