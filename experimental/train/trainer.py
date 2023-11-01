import os
import shutil
import json
import torch
import math
import numpy as np
from tqdm import tqdm
import wandb
from pathlib import Path


class PretrainTrainer:

    def __init__(self,
                 args,
                 model,
                 optimizer,
                 lr_scheduler,
                 dataloader,
                 logger,
                 accelerator,
                 tokenizer,
                 max_grad_norm=0.0,
                 from_checkpoint=None,
                ):

        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.dataloader = dataloader
        self.logger = logger
        self.completed_steps = 0
        self.accelerator = accelerator
        self.tokenizer = tokenizer
        self.max_grad_norm = max_grad_norm
        self._iter = iter(dataloader)
        self.from_checkpoint = from_checkpoint

    def _move_to_device(self, batch):
        for k in batch:
            batch[k] = batch[k].to(self.args.device)
        return batch

    def _save_model(self, save_path=None):
        if save_path is None:
            save_path = self.args.output_dir

        if self.accelerator.is_main_process:
           unwrapped_model = self.accelerator.unwrap_model(self.model)
           unwrapped_model.save_pretrained(save_path, save_function=self.accelerator.save)

    def _save_trained(self, save_path=None):
        if save_path is None:
            save_path = self.args.output_dir
        Path(save_path).mkdir(parents=True, exist_ok=True)
        torch.save(self.optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
        torch.save(self.lr_scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
        trainer_state = {
            "completed_steps": self.completed_steps,
            "best_metric": 0.0,
            "test_results": 0.0,
        }
        if self.accelerator.is_main_process:
            with open(os.path.join(save_path, "trainer_state.json"), "w") as f:
                json.dump(trainer_state, f)
        self._save_model(save_path=save_path)

    def evaluate(self):
        pass

    def _get_batch(self):
        try:
            batch = next(self._iter)
        except StopIteration:
            self._iter = iter(self.dataloader)
            batch = next(self._iter)

        batch = self._move_to_device(batch)

        if 'rank' in batch:
            batch.pop('rank')
        return batch

    def compute_loss(self):
        self.model.train()
        batch = self._get_batch()
        outputs = self.model(**batch)
        loss = outputs.loss
        loss = loss / self.args.gradient_accumulation_steps
        self.accelerator.backward(loss)
        return loss.item()

    def _prepare_from_checkpoint(self):

        if self.from_checkpoint is None:
            return

        state_file = os.path.join(self.from_checkpoint, "trainer_state.json")
        optim_file = os.path.join(self.from_checkpoint, "optimizer.pt")
        sched_file = os.path.join(self.from_checkpoint, "scheduler.pt")
        if os.path.exists(sched_file):
            sched_state = torch.load(sched_file)
            self.lr_scheduler.load_state_dict(sched_state)
        if not os.path.exists(state_file):
            return

        with open(state_file, "r") as f:
            state = json.load(f)
            self.pre_completed_steps = state["completed_steps"]
            self.best_metric = state["best_metric"]

        self.logger.info(f"pretrained steps: {self.pre_completed_steps}, best dev metric {self.best_metric}")
        self.accelerator.wait_for_everyone()

    def update(self, tr_loss, loss_step):
        if self.completed_steps % self.args.steps_to_log == 0:
            self.logger.info(
                "step {}, learning rate {}, average loss {}".format(
                    self.completed_steps,
                    self.optimizer.param_groups[0]["lr"],
                    tr_loss / loss_step
                )
            )
            if self.accelerator.is_main_process:
                wandb.log({'step': self.completed_steps, 'loss': tr_loss / loss_step,
                           'lr': self.optimizer.param_groups[0]["lr"]})
            tr_loss = 0.0
            loss_step = 0

        if self.completed_steps % self.args.steps_to_eval == 0:
            self.evaluate()

        if self.completed_steps % self.args.steps_to_save == 0:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self._save_trained(
                    save_path = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(self.completed_steps))
                )
                # delete outdated checkpoints
                for files in os.listdir(self.args.output_dir):
                    file_name = os.path.join(self.args.output_dir, files)
                    if os.path.isdir(file_name) and files.startswith('checkpoint-'):
                        checked_step = int(files[11:])
                        if self.completed_steps - checked_step >= self.args.max_ckpts_to_keep * self.args.steps_to_save:
                            if self.accelerator.is_main_process:
                                shutil.rmtree(file_name)

    def train(self):

        total_batch_size = self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * self.accelerator.num_processes
        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(self.dataloader.dataset)}")
        self.logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(self.args.max_train_steps))
        self.completed_steps = 0
        self.pre_completed_steps = 0
        self._prepare_from_checkpoint()
        tr_loss = 0.0
        loss_step = 0
        for step in range(self.args.max_train_steps * self.args.gradient_accumulation_steps):
            if self.completed_steps < self.pre_completed_steps:
                self._get_batch()
                if step % self.args.gradient_accumulation_steps == 0:
                    self.completed_steps += 1
                    progress_bar.update(1)
                continue
            if step % self.args.gradient_accumulation_steps == 0:
                tr_loss += self.compute_loss()
                loss_step += 1
                if self.max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)
                self.completed_steps += 1
                self.update(tr_loss, loss_step)
                tr_loss = 0.0
                loss_step = 0
            else:
                with self.accelerator.no_sync(self.model):
                    tr_loss += self.compute_loss()
                    loss_step += 1

        self.evaluate()
        self._save_trained(
            save_path = os.path.join(self.args.output_dir, 'final')
        )
