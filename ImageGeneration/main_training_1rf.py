"""Train a 1-Rectified Flow model.

This file is a port of `main.py --mode train` and `run_lib_pytorch.train`.
"""

import os

import numpy as np
import logging
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import sde_lib
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.mark_flags_as_required(["workdir", "config"])


def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TensorBoard summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  os.makedirs(sample_dir, exist_ok=True)

  tb_dir = os.path.join(workdir, "tensorboard")
  os.makedirs(tb_dir, exist_ok=True)
  writer = tensorboard.SummaryWriter(tb_dir)

  # Initialize model.
  score_model = mutils.create_model(config)
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  os.makedirs(checkpoint_dir, exist_ok=True)
  os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)
  # Resume training when intermediate checkpoints are detected
  state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
  initial_step = int(state['step'])

  # Build data iterators
  train_ds, eval_ds = datasets.get_pytorch_dataset(config)

  train_ds_loader = torch.utils.data.DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True, drop_last=True, num_workers=4)
  eval_ds_loader = torch.utils.data.DataLoader(eval_ds, batch_size=config.training.batch_size, shuffle=True, drop_last=True, num_workers=4)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs
  if config.training.sde.lower() == 'rectified_flow':
    sde = sde_lib.RectifiedFlow(init_type=config.sampling.init_type, noise_scale=config.sampling.init_noise_scale, use_ode_sampler=config.sampling.use_ode_sampler)
    sampling_eps = 1e-3
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting)
  eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting)

  # Building sampling functions
  if config.training.snapshot_sampling:
    sampling_shape = (config.training.batch_size, config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  num_train_steps = config.training.n_iters

  logging.info("Starting training loop at step %d." % (initial_step,))

  step = initial_step - 1
  while step <= (num_train_steps + 1):
    for _, data in enumerate(train_ds_loader):
        step += 1
        batch = data.to(config.device).float()
        batch = scaler(batch)
    
        # Execute one training step
        loss = train_step_fn(state, batch)
        if step % config.training.log_freq == 0:
          logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
          writer.add_scalar("training_loss", loss, step)

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
          save_checkpoint(checkpoint_meta_dir, state)

        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0:
          for _, eval_data in enumerate(eval_ds_loader):  
            break  
          eval_batch = eval_data.to(config.device).float()
          eval_batch = scaler(eval_batch)
          eval_loss = eval_step_fn(state, eval_batch)
          logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
          writer.add_scalar("eval_loss", eval_loss.item(), step)

        # Save a checkpoint periodically and generate samples if needed
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
          # Save the checkpoint.
          save_step = step // config.training.snapshot_freq
          save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

          # Generate and save samples
          if config.training.snapshot_sampling:
            ema.store(score_model.parameters())
            ema.copy_to(score_model.parameters())
            sample, n = sampling_fn(score_model)
            ema.restore(score_model.parameters())
            this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
            os.makedirs(this_sample_dir, exist_ok=True)
            nrow = int(np.sqrt(sample.shape[0]))
            image_grid = make_grid(sample, nrow, padding=2)
            sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
            with open(
                os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
              np.save(fout, sample)

            with open(
                os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
              save_image(image_grid, fout)


def main(argv):
  # Create the working directory
  os.makedirs(FLAGS.workdir, exist_ok=True)
  # Set logger so that it outputs to both console and file
  gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'w')
  handler = logging.StreamHandler(gfile_stream)
  formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
  handler.setFormatter(formatter)
  logger = logging.getLogger()
  logger.addHandler(handler)
  logger.setLevel('INFO')
  # Run the training pipeline
  if 'pytorch' in FLAGS.config.data.dataset.lower():
      train(FLAGS.config, FLAGS.workdir)
  else:
      raise ValueError(f"Dataset {FLAGS.config.data.dataset} is not supported. The name must contain 'Pytorch'.")


if __name__ == "__main__":
  app.run(main)
