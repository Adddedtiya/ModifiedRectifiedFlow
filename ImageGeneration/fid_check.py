"""Calculate FID and IS of CIFAR-10 samples with torch-fidelity.

The samples are the `samples_*.npz` files that `run_lib_pytorch.evaluate` writes.
The reference is the CIFAR-10 training set.
"""

import glob
import os
import re

from absl import app
from absl import flags
import numpy as np
import torch
from torch.utils.data import Dataset
import torch_fidelity

FLAGS = flags.FLAGS

flags.DEFINE_string("samples_dir", None, "Folder with the samples_*.npz files.")
flags.DEFINE_integer("num_samples", 50000, "Number of samples for the metrics.")
flags.DEFINE_integer("batch_size", 64, "Batch size for the Inception network.")
flags.mark_flags_as_required(["samples_dir"])


class SamplesDataset(Dataset):
  """Samples as uint8 tensors in CHW order."""

  def __init__(self, samples):
    self.samples = samples

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    return torch.from_numpy(self.samples[idx]).permute(2, 0, 1)


def load_samples(samples_dir, num_samples):
  files = glob.glob(os.path.join(samples_dir, "samples_*.npz"))
  files = sorted(files, key=lambda f: int(re.search(r"samples_(\d+)\.npz$", f).group(1)))
  samples = np.concatenate([np.load(f)["samples"] for f in files], axis=0)
  if samples.dtype != np.uint8 or samples.ndim != 4 or samples.shape[-1] != 3:
    raise ValueError(f"Samples must be uint8 with the shape (N, H, W, 3). Found {samples.dtype} {samples.shape}.")
  if len(samples) < num_samples:
    raise ValueError(f"Found {len(samples)} samples in {len(files)} files. {num_samples} are necessary.")
  return samples[:num_samples]


def main(argv):
  samples = load_samples(FLAGS.samples_dir, FLAGS.num_samples)
  print("Samples:", samples.shape)
  metrics = torch_fidelity.calculate_metrics(
    input1=SamplesDataset(samples),
    input2="cifar10-train",
    datasets_root=os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_data"),
    cuda=torch.cuda.is_available(),
    batch_size=FLAGS.batch_size,
    fid=True,
    isc=True,
  )
  print("FID:", metrics["frechet_inception_distance"])
  print("IS:", metrics["inception_score_mean"], "+/-", metrics["inception_score_std"])


if __name__ == "__main__":
  app.run(main)
