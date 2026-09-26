# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
import os

def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


def get_pytorch_dataset(config):
    import pytorch_datasets as tds
    import torchvision.transforms as tr
    if config.data.dataset == 'CelebA-HQ-Pytorch':
        transform = tr.Resize(256)
        return tds.celeba_hq_dataset(config.training.data_dir, config.training.batch_size, transform), tds.celeba_hq_dataset(config.training.data_dir, config.training.batch_size, transform)
    elif config.data.dataset == 'AFHQ-CAT-Pytorch': 
        transform = tr.Resize(256)
        return tds.afhq_dataset(config.training.data_dir, config.training.batch_size, 'cat', transform), tds.afhq_dataset(config.training.data_dir, config.training.batch_size, 'cat', transform)
    elif config.data.dataset == 'Folder-Pytorch':
        transform = tr.Compose([tr.Resize(config.data.image_size), tr.CenterCrop(config.data.image_size)])
        return tds.folder_dataset(config.training.data_dir, transform), tds.folder_dataset(config.training.data_dir, transform)
    elif config.data.dataset == 'CIFAR10-Pytorch':
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_data')
        transform = [tr.ToTensor()]
        if config.data.random_flip:
            transform.append(tr.RandomHorizontalFlip())
        transform = tr.Compose(transform)
        return tds.cifar10_dataset(data_dir, True, transform), tds.cifar10_dataset(data_dir, False, transform)
    else:
        assert False, 'Not implemented'
