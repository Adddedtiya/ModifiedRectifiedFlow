import torch
import os
import logging
import numpy as np

# Old checkpoints keep the learning rate as a NumPy float64 in the optimizer state.
NUMPY_SAFE_GLOBALS = [np._core.multiarray.scalar, (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
                      np.dtype, np.dtypes.Float64DType]


def restore_checkpoint(ckpt_dir, state, device, weights_only = True):
  if not os.path.exists(ckpt_dir):
    os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    with torch.serialization.safe_globals(NUMPY_SAFE_GLOBALS):
      loaded_state = torch.load(ckpt_dir, map_location=device, weights_only = weights_only)

    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    return state


def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step']
  }
  torch.save(saved_state, ckpt_dir)