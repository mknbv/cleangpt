""" Defies test case for testing models written in pytorch. """
import random
from unittest import TestCase
import numpy as np
import numpy.testing as nt
import torch


def _np(tensor):
  """ Converts tensor to numpy array. """
  if isinstance(tensor, torch.Tensor):
    return tensor.cpu().detach().numpy()
  return tensor



class TorchTestCase(TestCase):
  """ Test case for testing code with models written in pytorch. """
  def reset_seeds(self):
    """Resets all the seeds for reproducible results."""
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

  def setUp(self):
    self.reset_seeds()

  # pylint: disable=invalid-name
  def assertEqual(self, first, second, msg=None):
    """Tests if values are equal."""
    if isinstance(first, torch.Tensor) or isinstance(second, torch.Tensor):
      if msg is None:
        nt.assert_equal(_np(first), _np(second))
      elif not np.all(_np(first) == _np(second)):
        raise AssertionError(msg)
      return
    super().assertEqual(first, second, msg=msg)

  def assertAllClose(self, actual, expected, rtol=1e-7, atol=0.):
    """ Checks that actual and expected arrays or torch tensors are equal. """
    nt.assert_allclose(_np(actual), _np(expected), rtol=rtol, atol=atol)
