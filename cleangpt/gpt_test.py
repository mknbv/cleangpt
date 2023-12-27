"""GPT module tests."""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from os import devnull
from itertools import zip_longest
from unittest import skipIf
import torch
from cleangpt import gpt
from cleangpt.torch_test import TorchTestCase

HAVE_MINGPT = True
try:
  from mingpt.model import GPT as MinGPT
except ImportError:
  HAVE_MINGPT = False


@contextmanager
def suppress_stdout_stderr():
  """A context manager that redirects stdout and stderr to devnull"""
  with open(devnull, 'w', encoding='utf8') as fnull:
    with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
      yield (err, out)


class GPTTest(TorchTestCase):
  """GPT module test."""
  def make_gpt(self, **kwargs):
    """Construct GPT instance."""
    self.reset_seeds()
    return gpt.make(**self.kwargs | kwargs)

  def setUp(self):
    super().setUp()
    self.kwargs = dict(
        vocab_size=8,
        seqlen=10,
        embedding_size=4,
        nblocks=3,
        nheads=2,
        output_size=3,
        embedding_pdrop=0.1,
        attn_pdrop=0.1,
        out_pdrop=0.1,
    )
    self.gpt = self.make_gpt()

  def make_mingpt(self, **kwargs):
    """Creates and returns a mingpt instance."""
    kwargs = self.kwargs | kwargs
    config = MinGPT.get_default_config()
    config.model_type = None
    config.vocab_size = kwargs["vocab_size"]
    config.block_size = kwargs["seqlen"]
    config.n_embd = kwargs["embedding_size"]
    config.n_layer = kwargs["nblocks"]
    config.n_head = kwargs["nheads"]
    config.embd_pdrop = kwargs["embedding_pdrop"]
    config.attn_pdrop = kwargs["attn_pdrop"]
    config.resid_pdrop = kwargs["out_pdrop"]

    self.reset_seeds()
    with suppress_stdout_stderr():
      return MinGPT(config=config)

  def test_param_shapes(self):
    """Tests the number of parameters of the GPT instance."""
    nparams = sum(1 for _ in self.gpt.parameters())
    self.assertEqual(nparams, 2 + 12 * 3 + 2 + 1)
    shapes = [p.shape for p in self.gpt.parameters()]
    self.assertEqual(shapes, [
        (8, 4),               # vocab emb
        (10, 4),              # pos emb
        *[                    # block
          (4,), (4,),         # layer norm
          (12, 4), (12,),     # attention input layer
          (4, 4), (4,),       # attention output layer
          (4,), (4,),         # layer norm
          (16, 4), (16,),     # block.second first layer
          (4, 16), (4,),      # block.second second layer
        ] * 3,                # nblocks
        (4,), (4,),           # layer norm
        (3, 4),               # output
    ])

  def test_forward_shape(self):
    """Test the forward method of the GPT implementation."""
    inputs = torch.randint(0, 8, (5, 10))
    outputs = self.gpt(inputs)
    self.assertEqual(outputs.shape, (5, 10, 3))

  @skipIf(not HAVE_MINGPT,
          "skipping param test since mingpt is not installed")
  def test_params(self):
    """Tests parameter values."""
    mingpt = self.make_mingpt()
    self.gpt = self.make_gpt(output_size=8)
    pcount = 0
    for pact, pexp in zip_longest(self.gpt.named_parameters(),
                                  mingpt.named_parameters()):
      with self.subTest(actual_param=pact[0], expected_param=pexp[0]):
        self.assertAllClose(pact[1], pexp[1])
      pcount += 1
    self.assertEqual(pcount, 2 + 12 * 3 + 2 + 1)

  @skipIf(not HAVE_MINGPT,
          "skipping block forward outputs test since mingpt is not installed")
  def test_block_outputs(self):
    """Tests block level outputs."""
    inputs = torch.randn(5, 10, 4, dtype=torch.float64)
    mingpt_block = self.make_mingpt().transformer.h[0].double()
    gpt_block = self.make_gpt(output_size=8).blocks[0].double()
    self.reset_seeds()
    expected = mingpt_block(inputs)
    self.reset_seeds()
    actual = gpt_block(inputs)
    self.assertAllClose(actual, expected)

  @skipIf(not HAVE_MINGPT,
          "skipping gpt forward outputs test since mingpt is not installed")
  def test_forward_outputs(self):
    """Tests the outputs of the model."""
    inputs = torch.randint(0, 8, (5, 10))
    mingpt = self.make_mingpt().double()
    expected, _ = mingpt(inputs)

    self.gpt = self.make_gpt(output_size=8).double()
    actual = self.gpt(inputs)

    self.assertAllClose(actual, expected)

  def test_param_groups(self):
    """Tests parameter groups for optimization/weight decay."""
    param_groups = self.gpt.param_groups(weight_decay=0.01)

    self.assertEqual(len(param_groups), 2)
    intersection = (set(param_groups[0]["params"])
                    & set(param_groups[1]["params"]))
    self.assertFalse(intersection)

    all_params = dict(self.gpt.named_parameters())
    union = set(p for pg in param_groups for p in pg["params"])
    skipped = set(n for n, p in all_params.items() if p not in union)
    self.assertFalse(skipped)

    self.assertFalse(None in union)

    self.assertEqual(sum(len(pg["params"]) for pg in param_groups), 41)
