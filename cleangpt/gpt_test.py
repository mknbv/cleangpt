"""GPT module tests."""
from collections import namedtuple
from cleangpt import gpt
from cleangpt.torch_test import TorchTestCase


class GPTTest(TorchTestCase):
  """GPT module test."""
  def setUp(self):
    super().setUp()
    self.gpt = gpt.make(
        vocab_size=8,
        seqlen=10,
        embedding_size=4,
        nblocks=3,
        nheads=2,
        output_size=3,
    )

  def test_params(self):
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
