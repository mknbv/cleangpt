""" Layers unit testing. """
from unittest import skipIf
import torch
from torch import nn
from cleangpt import layers
from cleangpt.torch_test import TorchTestCase


HAVE_MINGPT = True
try:
  from mingpt.model import GPT, CausalSelfAttention
except ImportError:
  HAVE_MINGPT = False


class SoftmaxTest(TorchTestCase):
  """ Softmax layer tests. """
  def test_forward(self):
    """Forward method test."""
    inputs = torch.normal(0, 1, (2, 3, 5), dtype=torch.float64)
    actual = layers.Softmax().forward(inputs)
    expected = nn.Softmax(dim=-1).forward(inputs)
    self.assertAllClose(actual, expected)


class DropoutTest(TorchTestCase):
  """ Dropout layer tests. """
  # pylint: disable=invalid-name
  def test_forward(self):
    """Forward method test."""
    inputs = torch.normal(0, 1, (2, 3, 5), dtype=torch.float64)
    self.reset_seeds()
    actual = layers.Dropout(prob=0.2).forward(inputs)
    self.reset_seeds()
    expected = nn.Dropout(p=0.2).forward(inputs)
    self.assertAllClose(actual, expected)


class LinearTest(TorchTestCase):
  """ Linear layer test. """
  def test_forward(self):
    """Forward method test."""
    inputs = torch.normal(0, 1, (5, 3, 2))
    self.reset_seeds()
    actual = layers.Linear(2, 4).forward(inputs)
    self.reset_seeds()
    expected = nn.Linear(2, 4).forward(inputs)
    self.assertAllClose(actual, expected)


class AttentionTest(TorchTestCase):
  """ Attention layer test. """
  def tensor_assign(self, lhs, rhs):
    """Checks the shape and assigns to the tensor."""
    self.assertEqual(lhs.shape, rhs.shape)
    lhs.data = rhs.data

  def test_forward(self):
    """Forward method test."""
    inputs = torch.normal(0, 1, (3, 5, 4), dtype=torch.float64)

    attention_layer = layers.Attention(
        embedding_size=4, nheads=2,
        attn_pdrop=0.2,
    ).double()
    self.reset_seeds()
    actual  = attention_layer.forward(inputs, return_attn_weights=True)

    attention_nn = nn.MultiheadAttention(
        embed_dim=4,
        num_heads=2,
        dropout=0.2,
        batch_first=True,
    )
    self.tensor_assign(attention_nn.in_proj_weight,
                       attention_layer.input_linear.weight)
    self.tensor_assign(attention_nn.in_proj_bias,
                       attention_layer.input_linear.bias)
    self.tensor_assign(attention_nn.out_proj.weight,
                       attention_layer.output_linear.weight)
    self.tensor_assign(attention_nn.out_proj.bias,
                       attention_layer.output_linear.bias)

    attn_mask = nn.Transformer.generate_square_subsequent_mask(inputs.shape[1])
    self.reset_seeds()
    expected = attention_nn.forward(
        inputs, inputs, inputs,
        attn_mask=attn_mask.double(),
        is_causal=True,
        average_attn_weights=False,
    )

    self.assertAllClose(actual[1], expected[1])
    self.assertEqual(actual[1] == 0, expected[1] == 0)
    self.assertAllClose(actual[0], expected[0])

  @skipIf(not HAVE_MINGPT,
          "could not import mingpt; skipping attention layer test")
  def test_forward_mingpt(self):
    """Test forward method by comparing the results with mingpt layer."""
    inputs = torch.normal(0, 1, (1, 2, 4), dtype=torch.float64)
    attention_layer = layers.Attention(embedding_size=4,
                                       nheads=2,
                                       attn_pdrop=0.1).double()
    config = GPT.get_default_config()
    config.block_size = 2
    config.n_embd = 4
    config.n_head = 2
    config.attn_pdrop = 0.1
    config.resid_pdrop = 0.
    mingpt_attention = CausalSelfAttention(config).double()

    self.tensor_assign(mingpt_attention.c_attn.weight,
                       attention_layer.input_linear.weight)
    self.tensor_assign(mingpt_attention.c_attn.bias,
                       attention_layer.input_linear.bias)
    self.tensor_assign(mingpt_attention.c_proj.weight,
                       attention_layer.output_linear.weight)
    self.tensor_assign(mingpt_attention.c_proj.bias,
                       attention_layer.output_linear.bias)

    self.reset_seeds()
    actual = attention_layer(inputs)
    self.reset_seeds()
    expected = mingpt_attention(inputs)

    self.assertAllClose(actual, expected)


class LayerNormTest(TorchTestCase):
  """Layer normalization test."""
  def test_forward(self):
    """Test the forward method of the LayerNorm module."""
    inputs = torch.normal(0, 1, (2, 3, 5), dtype=torch.float64)
    actual = layers.LayerNorm((3, 5)).double().forward(inputs)
    expected = nn.LayerNorm((3, 5)).double().forward(inputs)
    self.assertAllClose(actual, expected)


class CrossEntropyWithTest(TorchTestCase):
  """Cross entropy module test."""
  def test_forward(self):
    inputs = torch.normal(0, 1, (2, 3, 5), dtype=torch.float64)
    labels = torch.randint(0, 3, (2, 5))
    actual = layers.CrossEntropy().forward(inputs, labels)
    expected = nn.CrossEntropyLoss().forward(inputs, labels)
    self.assertAllClose(actual, expected)
