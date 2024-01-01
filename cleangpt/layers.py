"""GPT layers."""
from math import sqrt
import torch
from torch import nn


def approx_gelu(inputs):
  """Applies approximate GELU activation function.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415.
  """
  return 0.5 * inputs * (
      1 + torch.tanh(
          sqrt(2 / torch.pi)
          * (inputs + 0.044715 * torch.pow(inputs, 3))
      )
  )


class ApproxGELU(nn.Module):
  """Apprximate GELU activation function.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415.
  """
  def forward(self, inputs):
    """Applyies GELU activation to the inputs."""
    return approx_gelu(inputs)


def softmax(logits, dim=-1):
  """Softmax across the specified dimension of inputs."""
  logits = logits - torch.max(logits, dim=dim, keepdims=True).values
  return (
      torch.exp(logits)
      / torch.sum(torch.exp(logits), dim=dim, keepdims=True)
  )


class Softmax(nn.Module):
  """Softmax across the specified dimension of inputs."""
  def __init__(self, dim=-1):
    super().__init__()
    self.dim = dim

  def forward(self, logits):
    """Applies softmax activation to the inputs."""
    return softmax(logits, self.dim)


def log_softmax(logits):
  """Computes log of the the softmax function."""
  logits = logits - torch.max(logits, dim=-1, keepdims=True).values
  return logits - torch.logsumexp(logits, dim=-1, keepdims=True)


class CrossEntropy(nn.Module):
  """Cross entropy with labels from logits."""
  def forward(self, logits, labels):
    """Computes cross entropy of the given logits wrt labels."""
    expected_labels_shape = logits.shape[:1] + logits.shape[2:]
    if labels.shape != expected_labels_shape:
      raise ValueError(f"{labels.shape=}, while expected "
                       f"shape {expected_labels_shape}")
    logits = torch.flatten(logits.transpose(1, -1), 0, -2)
    size = logits.shape[0]
    onehot = torch.zeros(logits.shape, device=logits.device)
    onehot[torch.arange(size), torch.flatten(labels)] = 1
    log_probs = log_softmax(logits)
    return -torch.mean(torch.sum(log_probs * onehot, -1))


class Dropout(nn.Module):
  """Tensor elements are zeroed with probability prob during training."""
  def __init__(self, prob=0.5):
    super().__init__()
    self.prob = prob

  def forward(self, inputs):
    """Applies dropout layer to the inputs."""
    if not self.training:
      return inputs
    probs = torch.full_like(inputs, 1 - self.prob)
    return inputs * torch.bernoulli(probs) / (1 - self.prob)


class Linear(nn.Module):
  """Linear transformation."""
  def __init__(self, input_features, output_features, bias=True):
    super().__init__()
    bound = 1 / sqrt(input_features)
    self.weight = nn.Parameter(
        torch.empty(output_features, input_features)
        .uniform_(-bound, bound)
    )
    self.bias = nn.Parameter(
        torch.empty(output_features).uniform_(-bound, bound)
    ) if bias else None

  def forward(self, inputs):
    """Returns the result of the linear transformation of the inputs."""
    return (torch.squeeze(self.weight @ inputs[..., None], -1)
            + (self.bias if self.bias is not None else 0))


class Attention(nn.Module):
  """Attention as described in the 'Attention is All You Need' paper.

  See https://arxiv.org/pdf/1706.03762.pdf
  """
  def __init__(self, embedding_size, nheads, attn_pdrop=0.1):
    super().__init__()
    if embedding_size % nheads != 0:
      raise ValueError(f"{embedding_size=} does not divide {nheads=}")
    self.input_linear = Linear(embedding_size, 3 * embedding_size)
    self.attn_dropout = Dropout(attn_pdrop)
    self.output_linear = Linear(embedding_size, embedding_size)
    self.nheads = nheads

  def forward(self, inputs, return_attn_weights=False):
    """Applies masked multi-head self attention to the inputs."""
    batch_size, seqlen, embedding_size = inputs.shape
    queries, keys, values = self.input_linear(inputs).split(
        embedding_size, -1)
    queries, keys, values = map(
        lambda t: t.reshape(batch_size, seqlen,
                            self.nheads, -1).transpose(1, 2),
        (queries, keys, values)
    )

    # (batch_size x nheads x seqlen x hidden_dim)
    # * (batch_size x nheads x hidden_dim x seqlen
    # = (batch_size x nheads x seqlen x seqlen)
    softmax_input = queries @ keys.transpose(-1, -2) / sqrt(keys.size(-1))
    mask = torch.tril(torch.ones(seqlen, seqlen)).to(softmax_input.device)
    softmax_input = softmax_input.masked_fill(mask == 0, -float('inf'))
    softmax_output = softmax(softmax_input)
    attn_weights = self.attn_dropout(softmax_output)

    # batch_size x nheads x seqlen x hidden_dim
    attn_output = attn_weights @ values
    output = self.output_linear(
        attn_output.transpose(1, 2)
        .reshape(batch_size, seqlen, embedding_size)
    )
    return (output, attn_weights) if return_attn_weights else output


class LayerNorm(nn.Module):
  """Layer normalization as is described in the paper.

  See https://arxiv.org/abs/1607.06450
  """
  def __init__(self, normalized_shape, eps=1e-5):
    super().__init__()
    if isinstance(normalized_shape, int):
      normalized_shape = (normalized_shape,)
    self.normalized_shape = normalized_shape
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(*normalized_shape))
    self.bias = nn.Parameter(torch.zeros(*normalized_shape))

  def forward(self, inputs):
    """Applies layer normalization to the inputs."""
    if inputs.shape[-len(self.normalized_shape):] != self.normalized_shape:
      raise ValueError(f"{inputs.shape=} does not match "
                       f"{self.normalized_shape=}")
    dims = tuple(i for i in range(-len(self.normalized_shape), 0))
    mean = inputs.mean(dims, keepdims=True)
    var = inputs.var(dims, unbiased=False, keepdims=True)
    return (
        (inputs - mean) / torch.sqrt(var + self.eps)
        * self.weight + self.bias
    )
