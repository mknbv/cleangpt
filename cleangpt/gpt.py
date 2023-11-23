"""GPT implementation."""
import torch
from torch import nn
from cleangpt import layers


class Block(nn.Module):
  """A building block of the GPT model."""
  def __init__(self, embedding_size,
               out_pdrop=0.1, **attention_kwargs):
    super().__init__()
    self.first = nn.Sequential(
        layers.LayerNorm(embedding_size),
        layers.Attention(embedding_size=embedding_size, **attention_kwargs),
        layers.Dropout(out_pdrop),
    )
    self.second = nn.Sequential(
        layers.Linear(embedding_size, 4 * embedding_size),
        layers.ApproxGELU(),
        layers.Linear(4 * embedding_size, embedding_size),
        layers.Dropout(out_pdrop),
    )

  def forward(self, inputs):
    """Returns the result of applying the block to the inputs."""
    first = inputs + self.first(inputs)
    second = first + self.second(first)
    return second


class Embedding(nn.Module):
  """GPT embedding module."""
  def __init__(self, vocab_size, seqlen, embedding_size, pdrop=0.1):
    super().__init__()
    self.input_embedding = nn.Embedding(vocab_size, embedding_size)
    self.position_embedding = nn.Embedding(seqlen, embedding_size)
    self.dropout = layers.Dropout(pdrop)

  def forward(self, inputs):
    """Comptues and returns the embeddings of the inputs."""
    _, seqlen = inputs.shape
    if seqlen != self.position_embedding.num_embeddings:
      raise ValueError(
          f"expected inputs.shape={self.position_embedding.num_embeddings}, "
          f"got {inputs.shape=}"
      )
    positions = torch.arange(seqlen)
    return self.dropout(self.input_embedding(inputs)
                        + self.position_embedding(positions))


class GPT(nn.Module):
  """GPT model."""
  def __init__(self, embedding, embedding_size, nblocks,
               output_size, **block_kwargs):
    super().__init__()
    self.embedding = embedding
    self.blocks = nn.Sequential(*[
        Block(embedding_size=embedding_size, **block_kwargs)
        for _ in range(nblocks)
    ])
    self.output = nn.Sequential(
        layers.LayerNorm(embedding_size),
        layers.Linear(embedding_size, output_size, bias=False),
    )

  @classmethod
  def make(cls, vocab_size, seqlen, embedding_size,
           embedding_pdrop=0.1, **init_kwargs):
    """Creates an instance with the default embedding module."""
    embedding = Embedding(vocab_size, seqlen,
                          embedding_size, embedding_pdrop)
    return cls(embedding, embedding_size=embedding_size, **init_kwargs)

  def forward(self, inputs):
    """Returns the result of processing the inputs with this model."""
    embedding_outputs = self.embedding(inputs)
    block_outputs = self.blocks(embedding_outputs)
    return self.output(block_outputs)
