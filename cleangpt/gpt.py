"""GPT implementation."""
from math import sqrt
import torch
from torch import nn
from cleangpt import layers


class Block(nn.Module):
  """A building block of the GPT model."""
  def __init__(self, embedding_size, out_pdrop=0.1, **attention_kwargs):
    super().__init__()
    self.first = nn.Sequential(
        layers.LayerNorm(embedding_size),
        layers.Attention(embedding_size=embedding_size, **attention_kwargs),
        layers.Dropout(out_pdrop),
    )
    self.second = nn.Sequential(
        layers.LayerNorm(embedding_size),
        layers.Linear(embedding_size, 4 * embedding_size),
        layers.ApproxGELU(),
        layers.Linear(4 * embedding_size, embedding_size),
        layers.Dropout(out_pdrop),
    )

  def projection_weights(self):
    """Iterates over the 'projection' weights of this block."""
    yield self.first[1].output_linear.weight
    yield self.second[3].weight

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

  @property
  def vocab_size(self):
    """Vocabulary size of the embedding."""
    return self.input_embedding.num_embeddings

  @property
  def seqlen(self):
    """Expected sequence length of the embedding."""
    return self.position_embedding.num_embeddings

  def forward(self, inputs):
    """Comptues and returns the embeddings of the inputs."""
    _, seqlen = inputs.shape
    if seqlen != self.position_embedding.num_embeddings:
      raise ValueError(
          "expected inputs.shape[-1]="
          f"{self.position_embedding.num_embeddings}, "
          f"got {inputs.shape=}"
      )
    positions = torch.arange(
        seqlen, device=self.position_embedding.weight.device)
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
    self.weight_init()

  @property
  def vocab_size(self):
    """Vocabulary size."""
    return self.embedding.vocab_size

  @property
  def seqlen(self):
    """Input sequence length."""
    return self.embedding.seqlen

  @classmethod
  def make(cls, vocab_size, seqlen, embedding_size,
           embedding_pdrop=0.1, **init_kwargs):
    """Creates an instance with the default embedding module."""
    embedding = Embedding(vocab_size, seqlen,
                          embedding_size, embedding_pdrop)
    return cls(embedding, embedding_size=embedding_size, **init_kwargs)

  def projection_weights(self):
    """Iterates over the 'projection' weights of this GPT."""
    for block in self.blocks:
      yield from block.projection_weights()

  def weight_init(self, mean=0., std=0.02):
    """Weight initialization for a submodule of this GPT."""
    def init(module):
      if isinstance(module, layers.Linear):
        nn.init.normal_(module.weight, mean=mean, std=std)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=mean, std=std)
      elif isinstance(module, layers.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
      elif (len(list(module.children())) == 0
            and len(list(module.parameters())) > 0):
        raise ValueError(f"unexpected module {module}")

    self.apply(init)
    nblocks = len(self.blocks)
    for weight in self.projection_weights():
      nn.init.normal_(weight, mean=mean, std=std / sqrt(2 * nblocks))

  def forward(self, inputs):
    """Returns the result of processing the inputs with this model."""
    embedding_outputs = self.embedding(inputs)
    block_outputs = self.blocks(embedding_outputs)
    return self.output(block_outputs)

  def param_groups(self, weight_decay=0.1):
    """Creates parameter groups for the optimizer."""
    decayed = set()
    non_decayed = set()
    nparams = 0

    def group(module):
      nonlocal decayed, non_decayed, nparams
      if isinstance(module, layers.Linear):
        decayed.add(module.weight)
        if module.bias is not None:
          non_decayed.add(module.bias)
      elif isinstance(module, nn.Embedding):
        non_decayed.add(module.weight)
      elif isinstance(module, layers.LayerNorm):
        non_decayed.add(module.weight)
        non_decayed.add(module.bias)
      elif (len(list(module.children())) == 0
            and len(list(module.parameters())) > 0):
        raise ValueError(f"unexpected module {module}")

    self.apply(group)
    return [
        {"params": list(decayed), "weight_decay": weight_decay},
        {"params": list(non_decayed), "weight_decay": 0.}
    ]

  @torch.no_grad
  def generate(self, tokens, num_new_tokens,
               temperature=1., sample=False,
               topk=None):
    """Generates tokens after the inputs using the model."""
    for _ in range(num_new_tokens):
      inputs = tokens[:, -self.seqlen:]
      logits = self(inputs)[:, -1, :] / temperature
      if topk is not None:
        topk_logits, _ = torch.topk(logits, topk, dim=-1)
        logits[
            logits < topk_logits.min(dim=-1, keepdim=True).values
        ] = -float('inf')
      probs = layers.softmax(logits)
      if sample:
        new_tokens = torch.multinomial(probs, num_samples=1)
      else:
        new_tokens = torch.max(logits, dim=-1, keepdims=True).indices
      tokens = torch.cat([tokens, new_tokens], 1)
    return tokens


def configs(config_name=None):
  """Returns the dictionary of default configurations."""
  if config_name is not None:
    all_configs = configs()
    return all_configs[config_name]
  return {
      "nano": dict(nblocks=3, nheads=3, embedding_size=48),
      "micro": dict(nblocks=4, nheads=4, embedding_size=128),
      "mini": dict(nblocks=6, nheads=6, embedding_size=192),
  }


def make(vocab_size, seqlen, config_name=None, **kwargs):
  """Creates a GPT instance."""
  if config_name is not None:
    kwargs = configs(config_name) | kwargs
    if "output_size" not in kwargs:
      kwargs["output_size"] = vocab_size
  return GPT.make(vocab_size=vocab_size, seqlen=seqlen, **kwargs)
