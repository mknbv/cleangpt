"""Loads a random article from SEP and preprocesses it for training."""
from functools import partial
import os
import requests
from bs4 import BeautifulSoup
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from cleangpt.bpe import make_encoder


URL = "https://plato.stanford.edu/cgi-bin/encyclopedia/random"
DATA_PATH = "../data/"


def get_page(url=URL, data_path=DATA_PATH, timeout=1):
  """Returns the contents of a SEOP page."""
  if not url.startswith("https://plato.stanford.edu/"):
    raise ValueError(f"invalid url; expected a SEOP page, got {url}")
  if not os.path.isdir(data_path):
    os.makedirs(data_path)
  response = requests.get(url, timeout=timeout)
  response.raise_for_status()
  soup = BeautifulSoup(response.text, "html.parser")
  content = str(soup.find("div", id="aueditable"))
  if data_path is not None:
    filepath = os.path.join(data_path, response.url.split("/")[-2] + ".html")
    with open(filepath, "w", encoding="utf8") as outfile:
      outfile.write(content)
  return content


def most_recent_file(dirpath=DATA_PATH):
  """Returns the most recent file from the directory path."""
  maxtime = 0
  result = None
  for name in os.listdir(dirpath):
    path = os.path.join(dirpath, name)
    if os.path.isfile(path) and (newtime := os.path.getctime(path)) > maxtime:
      result = name
      maxtime = newtime
  return result


def tokenize(text, inverse=False, special_chars=("—",)):
  """Character-level tokenization of text."""
  if inverse:
    chars = ({i: chr(i) for i in range(128)}
             | {128 + i: ch for i, ch in enumerate(special_chars)})
    return ''.join(chars[i] for i in text)
  chars = ({chr(i): i for i in range(128)}
           | {ch: 128 + i for i, ch in enumerate(special_chars)})
  return [chars[ch] for ch in text if ch in chars]


class TextDataset(Dataset):
  """Dataset of tokens."""
  def __init__(self, tokens, seqlen, filename=None, decode=None):
    self.tokens = tokens
    self.seqlen = seqlen
    self.filename = filename
    self.decode = decode

  @classmethod
  def from_text(cls, text, **kwargs):
    """Creates an instance from the given text."""
    return cls(tokenize(text), **kwargs)

  @classmethod
  def from_page(cls, url=URL, encoder=None, **kwargs):
    """Creates an instance from random SEOP page."""
    content = get_page(url)
    filename = most_recent_file()
    if encoder is None:
      return cls.from_text(content, filename=filename, **kwargs)
    return cls(encoder.encode(content), filename=filename,
               decode=encoder.decode, **kwargs)

  def __len__(self):
    return len(self.tokens) - self.seqlen

  def __getitem__(self, index):
    return (torch.tensor(self.tokens[index : index + self.seqlen]),
            torch.tensor(self.tokens[index + 1 : index + self.seqlen + 1]))

  def to_loader(self, batch_size=128,
                sampler=partial(RandomSampler, replacement=True),
                num_workers=4, **kwargs):
    """Converts the dataset to DataLoader."""
    return DataLoader(self, batch_size=batch_size,
                      sampler=sampler(self),
                      num_workers=num_workers,
                      **kwargs)


def make_loader(url=URL, seqlen=128, batch_size=128,
                num_workers=4, **encoder_kwargs):
  """Creates a dataloader from a random SEOP webpage."""
  encoder = make_encoder(**encoder_kwargs)
  return TextDataset.from_page(
      url=url, seqlen=seqlen, encoder=encoder).to_loader(
          batch_size, num_workers=num_workers)
