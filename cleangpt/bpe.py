"""Bytes-pair-encoding and decoding of text.

This implementation is based on
https://github.com/karpathy/minGPT/blob/master/mingpt/bpe.py
"""
import json
import os
from pathlib import Path
import regex as re
import requests


CACHE_PATH = (
    Path(os.path.dirname(os.path.realpath(__file__))).parent
    / "data" / ".cache"
)
ENCODER_URL = """
https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json
""".strip()
VOCAB_URL = """
https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe
""".strip()


def bytes_to_unicode():
  """
  Every possible byte (really an integer 0..255) gets mapped by OpenAI to a
  unicode character that represents it visually. Some bytes have their
  appearance preserved because they don't cause any trouble. These are
  defined in list bs. For example: chr(33) returns "!", so in the returned
  dictionary we simply have d[33] -> "!".  However, chr(0), for example, is
  '\x00', which looks ugly. So OpenAI maps these bytes, into new characters
  in a range where chr() returns a single nice character.  So in the final
  dictionary we have d[0] -> 'Ā' instead, which is just chr(0 + 2**8).  In
  particular, the space character is 32, which we can see by ord(' ').
  Instead, this function will shift space (32) by 256 to 288, so d[32] ->
  'Ġ'.  So this is just a simple one-to-one mapping of bytes 0..255 into
  unicode characters that "look nice", either in their original form, or a
  funny shifted character like 'Ā', or 'Ġ', etc.
  """
  # The 188 integers that render fine in their original form
  # and need no shifting
  byteslist = (
      list(range(ord('!'), ord('~') + 1))
      + list(range(ord('¡'), ord('¬') + 1))
      + list(range(ord("®"), ord("ÿ") + 1))
  )
  charslist = [chr(b) for b in byteslist]
  # now get the representations of the other 68 integers that do need
  # shifting each will get mapped chr(256 + n), where n will grow from
  # 0...67 in the loop
  counter = 0
  for b in range(2 ** 8):
    if b not in byteslist:
      # if this byte is "ugly" then map it to the next available "nice"
      # character
      byteslist.append(b)
      charslist.append(chr(2 ** 8 + counter))
      counter += 1
  return dict(zip(byteslist, charslist))


def get_pairs(word):
  """Return consecutive pairs as a set of tuples in word."""
  return list(zip(word, word[1:]))


class BytesPairEncoder:
  """Bytes-pair-encoding encoder/decoder."""
  def __init__(self, encoder, merges):
    self.byte_encoder = bytes_to_unicode()
    self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
    self.encoder = encoder
    self.decoder = {v: k for k, v in self.encoder.items()}

    # merge list that defines the bpe "tree", of tuples (a,b)
    # that are to merge to token ab
    self.merges = dict(zip(merges, range(len(merges))))

    # the splitting pattern used for pre-tokenization
    # Should haved added re.IGNORECASE so BPE merges can happen for capitalized
    # versions of contractions <-- original openai comment

    # Python re reference: https://docs.python.org/3/library/re.html
    # - the vertical bars | is OR, so re.findall will chunkate text as the
    #   pieces match, from left to right
    # - '\'s' would split up things like Andrej's -> (Andrej, 's)
    # - ' ?\p{L}': optional space followed by 1+ unicode code points
    #   in the category "letter"
    # - ' ?\p{N}': optional space followed by 1+ unicode code points
    #   in the category "number"
    # - ' ?[^\s\p{L}\p{N}]+': optional space, then 1+ things that are
    #   NOT a whitespace, letter or number
    # - '\s+(?!\S)': 1+ whitespace characters (e.g. space or tab or etc)
    #   UNLESS they are followed by non-whitespace
    # So this will consume whitespace characters in a sequence but exclude
    # the last whitespace in that sequence. That last whitespace has the
    # opportunity to then match the optional ' ?' in earlier patterns.
    # - '\s+': 1+ whitespace characters, intended probably to catch a full
    # trailing sequence of whitespaces at end of string
    # So TLDR:
    # - we are special casing a few common apostrophe constructs ('s, 't, 're,
    # ...) and making those into separate tokens
    # - we then separate out strings into consecutive chunks of 1) letters, 2)
    # numbers, 3) non-letter-numbers, 4) whitespaces
    self.pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| """
                              r"""?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    self.cache = {}

  def replace(self, elements, replacement):
    """Replaces replacement pair in elements by it's concatenation."""
    first, second = replacement
    result = []
    i = 0
    while i < len(elements):
      try:
        j = elements.index(first, i)
      except ValueError:
        j = len(elements)
      result.extend(elements[i:j])
      i = j

      if i < len(elements):
        merge = i + 1 < len(elements) and elements[i + 1] == second
        result.append(''.join(replacement) if merge else elements[i])
        i += 1 + merge
    return result

  def merge(self, token):
    """ Iteratively merges BPE tokens."""
    if token in self.cache:
      return self.cache[token]

    elements = tuple(token)
    while len(elements) > 1:
      pairs = get_pairs(elements)
      replacement = min(pairs, key=lambda p: self.merges.get(p, float('inf')))
      if replacement not in self.merges:
        break
      elements = self.replace(elements, replacement)

    # concat all words into a string, and use ' ' as the separator. Note that
    # by now all characters have been byte encoded, guaranteeing that ' ' is
    # not used in the actual data and is a 'special' delimiter character
    result = ' '.join(elements)
    self.cache[token] = result
    return result

  def encode(self, text):
    """ string goes in, list of integers comes out """
    result = []
    for token in re.findall(self.pattern, text):
      # encode the token as a bytes (b'') object
      token_bytes = token.encode('utf-8')
      # translate all bytes to their unicode string representation and flatten
      token_translated = ''.join(self.byte_encoder[b] for b in token_bytes)
      # perform all the applicable bpe merges according to self.bpe_ranks
      token_merged = self.merge(token_translated).split(' ')
      # translate all bpe tokens to integers
      indices = [self.encoder[bpe_token] for bpe_token in token_merged]
      # extend our running list of all output integers
      result.extend(indices)
    return result

  def encode_and_show_work(self, text):
    """Debugging function, same as encode but returns all intermediate work."""
    encoding = []
    parts = []
    tokens = re.findall(self.pattern, text)
    for token in tokens:
      token_bytes = token.encode('utf-8')
      token_translated = ''.join(self.byte_encoder[b] for b in token_bytes)
      token_merged = self.merge(token_translated).split(' ')
      indices = [self.encoder[bpe_token] for bpe_token in token_merged]
      encoding.extend(indices)
      parts.append({
          "token": token,
          "token_bytes": token_bytes,
          "token_translated": token_translated,
          "token_merged": token_merged,
          "token_encoding": indices,
      })
    out = {
        "encoding": encoding,
        "tokens": tokens,
        "parts": parts,
    }
    return out

  def decode(self, encoding):
    """ list of integers comes in, string comes out """
    bytestr = ''.join(self.decoder[token] for token in encoding)
    tokens_bytes = bytearray([self.byte_decoder[c] for c in bytestr])
    return tokens_bytes.decode('utf-8', errors='replace')


def get_file(url, filepath, timeout=1):
  """Downloads from url to filepath if filepath does not exist."""
  if not os.path.isfile(filepath):
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    with open(filepath, "wb") as outfile:
      outfile.write(response.content)


def make_encoder(encoder_filepath=os.path.join(CACHE_PATH, "encoder.json"),
                 vocab_filepath=os.path.join(CACHE_PATH, "vocab.bpe"),
                 nmerged=None):
  """Returns an instance of the GPT BPE Encoder/Decoder."""
  os.makedirs(CACHE_PATH, exist_ok=True)

  # load encoder.json that has the raw mappings from token -> bpe index
  encoder_filepath = os.path.join(CACHE_PATH, "encoder.json")
  get_file(ENCODER_URL, encoder_filepath)
  with open(encoder_filepath, "rb") as infile:
    encoder = json.load(infile)
  # 256 individual byte tokens, 50,000 merged tokens,
  # and 1 special <|endoftext|> token
  assert len(encoder) == 50257, len(encoder)

  # load vocab.bpe that contains the bpe merges, i.e. the bpe tree structure
  # in the form tuples (a, b), that indicate that (a, b) is to be merged to
  # one token ab
  vocab_filepath = os.path.join(CACHE_PATH, "vocab.bpe")
  get_file(VOCAB_URL, vocab_filepath)
  with open(vocab_filepath, 'r', encoding="utf-8") as infile:
    bpe_data = infile.read()
  # light postprocessing: strip the version on first line and the last line is
  # a blank
  bpe_merges = [
      tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]
  ]
  assert len(bpe_merges) == 50000, len(bpe_merges)
  if nmerged is not None:
    bpe_merges = bpe_merges[:nmerged]
    assert list(encoder.values()) == list(range(len(encoder)))
    encoder = ({ch: i for ch, i in encoder.items() if i < 256}
               | {''.join(k): encoder[''.join(k)] for k in bpe_merges})

  return BytesPairEncoder(encoder, bpe_merges)
