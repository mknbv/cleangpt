"""Bytes-pair-encoding tests."""
from unittest import TestCase
from cleangpt.bpe import make_encoder


class BPETest(TestCase):
  def test_encoding(self):
    text = "Hello!! I'm Mikhail Konobeev. It's 2023. w00t :D 🤗"
    encoder = make_encoder()

    result = encoder.encode_and_show_work(text)

    expected_tokens = [
        "Hello", "!!", " I", "'m", " Mikhail", " Konobeev", ".", " It", "'s",
        " 2023", ".", " w", "00", "t", " :", "D", " 🤗",
    ]
    self.assertEqual(result["tokens"], expected_tokens)

    expected_parts = [
        {
            "token": "Hello",
            "token_bytes": b"Hello",
            "token_translated": "Hello",
            "token_merged": ["Hello"],
            "token_encoding": [15496],
        },
        {
            "token": "!!",
            "token_bytes": b"!!",
            "token_translated": "!!",
            "token_merged": ["!!"],
            "token_encoding": [3228],
        },
        {
            "token": " I",
            "token_bytes": b" I",
            "token_translated": "ĠI",
            "token_merged": ["ĠI"],
            "token_encoding": [314],
        },
        {
            "token": "'m",
            "token_bytes": b"'m",
            "token_translated": "'m",
            "token_merged": ["'m"],
            "token_encoding": [1101],
        },
        {
            "token": " Mikhail",
            "token_bytes": b" Mikhail",
            "token_translated": "ĠMikhail",
            "token_merged": ["ĠMikhail"],
            "token_encoding": [42040],
        },
        {
            "token": " Konobeev",
            "token_bytes": b" Konobeev",
            "token_translated": "ĠKonobeev",
            "token_merged": ["ĠKon", "ob", "ee", "v"],
            "token_encoding": [17431, 672, 1453, 85],
        },
        {
            "token": ".",
            "token_bytes": b".",
            "token_translated": ".",
            "token_merged": ["."],
            "token_encoding": [13],
        },
        {
            "token": " It",
            "token_bytes": b" It",
            "token_translated": "ĠIt",
            "token_merged": ["ĠIt"],
            "token_encoding": [632],
        },
        {
            "token": "'s",
            "token_bytes": b"'s",
            "token_translated": "'s",
            "token_merged": ["'s"],
            "token_encoding": [338],
        },
        {
            "token": " 2023",
            "token_bytes": b" 2023",
            "token_translated": "Ġ2023",
            "token_merged": ["Ġ20", "23"],
            "token_encoding": [1160, 1954],
        },
        {
            "token": ".",
            "token_bytes": b".",
            "token_translated": ".",
            "token_merged": ["."],
            "token_encoding": [13],
        },
        {
            "token": " w",
            "token_bytes": b" w",
            "token_translated": "Ġw",
            "token_merged": ["Ġw"],
            "token_encoding": [266],
        },
        {
            "token": "00",
            "token_bytes": b"00",
            "token_translated": "00",
            "token_merged": ["00"],
            "token_encoding": [405],
        },
        {
            "token": "t",
            "token_bytes": b"t",
            "token_translated": "t",
            "token_merged": ["t"],
            "token_encoding": [83],
        },
        {
            "token": " :",
            "token_bytes": b" :",
            "token_translated": "Ġ:",
            "token_merged": ["Ġ:"],
            "token_encoding": [1058],
        },
        {
            "token": "D",
            "token_bytes": b"D",
            "token_translated": "D",
            "token_merged": ["D"],
            "token_encoding": [35],
        },
        {
            "token": " 🤗",
            "token_bytes": b" \xf0\x9f\xa4\x97",
            "token_translated": "ĠðŁ¤Ĺ",
            "token_merged": ["ĠðŁ", "¤", "Ĺ"],
            "token_encoding": [12520, 97, 245],
        },
    ]

    self.assertEqual(len(result["parts"]), len(expected_parts))
    for i, (actual, expected) in enumerate(zip(result["parts"],
                                               expected_parts)):
      with self.subTest(i=i):
        self.assertEqual(actual, expected)

    expected_encoding = [
        15496, 3228, 314, 1101, 42040, 17431, 672, 1453, 85, 13, 632, 338,
        1160, 1954, 13, 266, 405, 83, 1058, 35, 12520, 97, 245
    ]
    self.assertEqual(result["encoding"], expected_encoding)
