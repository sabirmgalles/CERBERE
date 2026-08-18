"""
tests/test_url_vocab.py — Tests de l'encodage caractère-niveau des URLs.
Ne dépend pas de torch (voir src/url_vocab.py).
Lancement : python -m unittest tests.test_url_vocab -v
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.url_vocab import encode_url, MAX_LEN, CHAR2IDX


class TestEncodeUrl(unittest.TestCase):
    def test_output_length_always_max_len(self):
        self.assertEqual(len(encode_url("http://a.com")), MAX_LEN)
        self.assertEqual(len(encode_url("")), MAX_LEN)
        self.assertEqual(len(encode_url("x" * 500)), MAX_LEN)

    def test_padding_is_zero(self):
        ids = encode_url("abc")
        self.assertEqual(ids[3:], [0] * (MAX_LEN - 3))

    def test_known_chars_encoded_nonzero(self):
        ids = encode_url("abc")
        self.assertTrue(all(i > 0 for i in ids[:3]))

    def test_case_insensitive(self):
        self.assertEqual(encode_url("ABC"), encode_url("abc"))

    def test_truncates_beyond_max_len(self):
        long_url = "http://" + "a" * 300
        ids = encode_url(long_url)
        self.assertEqual(len(ids), MAX_LEN)


if __name__ == "__main__":
    unittest.main()
