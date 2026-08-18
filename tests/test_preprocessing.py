"""
tests/test_preprocessing.py — Tests du parsing HTML (BeautifulSoup).
Lancement : python -m unittest tests.test_preprocessing -v
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import preprocessing as P


class TestHtmlToTextAndLinks(unittest.TestCase):
    def test_plain_text_no_links(self):
        out = P.html_to_text_and_links("Bonjour, ceci est un message simple.")
        self.assertEqual(out["links"], [])
        self.assertIn("message simple", out["text"])

    def test_extracts_href_and_visible_text(self):
        html = '<p>Cliquez <a href="http://evil.tk/x">ici</a> pour continuer.</p>'
        out = P.html_to_text_and_links(html)
        self.assertEqual(len(out["links"]), 1)
        self.assertEqual(out["links"][0]["href"], "http://evil.tk/x")
        self.assertEqual(out["links"][0]["text"], "ici")

    def test_detects_mismatched_link(self):
        html = '<a href="http://evil-site.tk/x">https://paypal.com/login</a>'
        out = P.html_to_text_and_links(html)
        self.assertTrue(out["links"][0]["mismatch"])

    def test_matching_link_not_flagged(self):
        html = '<a href="https://paypal.com/login">https://paypal.com/login</a>'
        out = P.html_to_text_and_links(html)
        self.assertFalse(out["links"][0]["mismatch"])

    def test_link_without_url_text_not_flagged_as_mismatch(self):
        # texte du lien = "cliquez ici" (pas une URL) -> pas de mismatch au sens strict
        html = '<a href="http://evil.tk/x">cliquez ici</a>'
        out = P.html_to_text_and_links(html)
        self.assertFalse(out["links"][0]["mismatch"])

    def test_ignores_anchor_without_href(self):
        html = '<a name="top">Haut de page</a><a href="http://ok.com">Suite</a>'
        out = P.html_to_text_and_links(html)
        self.assertEqual(len(out["links"]), 1)


if __name__ == "__main__":
    unittest.main()
