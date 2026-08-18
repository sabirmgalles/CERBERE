"""
tests/test_heuristics.py — Tests unitaires du moteur de règles.
Ne dépend que de la stdlib : exécutable partout, y compris sans
torch/transformers/fastapi installés.

Lancement : python -m unittest tests.test_heuristics -v
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import heuristics as H


class TestLevenshtein(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(H.levenshtein("paypal", "paypal"), 0)

    def test_one_substitution(self):
        self.assertEqual(H.levenshtein("paypal", "paypa1"), 1)

    def test_accents_normalized(self):
        # 'sécurité' vs 'securite' ne doivent différer que par les accents,
        # neutralisés par _norm avant comparaison.
        self.assertEqual(H.levenshtein("sécurité", "securite"), 0)


class TestExtractDomain(unittest.TestCase):
    def test_from_url(self):
        self.assertEqual(H.extract_domain("https://www.biat.com.tn/login"), "www.biat.com.tn")

    def test_from_email(self):
        self.assertEqual(H.extract_domain("securite@paypal-alerte.tk"), "paypal-alerte.tk")

    def test_empty(self):
        self.assertEqual(H.extract_domain(""), "")


class TestIsIp(unittest.TestCase):
    def test_valid_ip(self):
        self.assertTrue(H.is_ip("185.23.11.9"))

    def test_domain_not_ip(self):
        self.assertFalse(H.is_ip("biat.com.tn"))


class TestAnalyzeUrlString(unittest.TestCase):
    def test_ip_url_flagged(self):
        r = H.analyze_url_string("http://185.23.11.9/login")
        self.assertGreater(r.score, 0)
        self.assertTrue(any("adresse IP" in i.txt for i in r.indicators))

    def test_typosquatting_detected(self):
        r = H.analyze_url_string("https://accounts-secure.paypa1.tk/login")
        self.assertGreater(r.score, 35)
        self.assertTrue(any("typosquatting" in i.txt for i in r.indicators))

    def test_legit_url_low_score(self):
        r = H.analyze_url_string("https://www.attijaribank.com.tn/particuliers")
        self.assertLess(r.score, 20)

    def test_suspicious_tld_flagged(self):
        r = H.analyze_url_string("https://mon-site-test.xyz/page")
        self.assertTrue(any(".xyz" in i.txt for i in r.indicators))

    def test_score_capped_at_100(self):
        # cumul volontairement extrême de tous les signaux
        r = H.analyze_url_string("http://user@185.23.11.9/a-b-c-d-e-secure-login-verify-account.tk")
        self.assertLessEqual(r.score, 100)


class TestAnalyzeText(unittest.TestCase):
    def test_urgency_words_detected(self):
        r = H.analyze_text("Action requise", "Votre compte sera suspendu, cliquez ici immédiatement.")
        self.assertGreater(r.score, 0)

    def test_sensitive_request_detected(self):
        r = H.analyze_text("", "Merci de confirmer votre mot de passe et votre numéro de carte bancaire.")
        self.assertTrue(any("informations sensibles" in i.txt for i in r.indicators))

    def test_neutral_text_low_score(self):
        r = H.analyze_text("Réunion de demain", "Bonjour, la réunion est déplacée à 15h en salle B. Merci.")
        self.assertEqual(r.score, 0)


class TestAnalyzeHeaders(unittest.TestCase):
    def test_brand_domain_mismatch(self):
        r = H.analyze_headers("Support PayPal", "securite@paypal-alerte.tk", "")
        self.assertGreater(r.score, 0)
        self.assertTrue(any("paypal" in i.txt.lower() for i in r.indicators))

    def test_reply_to_mismatch(self):
        r = H.analyze_headers("Service Client", "contact@biat.com.tn", "reponse@autredomaine.ru")
        self.assertTrue(any("réponse" in i.txt for i in r.indicators))

    def test_legit_brand_domain_not_flagged(self):
        # régression : le domaine officiel de la marque ne doit jamais être signalé
        r = H.analyze_headers("Support PayPal", "service@paypal.com", "")
        self.assertFalse(any("n'appartient pas" in i.txt for i in r.indicators))

    def test_consistent_headers_low_score(self):
        r = H.analyze_headers("Ooredoo Tunisie", "contact@ooredoo.tn", "")
        self.assertEqual(r.score, 0)


class TestVerdict(unittest.TestCase):
    def test_safe_threshold(self):
        self.assertEqual(H.verdict(10)["level"], "safe")

    def test_suspect_threshold(self):
        self.assertEqual(H.verdict(45)["level"], "suspect")

    def test_phishing_threshold(self):
        self.assertEqual(H.verdict(80)["level"], "phishing")

    def test_boundary_30(self):
        self.assertEqual(H.verdict(30)["level"], "suspect")

    def test_boundary_65(self):
        self.assertEqual(H.verdict(65)["level"], "phishing")


if __name__ == "__main__":
    unittest.main()
