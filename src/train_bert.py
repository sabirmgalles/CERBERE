"""
train_bert.py — Fine-tuning d'un modèle Transformer (DistilBERT) pour la
classification binaire d'e-mails (phishing vs légitime), tête "NLP" de CERBÈRE.

Prérequis :
  pip install -r requirements.txt
  Un fichier data/text_dataset.csv avec les colonnes: text,label
  (label = 1 pour phishing, 0 pour légitime).
  -> Générez-le avec preprocessing.build_text_dataset_csv() à partir
     d'Enron-Spam (ham) + d'un corpus de phishing (ex. Nazario).

Usage :
  python src/train_bert.py --data data/text_dataset.csv --out models/text_model \
      --model distilbert-base-multilingual-cased --epochs 3

Le modèle multilingue est recommandé ici car le corpus cible (BIAT,
Ooredoo, etc.) mélange français et anglais.

Remarque : ce sandbox de développement n'a pas d'accès réseau vers
huggingface.co ni vers un GPU — ce script est prévu pour être exécuté sur
votre machine ou un environnement d'entraînement (Colab, serveur GPU, etc.).
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.enc = tokenizer(
            list(texts), truncation=True, padding=True, max_length=max_len
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/text_dataset.csv")
    ap.add_argument("--out", default="models/text_model")
    ap.add_argument("--model", default="distilbert-base-multilingual-cased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=256)
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(
            f"Introuvable : {args.data}\n"
            "Générez-le via preprocessing.build_text_dataset_csv() "
            "à partir d'Enron-Spam + d'un corpus de phishing. Voir README.md."
        )

    df = pd.read_csv(args.data).dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    print(f"Jeu de données : {len(df)} e-mails "
          f"({(df.label == 1).sum()} phishing / {(df.label == 0).sum()} légitimes)")

    train_df, val_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["label"]
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_ds = EmailDataset(train_df["text"], train_df["label"], tokenizer, args.max_len)
    val_ds = EmailDataset(val_df["text"], val_df["label"], tokenizer, args.max_len)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.out, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Métriques finales (validation) :", metrics)

    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Modèle sauvegardé dans {args.out} — utilisez api.py pour le servir.")


if __name__ == "__main__":
    main()
