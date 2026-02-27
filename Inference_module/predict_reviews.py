%%writefile llm_inference_pipeline/predict_reviews.py

import torch
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

class SentimentInference:

    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

        self.model.to(self.device)
        self.model.eval()

        self.label_map = {0: "Negative", 1: "Positive"}

    def predict(self, text):

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = F.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()

        return {
            "text": text,
            "prediction": self.label_map[prediction],
            "confidence": round(confidence, 4)
        }

    def predict_batch(self, text_list):

        inputs = self.tokenizer(
            text_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = F.softmax(outputs.logits, dim=1)
        predictions = torch.argmax(probs, dim=1)

        results = []

        for i, text in enumerate(text_list):
            results.append({
                "text": text,
                "prediction": self.label_map[predictions[i].item()],
                "confidence": round(probs[i][predictions[i]].item(), 4)
            })

        return results

    def summarize_sentiment(self, reviews):

        results = self.predict_batch(reviews)

        positives = sum(1 for r in results if r["prediction"] == "Positive")
        negatives = len(results) - positives

        return {
            "total_reviews": len(reviews),
            "positive": positives,
            "negative": negatives,
            "positive_ratio": round(positives / len(reviews), 2)
        }