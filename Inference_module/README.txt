%%writefile llm_inference_pipeline/README.md
# Sentiment Inference Pipeline

This module loads a fine-tuned DistilBERT model for sentiment classification.

## Usage

from predict_reviews import SentimentInference

model_path = "path_to_saved_model"

inference = SentimentInference(model_path)

result = inference.predict("This product is amazing!")
print(result)