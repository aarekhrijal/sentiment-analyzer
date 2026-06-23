# Sentiment Analyzer

A sentiment classifier that determines if movie reviews are positive or negative, fine-tuned on the IMDB dataset.

## Approach
- Used IMDB movie review dataset (25,000 train, 25,000 test)
- Fine-tuned DistilBERT (distilbert-base-uncased) for sentiment classification
- Trained on 2,000 samples for 3 epochs

## Results
- Accuracy: 88.4%
- F1 Score: 0.884

## Tech Stack
- Python, Hugging Face Transformers, PyTorch
- Trained on Kaggle (free GPU)

## Files
- `notebook.py` - training code
- `app.py` - Gradio deployment interface
- `sentiment_model/` - trained model and tokenizer files

## Example Usage
```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="./sentiment_model")
classifier("This movie was fantastic!")
```
