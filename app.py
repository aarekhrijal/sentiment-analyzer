import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokz = AutoTokenizer.from_pretrained('./sentiment_model')
model = AutoModelForSequenceClassification.from_pretrained('./sentiment_model')

def predict(text):
    inputs = tokz(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    return {
        "Negative": float(probs[0]),
        "Positive": float(probs[1])
    }

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=5, placeholder="Enter a movie review..."),
    outputs=gr.Label(num_top_classes=2),
    title="Sentiment Analyzer",
    description="Enter a movie review to analyze its sentiment"
)

demo.launch()
