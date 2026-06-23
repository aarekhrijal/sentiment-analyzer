# Install libraries
!pip install -Uqq transformers datasets

# Load dataset
from datasets import load_dataset
dataset = load_dataset("imdb")

# Load tokenizer
from transformers import AutoTokenizer
model_nm = 'distilbert-base-uncased'
tokz = AutoTokenizer.from_pretrained(model_nm)

def tok_func(x):
    return tokz(x['text'], truncation=True, max_length=512)

tok_ds = dataset.map(tok_func, batched=True)

# Prepare training data (subset for faster training)
small_train = tok_ds['train'].shuffle(seed=42).select(range(2000))
small_test = tok_ds['test'].shuffle(seed=42).select(range(500))

# Load model
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=2)

# Setup metrics
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions)
    }

# Training configuration
from transformers import TrainingArguments, Trainer

args = TrainingArguments('outputs', learning_rate=2e-5, 
    eval_strategy="epoch", 
    per_device_train_batch_size=16, per_device_eval_batch_size=16, 
    num_train_epochs=3, weight_decay=0.01, report_to='none')

# Train
trainer = Trainer(model, args, train_dataset=small_train, eval_dataset=small_test,
    processing_class=tokz, compute_metrics=compute_metrics)

trainer.train()

# Save model
trainer.save_model('/kaggle/working/sentiment_model')
tokz.save_pretrained('/kaggle/working/sentiment_model')
