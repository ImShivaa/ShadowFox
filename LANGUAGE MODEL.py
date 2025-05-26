import torch
from transformers import BertTokenizer, BertForMaskedLM, BertForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

# Check if GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pretrained BERT tokenizer and models
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Masked Language Model for masked word prediction
mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(device)

# Sequence Classification model (fine-tuned on sentiment for demo)
cls_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# Function: Masked word prediction
def masked_word_prediction(text, mask_token="[MASK]"):
    # Encode input text with mask token
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    mask_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

    with torch.no_grad():
        outputs = mlm_model(input_ids)
        logits = outputs.logits

    mask_word_logits = logits[0, mask_index, :]
    top_tokens = torch.topk(mask_word_logits, 5, dim=1).indices[0].tolist()

    print(f"Original text: {text}")
    print(f"Top 5 predictions for the masked word:")
    for token in top_tokens:
        word = tokenizer.decode([token])
        print(f"  {word}")

# Example: Predict masked word
masked_word_prediction("The capital of France is [MASK].")

# Function: Sentiment classification (demo with dummy data)
def sentiment_classification(sentences):
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    outputs = cls_model(**inputs)
    logits = outputs.logits
    preds = torch.argmax(logits, axis=1).cpu().numpy()
    return preds

# Demo sentences & labels (0=negative, 1=positive)
sentences = [
    "I love this product! It's fantastic.",
    "This is the worst experience I have ever had.",
    "The movie was okay, not great but not terrible.",
    "I am extremely happy with the service.",
    "I do not recommend this at all."
]
true_labels = [1, 0, 0, 1, 0]

preds = sentiment_classification(sentences)
print("\nSentiment classification results:")
for sent, pred in zip(sentences, preds):
    label = "Positive" if pred == 1 else "Negative"
    print(f"Sentence: {sent}\nPredicted Sentiment: {label}\n")

print("Classification Report:")
print(classification_report(true_labels, preds, target_names=["Negative", "Positive"]))

# Visualization: Token embeddings for a sentence
def visualize_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = mlm_model.bert(**inputs)
    embeddings = outputs.last_hidden_state.squeeze(0).detach().numpy()

    tokens = tokenizer.tokenize(text)
    print(f"Tokens: {tokens}")

    plt.figure(figsize=(10, 6))
    sns.heatmap(embeddings.T, xticklabels=tokens, cmap="viridis")
    plt.title("Token Embeddings Heatmap")
    plt.xlabel("Tokens")
    plt.ylabel("Embedding Dimensions")
    plt.show()

visualize_embeddings("Machine learning is fascinating.")

# You can add more exploration, e.g., attention visualization, extended classification, or fine-tuning

print("\nBERT Exploration & Analysis completed.")
