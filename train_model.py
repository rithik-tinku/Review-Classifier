import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- LOAD DATASET ---
dataset_path = "dataset.csv"
data = pd.read_csv(dataset_path)

# --- DETECT REVIEW COLUMN ---
review_column = None
possible_review_columns = [
    "Review",
    "review",
    "reviewText",
    "reviews.text",
    "text",
    "content"
]
for col in possible_review_columns:
    if col in data.columns:
        review_column = col
        break
if review_column is None:
    raise ValueError("No review column found in dataset.")

# --- DETECT RATING COLUMN ---
rating_column = None
possible_rating_columns = [
    "Rating",
    "rating",
    "overall",
    "score"
]
for col in possible_rating_columns:
    if col in data.columns:
        rating_column = col
        break
if rating_column is None:
    raise ValueError("No rating column found in dataset.")

# --- DATA CLEANING ---
data = data.dropna(subset=[review_column, rating_column])
texts = data[review_column].astype(str).tolist()
labels = data[rating_column].tolist()

# convert ratings (1–5) to 0–4 labels for cross-entropy
labels = [int(x) - 1 for x in labels]

# --- TRAIN TEST SPLIT ---
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42
)

# --- TOKENIZER ---
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# --- DATASET CLASS ---
class ReviewDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ReviewDataset(train_texts, train_labels)
val_dataset = ReviewDataset(val_texts, val_labels)

# --- DATALOADER ---
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# --- MODEL ---
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=5
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- OPTIMIZER ---
optimizer = AdamW(model.parameters(), lr=2e-5)

# --- TRAINING LOOP ---
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader)

    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        batch_labels = batch["labels"].to(device)

        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            labels=batch_labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_description(
            f"Epoch {epoch+1} Loss {loss.item():.4f}"
        )

    avg_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch+1} Average Loss: {avg_loss:.4f}")

# --- SAVE MODEL ---
torch.save(model.state_dict(), "bert_model.pth")
print("\nModel training completed.")
print("Model saved as bert_model.pth")