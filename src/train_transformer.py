import os
import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- SET REPRODUCIBILITY SEEDS ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# --- LOAD TOKENS & MODELS ROBUSTLY ---
def initialize_model_and_tokenizer(model_name="distilbert-base-uncased", num_labels=3):
    """
    Initializes DistilBERT tokenizer and model.
    Tries official Hugging Face Hub first, then falls back to mirror, and
    saves configuration files locally to allow complete offline inference later.
    """
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    # 1. Download/Load Tokenizer
    try:
        print(f"Attempting to download tokenizer {model_name} from official Hub...")
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Official tokenizer download failed: {e}. Trying mirror...")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    # 2. Download/Load Model Structure
    try:
        print(f"Attempting to download model weights {model_name} from official Hub...")
        model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    except Exception as e:
        print(f"Official model download failed: {e}. Trying mirror...")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    return tokenizer, model

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_len
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_transformer(data_path, out_dir):
    print("Initiating DistilBERT Fine-Tuning pipeline...")
    
    # Initialize components
    tokenizer, model = initialize_model_and_tokenizer()
    
    # Load and split cleaned dataset
    df = pd.read_csv(data_path)
    
    # Downsample active training dataset balanced by class to keep CPU training practical
    df_sample = df.groupby("label").sample(n=100, random_state=42) # 300 total samples
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df_sample["Cleaned_Review"].astype(str).tolist(),
        df_sample["label"].astype(int).tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df_sample["label"]
    )
    
    train_dataset = ReviewDataset(train_texts, train_labels, tokenizer, max_len=32)
    val_dataset = ReviewDataset(val_texts, val_labels, tokenizer, max_len=32)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    
    epochs = 3
    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        model.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Training Loss: {avg_train_loss:.4f}")
        
        # Validation epoch
        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_val_loss += loss.item()
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                correct_predictions += torch.sum(preds == labels).item()
                total_samples += labels.size(0)
                
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct_predictions / total_samples
        print(f"Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {val_acc * 100:.2f}%")
        
        # Save checkpoint if it outperforms previous best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print("Validation loss improved. Saving model weights and tokenizer configs locally...")
            os.makedirs(out_dir, exist_ok=True)
            # Save Hugging Face model and tokenizer offline
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            
    print("\nTraining completed successfully! Model files saved for offline inference.")

if __name__ == "__main__":
    train_transformer("data/processed/dataset_cleaned.csv", "models/distilbert")
