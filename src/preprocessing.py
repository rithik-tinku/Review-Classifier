import re
import pandas as pd

def preprocess_text(text):
    """
    Cleans review text using standard NLP normalization techniques.
    Processes:
      1. Converts text to lowercase.
      2. Strips HTML tags.
      3. Removes URLs.
      4. Normalizes whitespaces.
      5. Handles punctuation (retains standard alphanumeric characters and spaces).
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove HTML tags
    text = re.sub(r'<[^>]*>', ' ', text)
    
    # 3. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # 4. Punctuation handling (keep standard letters, numbers, and basic punctuation spacing)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"]', ' ', text)
    
    # 5. Normalize multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_dataset(input_csv, output_csv):
    """
    Loads raw CSV dataset, applies text preprocessing, maps ratings to
    3-class sentiment labels, and writes the clean dataset.
    """
    print(f"Loading raw dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Preprocess text
    print("Applying text normalization...")
    df["Cleaned_Review"] = df["Review"].apply(preprocess_text)
    
    # Map ratings 1-5 to 3 classes (0: Negative, 1: Neutral, 2: Positive)
    rating_map = {
        1: 0, 2: 0,  # Negative
        3: 1,        # Neutral
        4: 2, 5: 2   # Positive
    }
    df["label"] = df["Rating"].map(rating_map)
    
    # Filter empty reviews after preprocessing
    df = df[df["Cleaned_Review"] != ""]
    
    print(f"Saving preprocessed dataset to {output_csv}...")
    df.to_csv(output_csv, index=False)
    print("Dataset preprocessing complete!")

if __name__ == "__main__":
    preprocess_dataset("data/raw/dataset.csv", "data/processed/dataset_cleaned.csv")
