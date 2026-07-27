import os
import pandas as pd
from datasets import load_dataset

def main():
    print("Initiating dataset acquisition...")
    
    # Try downloading from official HF Hub first
    try:
        print("Attempting download from official Hugging Face Hub...")
        dataset = load_dataset("Yelp/yelp_review_full", split="train")
    except Exception as e:
        print(f"Official download failed: {e}")
        print("Falling back to Hugging Face Mirror (https://hf-mirror.com)...")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        # Reloading datasets to pick up env change
        import urllib3
        urllib3.disable_warnings()
        dataset = load_dataset("Yelp/yelp_review_full", split="train")

    print("Yelp reviews dataset loaded successfully. Filtering and balancing...")
    df_raw = pd.DataFrame(dataset)
    
    # yelp_review_full labels are 0-indexed stars (0 to 4 corresponding to 1 to 5 stars)
    # Map raw labels (0-4) to ratings (1-5):
    df_raw["Rating"] = df_raw["label"] + 1
    df_raw["Review"] = df_raw["text"]
    
    # Group into Negative (1-2 stars), Neutral (3 stars), Positive (4-5 stars)
    neg_df = df_raw[df_raw["Rating"].isin([1, 2])]
    neu_df = df_raw[df_raw["Rating"] == 3]
    pos_df = df_raw[df_raw["Rating"].isin([4, 5])]
    
    # We target approximately 2,000 balanced samples:
    # 666 Negative, 666 Neutral, 668 Positive
    neg_sample = neg_df.sample(n=666, random_state=42)
    neu_sample = neu_df.sample(n=666, random_state=42)
    pos_sample = pos_df.sample(n=668, random_state=42)
    
    balanced_df = pd.concat([neg_sample, neu_sample, pos_sample]).sample(frac=1.0, random_state=42)
    
    # Retain only Review and Rating columns to match original dataset format
    final_df = balanced_df[["Review", "Rating"]]
    
    # Save raw dataset
    raw_path = "data/raw/dataset.csv"
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    final_df.to_csv(raw_path, index=False)
    print(f"Balanced dataset of {len(final_df)} samples successfully saved to {raw_path}!")
    
    # Also save a copy to the root directory for backward compatibility
    final_df.to_csv("dataset.csv", index=False)
    print("Backward compatibility dataset copy saved to root 'dataset.csv'.")

if __name__ == "__main__":
    main()
