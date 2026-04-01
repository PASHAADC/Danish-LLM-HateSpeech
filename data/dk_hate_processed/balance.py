"""
Script to extract a balanced dataset from dkhate_complete.csv
Extracts 1000 samples with label=0 and 1000 samples with label=1
"""

import pandas as pd
import os
from pathlib import Path

def extract_balanced_dataset(input_file, output_file, samples_per_class=1000):
    """
    Extract a balanced dataset with equal number of samples per label.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the balanced dataset
        samples_per_class: Number of samples to extract per label (default: 1000)
    """
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found!")
        return False
    
    try:
        # Read the CSV file
        print(f"Reading data from {input_file}...")
        df = pd.read_csv(input_file)
        
        # Display basic information about the dataset
        print(f"Total samples in dataset: {len(df)}")
        print(f"Label distribution:")
        print(df['label'].value_counts().sort_index())
        
        # Check if we have enough samples for each label
        label_counts = df['label'].value_counts()
        
        if 0 not in label_counts or label_counts[0] < samples_per_class:
            print(f"Warning: Not enough samples with label=0 (found: {label_counts.get(0, 0)})")
            samples_per_class = min(samples_per_class, label_counts.get(0, 0))
            
        if 1 not in label_counts or label_counts[1] < samples_per_class:
            print(f"Warning: Not enough samples with label=1 (found: {label_counts.get(1, 0)})")
            samples_per_class = min(samples_per_class, label_counts.get(1, 0))
        
        # Extract samples for each label
        print(f"\nExtracting {samples_per_class} samples per label...")
        
        # Get samples with label=0
        label_0_samples = df[df['label'] == 0].sample(n=samples_per_class, random_state=42)
        
        # Get samples with label=1
        label_1_samples = df[df['label'] == 1].sample(n=samples_per_class, random_state=42)
        
        # Combine the samples
        balanced_df = pd.concat([label_0_samples, label_1_samples], ignore_index=True)
        
        # Shuffle the combined dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save to output file
        balanced_df.to_csv(output_file, index=False)
        
        print(f"\nBalanced dataset saved to: {output_file}")
        print(f"Total samples in balanced dataset: {len(balanced_df)}")
        print(f"Label distribution in balanced dataset:")
        print(balanced_df['label'].value_counts().sort_index())
        
        # Display sample statistics
        print(f"\nSample text lengths (characters):")
        balanced_df['text_length'] = balanced_df['text'].str.len()
        print(balanced_df.groupby('label')['text_length'].describe())
        
        return True
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

def main():
    # Define file paths relative to this script's location
    script_dir = Path(__file__).parent
    input_file = str(script_dir / "dkhate_complete.csv")

    # Create output filename with balanced suffix
    input_path = Path(input_file)
    output_file = input_path.parent / f"{input_path.stem}_balanced_1000.csv"
    
    # Extract balanced dataset
    success = extract_balanced_dataset(input_file, str(output_file), samples_per_class=1000)
    
    if success:
        print(f"\n✓ Successfully created balanced dataset!")
        print(f"Output file: {output_file}")
        
        # Optional: Display first few samples from each class
        print("\n" + "="*50)
        print("Preview of extracted samples:")
        print("="*50)
        
        df = pd.read_csv(output_file)
        
        print("\nFirst 3 samples with label=0:")
        print("-"*30)
        for idx, row in df[df['label'] == 0].head(3).iterrows():
            text_preview = row['text'][:100] + "..." if len(row['text']) > 100 else row['text']
            print(f"ID: {row['id']}")
            print(f"Text: {text_preview}")
            print(f"Label: {row['label']}\n")
        
        print("\nFirst 3 samples with label=1:")
        print("-"*30)
        for idx, row in df[df['label'] == 1].head(3).iterrows():
            text_preview = row['text'][:100] + "..." if len(row['text']) > 100 else row['text']
            print(f"ID: {row['id']}")
            print(f"Text: {text_preview}")
            print(f"Label: {row['label']}\n")

if __name__ == "__main__":
    main()