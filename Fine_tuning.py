import pandas as pd
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
from tqdm import tqdm

for i in tqdm(range(100), desc="Processing Items"):
    time.sleep(0.1)  # Simulating work
# File paths
input_file = r"D:\BITS Pilani Sem 3\NLP Applications\VS-Code 1\grammar_error_correction\data\C4_200M_sampled.csv"
output_train_file = r"D:\BITS Pilani Sem 3\NLP Applications\VS-Code 1\grammar_error_correction\data\train_data.csv"
output_val_file = r"D:\BITS Pilani Sem 3\NLP Applications\VS-Code 1\grammar_error_correction\data\val_data.csv"

# Load the sampled data
print("Loading the sampled dataset...")
df = pd.read_csv(input_file)

# Ensure the dataset has 'text' column
if "text" not in df.columns:
    raise ValueError("The input dataset must contain a 'text' column.")

# Simulate grammar errors
def introduce_errors(text):
    """Simulate grammar errors in the text."""
    if random.random() > 0.7:  # Only apply errors to 70% of sentences
        return text
    words = text.split()
    if len(words) > 3:
        random_index = random.randint(0, len(words) - 1)
        error_type = random.choice(["remove", "duplicate", "replace", "swap", "capitalize"])
        if error_type == "remove":
            words.pop(random_index)
        elif error_type == "duplicate":
            words.insert(random_index, words[random_index])
        elif error_type == "replace":
            words[random_index] = random.choice(["is", "the", "a", "an", "has", "goes", "not", "in", "or"])
        elif error_type == "swap" and random_index < len(words) - 1:
            words[random_index], words[random_index + 1] = words[random_index + 1], words[random_index]
        elif error_type == "capitalize":
            words[random_index] = words[random_index].capitalize() if random.random() > 0.5 else words[random_index].lower()
    return " ".join(words)

# Create input-output pairs
print("Generating input-output pairs...")
df["input_text"] = [introduce_errors(text) for text in tqdm(df["text"], desc="Simulating errors")]
df["target_text"] = df["text"]

# Drop unnecessary columns
df = df[["input_text", "target_text"]]

# Validate data
df.dropna(subset=["input_text", "target_text"], inplace=True)
df = df[df["input_text"].str.len() > 3]
df = df[df["target_text"].str.len() > 3]

# Split into train and validation sets
print("Splitting dataset into training and validation sets...")
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Save to CSV
print(f"Saving training data to '{output_train_file}'...")
train_df.to_csv(output_train_file, index=False)

print(f"Saving validation data to '{output_val_file}'...")
val_df.to_csv(output_val_file, index=False)

# Print dataset statistics
print("Dataset Statistics:")
print(f"Total samples: {len(df)}")
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

print("Data preparation complete.")
