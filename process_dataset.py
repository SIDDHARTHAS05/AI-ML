import dask.dataframe as dd
from nltk.tokenize import word_tokenize
import nltk
import re
from tqdm import tqdm
import logging
from multiprocessing import Pool
from pathlib import Path
import time
# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Download NLTK resources
nltk.download("punkt")

from tqdm import tqdm

for i in tqdm(range(100), desc="Processing Items"):
    time.sleep(0.1)  # Simulating work

# Local File paths
file_path = Path(r"D:\BITS Pilani Sem 3\NLP Applications\VS-Code 1\grammar_error_correction\data\C4_200M.tsv-00000-of-00010")
output_sample_file = Path(r"D:\BITS Pilani Sem 3\NLP Applications\VS-Code 1\grammar_error_correction\data\C4_200M_sampled.csv")
output_tokenized_file = Path(r"D:\BITS Pilani Sem 3\NLP Applications\VS-Code 1\grammar_error_correction\data\tokenized_corpus.txt")

# one can create path like this as well in code
# import os

# Get the current script's directory
# current_folder = os.path.dirname(os.path.abspath(__file__))

# Create a "data" subfolder path
# data_path = os.path.join(current_folder, "data")
# os.makedirs(data_path, exist_ok=True)  
# Create the folder if it doesn't exist

# Define input and processed data file paths
# INPUT_DATA_PATH = os.path.join(data_path, "C4_200M.tsv-00000-of-00010")
# PROCESSED_DATA_PATH = os.path.join(data_path, "processed_corpus.pkl")

# Print the paths for debugging
# print(f"Input Data Path: {INPUT_DATA_PATH}")
# print(f"Processed Data Path: {PROCESSED_DATA_PATH}")



# Sampling fraction
sampling_fraction = 0.001

# Define text preprocessing function
def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", "", text)      # Remove special characters
    return text

# Load dataset
logger.info("Loading dataset with Dask...")
try:
    ddf = dd.read_csv(file_path, sep="\t", header=None)
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    exit(1)

# Sample dataset
logger.info("Sampling dataset...")
sampled_ddf = ddf.sample(frac=sampling_fraction, random_state=42)

# Save sampled dataset
logger.info(f"Saving sampled dataset to '{output_sample_file}'...")
sampled_ddf = sampled_ddf.compute()
sampled_ddf.columns = ["id", "text"]
sampled_ddf.to_csv(output_sample_file, index=False)

# Tokenization
logger.info("Tokenizing the sampled dataset...")
def preprocess_and_tokenize(text):
    preprocessed_text = preprocess_text(text)
    return word_tokenize(preprocessed_text.lower())

try:
    with Pool() as pool:
        tokenized_corpus = list(tqdm(pool.imap(preprocess_and_tokenize, sampled_ddf["text"].dropna().astype(str)), total=len(sampled_ddf)))
except Exception as e:
    logger.error(f"Error during tokenization: {e}")
    exit(1)

# Flatten tokenized corpus
tokenized_corpus = [token for tokens in tokenized_corpus for token in tokens]

# Save tokenized corpus
logger.info(f"Saving tokenized corpus to '{output_tokenized_file}'...")
chunk_size = 10000
with open(output_tokenized_file, "w") as f:
    for i in range(0, len(tokenized_corpus), chunk_size):
        f.write(" ".join(tokenized_corpus[i:i+chunk_size]) + "\n")

logger.info("Processing complete.")

