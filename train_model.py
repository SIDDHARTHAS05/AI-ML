from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate

import time
from tqdm import tqdm

for i in tqdm(range(100), desc="Processing Items"):
    time.sleep(0.1)  # Simulating work


# File paths for training and validation data
train_file = r"D:\BITS Pilani Sem 3\NLP Applications\VS-Code 1\grammar_error_correction\data\train_data.csv"
val_file = r"D:\BITS Pilani Sem 3\NLP Applications\VS-Code 1\grammar_error_correction\data\val_data.csv"

# Load datasets
print("Loading datasets...")
data_files = {"train": train_file, "validation": val_file}
raw_datasets = load_dataset("csv", data_files=data_files)

# Load tokenizer and model
model_name = "t5-small"  # Change to "t5-base" or another model if required
print(f"Loading pre-trained model and tokenizer: {model_name}...")
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Preprocessing function
def preprocess_function(examples):
    inputs = ["gec: " + text for text in examples["input_text"]]
    targets = examples["target_text"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length").input_ids
    model_inputs["labels"] = labels
    return model_inputs

# Preprocess datasets
print("Tokenizing datasets...")
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=["input_text", "target_text"])

# Define training arguments
training_args = TrainingArguments(
    output_dir="./grammar_model",         # Directory to save model
    evaluation_strategy="epoch",         # Evaluate at the end of each epoch
    learning_rate=5e-5,                   # Learning rate
    per_device_train_batch_size=8,        # Batch size for training
    per_device_eval_batch_size=8,         # Batch size for evaluation
    num_train_epochs=3,                   # Number of training epochs
    weight_decay=0.01,                    # Weight decay
    logging_dir="./logs",                 # Directory for logs
    logging_steps=200,                    # Log every 200 steps
    save_strategy="epoch",                # Save model after each epoch
    load_best_model_at_end=True,          # Load the best model at the end
    metric_for_best_model="eval_loss",    # Metric for determining the best model
    save_total_limit=2,                   # Keep only the last 2 checkpoints
)

# BLEU metric for evaluation
metric = evaluate.load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Compute BLEU score
    result = metric.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
    return {"bleu": result["score"]}

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
print("Starting training...")
trainer.train()

# Save the trained model
print("Saving the trained model...")
trainer.save_model("./grammar_model")
tokenizer.save_pretrained("./grammar_model")

print("Training complete.")
