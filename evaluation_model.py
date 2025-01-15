from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
import evaluate

# Paths to the trained model and test data
model_path = "./grammar_model"  # Path to your trained model directory
test_file = r"D:\BITS Pilani Sem 3\NLP Applications\VS-Code 1\grammar_error_correction\data\val_data.csv"  # Replace with your test dataset path

# Load trained model and tokenizer
print("Loading trained model and tokenizer...")
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

# Load test dataset
print("Loading test dataset...")
test_dataset = load_dataset("csv", data_files={"test": test_file})["test"]

# Preprocess the test dataset
def preprocess_function(examples):
    inputs = ["gec: " + text for text in examples["input_text"]]
    targets = examples["target_text"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, return_tensors="pt", padding="max_length")
    return {"input_ids": model_inputs.input_ids, "attention_mask": model_inputs.attention_mask, "labels": targets}

print("Preprocessing test dataset...")
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

# BLEU metric for evaluation
metric = evaluate.load("sacrebleu")

# Evaluate on test dataset
def evaluate_model_on_dataset(dataset):
    print("Evaluating on test dataset...")
    decoded_preds = []
    decoded_labels = []

    for example in dataset:
        input_ids = example["input_ids"]
        attention_mask = example["attention_mask"]

        # Generate predictions
        outputs = model.generate(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0), max_length=128, num_beams=4, early_stopping=True)
        decoded_pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        decoded_label = example["labels"]

        decoded_preds.append(decoded_pred)
        decoded_labels.append(decoded_label)

    # Compute BLEU score
    result = metric.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
    print(f"BLEU Score: {result['score']}")
    return result

# Run evaluation on the test dataset
results = evaluate_model_on_dataset(tokenized_test_dataset)
print("Evaluation complete.")

# Evaluate custom sentences
def test_on_custom_sentences(sentences):
    print("Evaluating custom sentences...")
    inputs = ["gec: " + sentence for sentence in sentences]
    model_inputs = tokenizer(inputs, return_tensors="pt", max_length=128, truncation=True, padding="max_length")

    # Generate predictions
    outputs = model.generate(input_ids=model_inputs.input_ids, attention_mask=model_inputs.attention_mask, max_length=128, num_beams=4, early_stopping=True)
    corrected_sentences = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    for original, corrected in zip(sentences, corrected_sentences):
        print(f"Original: {original}")
        print(f"Corrected: {corrected}")
    return corrected_sentences

# Example test sentences
custom_sentences = [
    "She go to school yesterday.",
    "I has a pen.",
    "He don't know nothing.",
]

# Test the model on custom sentences
test_on_custom_sentences(custom_sentences)
