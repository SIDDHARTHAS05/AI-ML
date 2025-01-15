from flask import Flask, request, jsonify, render_template
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os

# Initialize Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "./uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)  # Ensure the uploads folder exists

# Load trained model and tokenizer
model_path = "models\grammar_model"  # Path to your trained model directory
print("Loading model and tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Grammar correction function
def correct_grammar(input_text):
    """Correct grammar in the input text."""
    inputs = tokenizer("gec: " + input_text, return_tensors="pt", max_length=128, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=128, num_beams=4, early_stopping=True)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

@app.route("/")
def home():
    """Serve the homepage."""
    return render_template("index.html")

@app.route("/correct", methods=["POST"])
def correct_text():
    """Correct grammar for a single text input."""
    data = request.json
    input_text = data.get("text", "")
    if not input_text.strip():
        return jsonify({"error": "Input text is empty."}), 400

    corrected_text = correct_grammar(input_text)
    return jsonify({"corrected_text": corrected_text})

@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle file uploads for batch grammar correction."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "File name is empty."}), 400

    if not file.filename.endswith(".txt"):
        return jsonify({"error": "Invalid file type. Only .txt files are allowed."}), 400

    # Save the uploaded file
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Read the file and correct line by line
    corrected_lines = []
    with open(file_path, "r") as f:
        for line in f:
            corrected_line = correct_grammar(line.strip())
            corrected_lines.append(corrected_line)

    # Save the corrected lines to a new file
    corrected_file_path = os.path.join(app.config["UPLOAD_FOLDER"], "corrected_" + file.filename)
    with open(corrected_file_path, "w") as f:
        f.write("\n".join(corrected_lines))

    return jsonify({"corrected_file": corrected_file_path})

if __name__ == "__main__":
    app.run(debug=True)
