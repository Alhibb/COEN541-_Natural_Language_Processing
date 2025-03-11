# Import required libraries
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
from datasets import load_dataset, DatasetDict
import evaluate 
import numpy as np
import pandas as pd
import gradio as gr
from sklearn.metrics import accuracy_score, f1_score
import os 
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Define label mapping (based on hausa_voa_topics dataset)
label_map = {"Nigeria": 0, "Africa": 1, "World": 2, "Health": 3, "Politics": 4}
reverse_label_map = {v: k for k, v in label_map.items()}  

# Step 1: Load and preprocess the dataset
def load_and_preprocess_dataset():
    try:
        print("Loading dataset...")
        dataset = load_dataset("UdS-LSV/hausa_voa_topics")

        # Check dataset structure and handle missing splits
        print("Dataset structure:", dataset)

        if not isinstance(dataset, DatasetDict): 
            print("Error: Dataset is not a DatasetDict.  Check load_dataset.")
            return None

        if "validation" not in dataset or "test" not in dataset:
            print("Creating train/validation/test splits...")
            dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
            train_val_split = dataset["train"].train_test_split(test_size=0.1, seed=42)
            dataset["train"] = train_val_split["train"]
            dataset["validation"] = train_val_split["test"]
            dataset["test"] = dataset["test"]

        # Explore label distribution
        train_df = pd.DataFrame(dataset["train"])
        print("Label distribution in training set:\n", train_df["label"].value_counts())

        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Step 2: Tokenize the dataset
def tokenize_dataset(dataset):
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

        # Use 'news_title' instead of 'title' for tokenization
        def tokenize_function(examples):
            return tokenizer(examples["news_title"], padding="max_length", truncation=True, max_length=128)

        print("Tokenizing dataset...")
        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Map string labels to integers
        tokenized_dataset = tokenized_dataset.map(lambda x: {"labels": x["label"]})
        # Set format for PyTorch
        tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        return tokenized_dataset, tokenizer
    except Exception as e:
        print(f"Error tokenizing dataset: {e}")
        return None, None

# Step 3: Fine-tune the model
def train_model(tokenized_dataset, tokenizer):
    try:
        print("Loading pretrained model...")
        model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=5)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_dir="./logs",
            logging_steps=10,
        )

        # Define evaluation metrics using evaluate library
        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
            f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
            return {"accuracy": accuracy, "f1": f1}

        # Initialize Trainer
        print("Initializing trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        # Train the model
        print("Training model...")
        trainer.train()

        # Evaluate on test set
        print("Evaluating on test set...")
        test_results = trainer.evaluate(tokenized_dataset["test"])
        print("Test set results:", test_results)

        # Save the model
        print("Saving model...")
        trainer.save_model("./hausa_topic_classifier")

        return model, trainer
    except Exception as e:
        print(f"Error training model: {e}")
        return None, None

# Step 4: Build and launch Gradio application
def build_gradio_app():
    try:
        print("Loading fine-tuned model for inference...")

        # Check if the model directory exists
        if not os.path.exists("./hausa_topic_classifier"):
            print("Error: Model directory not found.  Train the model first.")
            return

        # Try loading pipeline with error handling
        try:
            classifier = pipeline("text-classification", model="./hausa_topic_classifier", tokenizer="xlm-roberta-base")
        except Exception as e:
            print(f"Error loading pipeline: {e}")
            return

        def classify_text(text):
            if not text.strip():
                return "Error: Please enter a valid Hausa news headline."
            try:
                result = classifier(text)[0]  
                # Extract the integer label.  Handles "LABEL_X" format.
                if isinstance(result["label"], str) and "LABEL_" in result["label"]:
                    predicted_label_int = int(result["label"].split("_")[-1])  
                else:
                    predicted_label_int = int(result["label"])
                predicted_label = reverse_label_map[predicted_label_int]
                return f"Predicted Topic: {predicted_label}"
            except Exception as e:
                return f"Prediction Error: {e}" 

        print("Building Gradio interface...")
        iface = gr.Interface(
            fn=classify_text,
            inputs=gr.Textbox(lines=2, placeholder="Enter Hausa news headline here..."),
            outputs="text",
            title="Hausa Topic Classification",
            description="Enter a Hausa news headline to classify its topic.",
        )

        print("Launching Gradio app...")
        iface.launch()
    except Exception as e:
        print(f"Error building Gradio app: {e}")

# Main function to run the entire pipeline
def main():
    print("Starting Hausa Topic Classification project...")

    # Step 1: Load and preprocess dataset
    dataset = load_and_preprocess_dataset()
    if dataset is None:
        return

    # Step 2: Tokenize dataset
    tokenized_dataset, tokenizer = tokenize_dataset(dataset)
    if tokenized_dataset is None or tokenizer is None:
        return

    # Step 3: Fine-tune model
    model, trainer = train_model(tokenized_dataset, tokenizer)
    if model is None or trainer is None:
        return

    # Step 4: Build and launch Gradio app
    build_gradio_app()

if __name__ == "__main__":
    main()