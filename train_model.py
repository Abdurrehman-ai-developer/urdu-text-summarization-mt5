# ==============================================================================
# FINAL TRAINING SCRIPT (mt5-base)
# ==============================================================================
import subprocess
import sys
import regex as re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import numpy as np
import evaluate

# ==============================================================================
# STEP 1: LOAD THE PREPARED DATA
# (This assumes the data is already cleaned and tokenized in a previous step.
# For a standalone script, we will redo the prep steps to be safe)
# ==============================================================================

# --- Define Cleaning Function ---
def clean_and_normalize_urdu(text):
    if not isinstance(text, str):
        return ""
    text = text.replace('یٰ', 'ی').replace('یٔ', 'ی').replace('ۓ', 'ے')
    text = text.replace('کٔ', 'ک').replace('گٔ', 'گ')
    text = text.replace('ۀ', 'ہ').replace('ۂ', 'ہ')
    diacritics_pattern = re.compile("[\u064B-\u0652]")
    text = re.sub(diacritics_pattern, "", text)
    urdu_pattern = re.compile(r'[^\u0600-\u06FF\s\d.,!؟]')
    text = re.sub(urdu_pattern, "", text)
    text = " ".join(text.split())
    return text

# --- Load and Clean ---
print("STEP 1: Loading and cleaning the full dataset...")
full_dataset = load_dataset('json', data_files={'train': 'urdu_train.jsonl', 'validation': 'urdu_val.jsonl'})
filtered_dataset = full_dataset.filter(lambda x: x['text'] and x['summary'] and not x['text'].isspace() and not x['summary'].isspace())
cleaned_dataset = filtered_dataset.map(
    lambda x: {'text': clean_and_normalize_urdu(x['text']), 'summary': clean_and_normalize_urdu(x['summary'])},
    load_from_cache_file=False
)
print("✅ Dataset cleaned.")

# --- Tokenize ---
print("\nSTEP 2: Tokenizing the cleaned dataset...")
model_checkpoint = "google/mt5-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_function(examples):
    inputs = examples["text"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(text_target=examples["summary"], max_length=150, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = cleaned_dataset.map(preprocess_function, batched=True, load_from_cache_file=False)
print("✅ Dataset tokenized.")


# ==============================================================================
# SECTION 3: TRAIN THE MODEL
# ==============================================================================
print("\nSTEP 3: Loading mt5-base model and starting the training process...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

args = Seq2SeqTrainingArguments(
    output_dir="./mt5_base_final_model_cleaned",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=5, # As requested
    predict_with_generate=True,
    fp16=False, # Set to False for maximum stability
    push_to_hub=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start the training
trainer.train()
print("\n\n✅ Training complete!")

# ==============================================================================
# SECTION 4: EVALUATE AND SAVE
# ==============================================================================
print("\nSTEP 4: Evaluating model performance with ROUGE scores...")
results = trainer.predict(tokenized_datasets["validation"])
decoded_preds = tokenizer.batch_decode(results.predictions, skip_special_tokens=True)
labels = np.where(results.label_ids != -100, results.label_ids, tokenizer.pad_token_id)
decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

rouge_metric = evaluate.load('rouge')
rouge_scores = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

for key, value in rouge_scores.items():
    rouge_scores[key] = round(value * 100, 4)

print("\n--- Final ROUGE Evaluation Results ---")
print(rouge_scores)
print("--------------------------------")

print("\nSTEP 5: Saving the final model and tokenizer...")
trainer.save_model("./mt5_base_final_model_cleaned/model_files")
tokenizer.save_pretrained("./mt5_base_final_model_cleaned/model_files")
print("\n✅ Model and tokenizer saved successfully.")