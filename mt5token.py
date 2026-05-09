# ==============================================================================
# SECTION 3: TOKENIZE THE CLEANED DATASET
# ==============================================================================
print("\nSTEP 3: Setting up the mt5-base tokenizer and preprocessing...")
from transformers import AutoTokenizer

# We are using the cleaned_dataset variable from the previous step
# (Assuming the script is run in a notebook where variables persist)
# If running as a standalone script, you'd pass the cleaned_dataset object.

# Using the stable and powerful mt5-base model
model_checkpoint = "google/mt5-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_function(examples):
    inputs = examples["text"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(text_target=examples["summary"], max_length=150, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# We apply the tokenization to the 'cleaned_dataset'
tokenized_datasets = cleaned_dataset.map(preprocess_function, batched=True, load_from_cache_file=False)
print("\n✅ All dataset splits have been successfully tokenized.")