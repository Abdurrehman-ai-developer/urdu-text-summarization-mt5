# ==============================================================================
# SECTION 2: LOAD, FILTER, AND CLEAN THE DATASET
# ==============================================================================
print("\nSTEP 2: Loading and cleaning the full dataset...")

# 1. Load the dataset using the file paths in your current directory
full_dataset = load_dataset('json', data_files={
    'train': 'urdu_train.jsonl',
    'validation': 'urdu_val.jsonl',
})
print("Dataset loaded successfully.")

# 2. Remove empty rows
print("Filtering out empty or whitespace-only records...")
filtered_dataset = full_dataset.filter(
    lambda example: example['text'] and example['summary'] and not example['text'].isspace() and not example['summary'].isspace()
)
print("Filtering complete.")

# 3. Apply the cleaning function to the 'text' and 'summary' columns
print("Applying cleaning and normalization to the dataset... (This will take a few minutes)")
cleaned_dataset = filtered_dataset.map(
    lambda example: {
        'text': clean_and_normalize_urdu(example['text']),
        'summary': clean_and_normalize_urdu(example['summary'])
    },
    batched=False, # Processing row-by-row
    load_from_cache_file=False # Prevents disk space errors
)

print("\n✅ Dataset has been successfully cleaned and normalized.")
print("Here is a sample of the cleaned data:")
# We select an example from the original dataset to show the 'before' and 'after'
original_sample_text = full_dataset['train'][0]['text']
cleaned_sample_text = cleaned_dataset['train'][0]['text']

print("\n--- ORIGINAL SAMPLE ---")
print(original_sample_text)
print("\n--- CLEANED SAMPLE ---")
print(cleaned_sample_text)