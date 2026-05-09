#!/usr/bin/env python3
"""
Standalone Urdu Summarization Model Validation Script
Usage: python run_validation_mt5_base.py
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import time
import psutil
import subprocess
from datetime import datetime
import re
from collections import Counter
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    GenerationConfig
)
import warnings
warnings.filterwarnings("ignore")

# Configuration - Update these paths for your setup
CONFIG = {
    'MODEL_PATH': './workspace/mt5_base_final_model_cleaned/model_files',
    'DATASET_PATH': './urdu_val.jsonl',
    'OUTPUT_DIR': './validation_results',
    'BATCH_SIZE': 24,  # Optimized for RTX 5090
    'MAX_LENGTH': 512,
    'MIN_LENGTH': 50,
    'TEXT_COLUMN': 'text',
    'SUMMARY_COLUMN': 'summary',
    'DEVICE': 'cuda',
    'SAMPLE_SIZE': None,  # Set to a number (e.g., 100) for testing, None for full dataset
    'SAVE_PREDICTIONS': True
}

class UrduROUGEEvaluator:
    """Enhanced ROUGE evaluator for Urdu text"""
    
    @staticmethod
    def urdu_tokenize(text):
        """Tokenizer optimized for Urdu text"""
        if not text or pd.isna(text):
            return []
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', str(text).strip())
        
        # Split by whitespace and clean punctuation
        words = []
        for word in text.split():
            # Remove punctuation from start and end, keep Urdu characters
            clean_word = re.sub(r'^[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+', '', word)
            clean_word = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+$', '', clean_word)
            if clean_word:  # Only add non-empty words
                words.append(clean_word)
        
        return words
    
    @staticmethod
    def calculate_rouge_n(reference_tokens, candidate_tokens, n):
        """Calculate ROUGE-N score"""
        if len(reference_tokens) == 0 or len(candidate_tokens) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Create n-grams
        def get_ngrams(tokens, n):
            if len(tokens) < n:
                return []
            return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        
        ref_ngrams = get_ngrams(reference_tokens, n)
        cand_ngrams = get_ngrams(candidate_tokens, n)
        
        if len(ref_ngrams) == 0 or len(cand_ngrams) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Count overlapping n-grams
        ref_counter = Counter(ref_ngrams)
        cand_counter = Counter(cand_ngrams)
        
        overlap = sum((ref_counter & cand_counter).values())
        
        precision = overlap / len(cand_ngrams)
        recall = overlap / len(ref_ngrams)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    @staticmethod
    def calculate_rouge_l(reference_tokens, candidate_tokens):
        """Calculate ROUGE-L score using Longest Common Subsequence"""
        def lcs_length(X, Y):
            m, n = len(X), len(Y)
            L = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if X[i-1] == Y[j-1]:
                        L[i][j] = L[i-1][j-1] + 1
                    else:
                        L[i][j] = max(L[i-1][j], L[i][j-1])
            
            return L[m][n]
        
        if len(reference_tokens) == 0 or len(candidate_tokens) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        lcs_len = lcs_length(reference_tokens, candidate_tokens)
        
        precision = lcs_len / len(candidate_tokens) if len(candidate_tokens) > 0 else 0.0
        recall = lcs_len / len(reference_tokens) if len(reference_tokens) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {'precision': precision, 'recall': recall, 'f1': f1}

class ModelValidator:
    """Main validation class for Urdu summarization model"""
    
    def __init__(self, config):
        self.config = config
        self.rouge_evaluator = UrduROUGEEvaluator()
        self.load_model()
        
    def load_model(self):
        """Load the trained model and tokenizer"""
        print(f"Loading model from {self.config['MODEL_PATH']}...")
        print(f"Using device: {self.config['DEVICE']}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config['MODEL_PATH'])
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config['MODEL_PATH'])
            self.model.to(self.config['DEVICE'])
            self.model.eval()
            
            # Create generation config
            self.generation_config = GenerationConfig(
                max_length=self.config['MAX_LENGTH'],
                min_length=self.config['MIN_LENGTH'],
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            print("Model loaded successfully!")
            print(f"Model device: {next(self.model.parameters()).device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def generate_summaries_batch(self, articles):
        """Generate summaries for a batch of articles"""
        try:
            # Tokenize inputs
            inputs = self.tokenizer(
                articles, 
                max_length=1024, 
                truncation=True, 
                padding=True, 
                return_tensors="pt"
            ).to(self.config['DEVICE'])
            
            # Generate summaries
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config
                )
            
            # Decode summaries
            summaries = self.tokenizer.batch_decode(
                outputs, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )
            
            return summaries
            
        except Exception as e:
            print(f"Error in batch generation: {e}")
            return [""] * len(articles)
    
    def load_dataset(self):
        """Load dataset from JSONL file"""
        dataset_path = Path(self.config['DATASET_PATH'])
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        print(f"Loading dataset from {dataset_path}...")
        
        # Load JSONL file
        data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        df = pd.DataFrame(data)
        print(f"Dataset loaded: {len(df)} samples")
        print(f"Columns: {list(df.columns)}")
        
        return df
    
    def validate_dataset(self):
        """Validate the model on the dataset"""
        # Create output directory
        output_dir = Path(self.config['OUTPUT_DIR'])
        output_dir.mkdir(exist_ok=True)
        
        # Load dataset
        df = self.load_dataset()
        
        # Sample dataset if specified
        if self.config['SAMPLE_SIZE'] and self.config['SAMPLE_SIZE'] < len(df):
            df = df.sample(n=self.config['SAMPLE_SIZE'], random_state=42)
            print(f"Sampled {self.config['SAMPLE_SIZE']} examples for validation")
        
        # Prepare data
        articles = df[self.config['TEXT_COLUMN']].fillna('').tolist()
        reference_summaries = df[self.config['SUMMARY_COLUMN']].fillna('').tolist()
        
        print(f"Starting validation on {len(articles)} samples...")
        print(f"Batch size: {self.config['BATCH_SIZE']}")
        
        # Initialize results storage
        all_generated_summaries = []
        all_rouge_scores = []
        
        # Process in batches
        num_batches = (len(articles) + self.config['BATCH_SIZE'] - 1) // self.config['BATCH_SIZE']
        
        start_time = time.time()
        
        for i in tqdm(range(0, len(articles), self.config['BATCH_SIZE']), 
                     desc="Generating summaries", total=num_batches):
            
            batch_articles = articles[i:i + self.config['BATCH_SIZE']]
            batch_references = reference_summaries[i:i + self.config['BATCH_SIZE']]
            
            # Generate summaries for batch
            batch_generated = self.generate_summaries_batch(batch_articles)
            all_generated_summaries.extend(batch_generated)
            
            # Calculate ROUGE scores for batch
            for ref, gen in zip(batch_references, batch_generated):
                ref_tokens = self.rouge_evaluator.urdu_tokenize(ref)
                gen_tokens = self.rouge_evaluator.urdu_tokenize(gen)
                
                rouge1 = self.rouge_evaluator.calculate_rouge_n(ref_tokens, gen_tokens, 1)
                rouge2 = self.rouge_evaluator.calculate_rouge_n(ref_tokens, gen_tokens, 2)
                rougeL = self.rouge_evaluator.calculate_rouge_l(ref_tokens, gen_tokens)
                
                all_rouge_scores.append({
                    'rouge1_f1': rouge1['f1'],
                    'rouge1_precision': rouge1['precision'],
                    'rouge1_recall': rouge1['recall'],
                    'rouge2_f1': rouge2['f1'],
                    'rouge2_precision': rouge2['precision'],
                    'rouge2_recall': rouge2['recall'],
                    'rougeL_f1': rougeL['f1'],
                    'rougeL_precision': rougeL['precision'],
                    'rougeL_recall': rougeL['recall']
                })
            
            # Print progress every 10 batches
            if (i // self.config['BATCH_SIZE'] + 1) % 10 == 0:
                elapsed = time.time() - start_time
                samples_processed = min(i + self.config['BATCH_SIZE'], len(articles))
                speed = samples_processed / elapsed
                eta = (len(articles) - samples_processed) / speed
                
                print(f"Processed {samples_processed}/{len(articles)} samples "
                      f"({speed:.1f} samples/sec, ETA: {eta/60:.1f}min)")
                self.print_gpu_stats()
        
        # Calculate final metrics
        results = self.calculate_final_metrics(all_rouge_scores)
        
        # Save results
        self.save_results(results, all_generated_summaries, articles, 
                         reference_summaries, output_dir)
        
        return results
    
    def calculate_final_metrics(self, rouge_scores):
        """Calculate aggregated metrics"""
        df_scores = pd.DataFrame(rouge_scores)
        
        results = {
            'total_samples': len(rouge_scores),
            'config': self.config,
            'rouge1': {
                'f1': float(df_scores['rouge1_f1'].mean()),
                'precision': float(df_scores['rouge1_precision'].mean()),
                'recall': float(df_scores['rouge1_recall'].mean()),
                'std': float(df_scores['rouge1_f1'].std())
            },
            'rouge2': {
                'f1': float(df_scores['rouge2_f1'].mean()),
                'precision': float(df_scores['rouge2_precision'].mean()),
                'recall': float(df_scores['rouge2_recall'].mean()),
                'std': float(df_scores['rouge2_f1'].std())
            },
            'rougeL': {
                'f1': float(df_scores['rougeL_f1'].mean()),
                'precision': float(df_scores['rougeL_precision'].mean()),
                'recall': float(df_scores['rougeL_recall'].mean()),
                'std': float(df_scores['rougeL_f1'].std())
            }
        }
        
        return results
    
    def save_results(self, results, generated_summaries, articles, reference_summaries, output_dir):
        """Save validation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        metrics_file = output_dir / f"validation_metrics_{timestamp}.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save predictions if requested
        if self.config['SAVE_PREDICTIONS']:
            predictions_file = output_dir / f"predictions_{timestamp}.csv"
            df_predictions = pd.DataFrame({
                'article': articles,
                'reference_summary': reference_summaries,
                'generated_summary': generated_summaries
            })
            df_predictions.to_csv(predictions_file, index=False)
            print(f"Predictions saved to: {predictions_file}")
        
        print(f"Metrics saved to: {metrics_file}")
    
    def print_results(self, results):
        """Print formatted results"""
        print("\n" + "="*60)
        print("URDU SUMMARIZATION MODEL VALIDATION RESULTS")
        print("="*60)
        print(f"Total samples evaluated: {results['total_samples']}")
        print()
        
        print("ROUGE-1 Scores:")
        print(f"  F1-Score:  {results['rouge1']['f1']:.4f} ± {results['rouge1']['std']:.4f}")
        print(f"  Precision: {results['rouge1']['precision']:.4f}")
        print(f"  Recall:    {results['rouge1']['recall']:.4f}")
        print()
        
        print("ROUGE-2 Scores:")
        print(f"  F1-Score:  {results['rouge2']['f1']:.4f} ± {results['rouge2']['std']:.4f}")
        print(f"  Precision: {results['rouge2']['precision']:.4f}")
        print(f"  Recall:    {results['rouge2']['recall']:.4f}")
        print()
        
        print("ROUGE-L Scores:")
        print(f"  F1-Score:  {results['rougeL']['f1']:.4f} ± {results['rougeL']['std']:.4f}")
        print(f"  Precision: {results['rougeL']['precision']:.4f}")
        print(f"  Recall:    {results['rougeL']['recall']:.4f}")
        print("="*60)
    
    def print_gpu_stats(self):
        """Print GPU memory usage"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            gpu_memory_max = torch.cuda.max_memory_allocated() / 1024**3
            print(f"GPU Memory: {gpu_memory:.2f}GB / {gpu_memory_max:.2f}GB")

def print_system_info():
    """Print system information"""
    print("System Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f}GB")
    print()

def install_requirements():
    """Install required packages"""
    try:
        print("Installing/updating required packages...")
        subprocess.check_call([
            "pip", "install", "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ])
        subprocess.check_call([
            "pip", "install", "transformers", "pandas", "numpy", "tqdm", "psutil"
        ])
        print("Packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        print("Please install manually: pip install torch transformers pandas numpy tqdm psutil")

def main():
    """Main validation function"""
    print("=" * 60)
    print("URDU SUMMARIZATION MODEL VALIDATION")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Print system info
    print_system_info()
    
    # Install requirements (comment out if already installed)
    # install_requirements()
    
    # Print configuration
    print("Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print()
    
    # Check if model and dataset exist
    if not Path(CONFIG['MODEL_PATH']).exists():
        print(f"Error: Model directory not found at {CONFIG['MODEL_PATH']}")
        print("Please check your model path.")
        return
    
    if not Path(CONFIG['DATASET_PATH']).exists():
        print(f"Error: Dataset file not found at {CONFIG['DATASET_PATH']}")
        print("Please check your dataset path.")
        return
    
    try:
        # Initialize validator
        print("Initializing validator...")
        validator = ModelValidator(CONFIG)
        
        # Run validation
        start_time = time.time()
        print("Starting validation...")
        results = validator.validate_dataset()
        
        total_time = time.time() - start_time
        
        # Print results
        validator.print_results(results)
        print(f"\nValidation completed in {total_time/60:.2f} minutes")
        print(f"Average time per sample: {total_time/results['total_samples']:.3f} seconds")
        
        print(f"\nResults saved in: {CONFIG['OUTPUT_DIR']}")
        
    except Exception as e:
        print(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()