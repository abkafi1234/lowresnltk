import os
import torch
import evaluate
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

# ---------------------------------------------------------
# Internal Dataset Class
# ---------------------------------------------------------

class _Seq2SeqDataset(Dataset):
    """Dataset for Machine Translation, Summarization, and Text Generation"""
    def __init__(self, X, y, tokenizer, max_input_length=128, max_target_length=128):
        # Tokenize the source texts (X)
        self.inputs = tokenizer(
            X, max_length=max_input_length, truncation=True
        )
        # Tokenize the target texts (y) using text_target
        self.targets = tokenizer(
            text_target=y, max_length=max_target_length, truncation=True
        )

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, idx):
        # We don't pad here. The DataCollatorForSeq2Seq will do it dynamically!
        return {
            "input_ids": self.inputs["input_ids"][idx],
            "attention_mask": self.inputs["attention_mask"][idx],
            "labels": self.targets["input_ids"][idx]
        }

# ---------------------------------------------------------
# Main API Class
# ---------------------------------------------------------

class UniversalGenerator:
    def __init__(self, model_name="csebuetnlp/banglat5", max_input_length=128, max_target_length=128):
        """
        model_name: HuggingFace model hub name (e.g., 't5-small', 'facebook/bart-base')
        """
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # The collator handles dynamic padding and the -100 label masking for Seq2Seq
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

    def fit(self, X, y, test_size=0.2, epochs=2, batch_size=4, random_state=42, output_dir='./seq2seq_logs'):
        """
        Sklearn-style fit method for translation/summarization.
        """
        # Train-Test Split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Create Datasets
        train_dataset = _Seq2SeqDataset(X_train, y_train, self.tokenizer, self.max_input_length, self.max_target_length)
        eval_dataset = _Seq2SeqDataset(X_val, y_val, self.tokenizer, self.max_input_length, self.max_target_length)

        # Seq2Seq Training Arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_strategy="epoch",
            save_strategy="no",
            predict_with_generate=True, # Crucial for Seq2Seq!
            report_to="none"
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            data_collator=self.data_collator
        )
        
        trainer.train()
        return self

    def predict(self, X, max_new_tokens=50, num_beams=4):
        """
        Generate predictions using Beam Search for better quality text.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        
        is_single = isinstance(X, str)
        if is_single:
            X = [X]
            
        # Tokenize inputs
        inputs = self.tokenizer(
            X, return_tensors='pt', padding=True, 
            truncation=True, max_length=self.max_input_length
        ).to(device)
        
        # Generate!
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams, # Beam search drastically improves translation/summarization quality
                early_stopping=True
            )
            
        # Decode the generated token IDs back into human text
        predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        return predictions[0] if is_single else predictions

    def save(self, path):
        """Save the trained generator."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Generator saved successfully to {path}")

    @classmethod
    def load(cls, path):
        """Load a trained UniversalGenerator from disk."""
        instance = cls(model_name=path)
        return instance
    
    def evaluate(self, X_test, y_test, task="translation", num_beams=4):
        """
        Evaluates generative text. 
        task: 'translation' (uses BLEU) or 'summarization' (uses ROUGE)
        """
        print(f"Generating text for {len(X_test)} samples... (This may take a moment)")
        predictions = self.predict(X_test, num_beams=num_beams)
        
        if task == "translation":
            sacrebleu = evaluate.load("sacrebleu")
            # Sacrebleu expects references as a list of lists: [[ref1, ref2], [ref1, ref2]]
            references = [[ref] for ref in y_test] 
            results = sacrebleu.compute(predictions=predictions, references=references)
            
            print("\n--- Translation Metrics (SacreBLEU) ---")
            print(f"BLEU Score: {results['score']:.2f}")
            
        elif task == "summarization":
            rouge = evaluate.load("rouge")
            results = rouge.compute(predictions=predictions, references=y_test)
            
            print("\n--- Summarization Metrics (ROUGE) ---")
            print(f"ROUGE-1: {results['rouge1']:.4f}")
            print(f"ROUGE-2: {results['rouge2']:.4f}")
            print(f"ROUGE-L: {results['rougeL']:.4f}")
            
        else:
            print("Invalid task. Choose 'translation' or 'summarization'.")