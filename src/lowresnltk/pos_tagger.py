import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import ast
import numpy as np
import os
from .main import download_model

class POSTaggingDataset(Dataset):
    def __init__(self, words, pos_labels, tokenizer, max_length=32):
        self.words = words
        self.pos_labels = pos_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        # Tokenize words and get word IDs
        encoding = self.tokenizer(
            self.words[idx],
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create labels array
        labels = [-100] * self.max_length
        word_ids = encoding.word_ids(batch_index=0)
        
        # Assign labels to first token of each word
        previous_word_id = None
        for i, word_id in enumerate(word_ids):
            if word_id is not None and word_id != previous_word_id:
                if word_id < len(self.pos_labels[idx]):
                    labels[i] = self.pos_labels[idx][word_id]
                previous_word_id = word_id

        # Prepare final item
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(labels)
        return item

class POSTagger:
    def __init__(self, data: pd.DataFrame, model_name="csebuetnlp/banglabert", max_length=32):
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
        # Process data
        self.words = [ast.literal_eval(w) for w in data['Words']]
        self.pos_tags = [ast.literal_eval(p) for p in data['POS']]
        
        # Create label encoder
        all_tags = [tag for tags in self.pos_tags for tag in tags]
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(all_tags)
        self.encoded_pos_tags = [self.label_encoder.transform(tags).tolist() for tags in self.pos_tags]
        
        # Create model
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, 
            num_labels=len(self.label_encoder.classes_)
        )

    def train(self, test_size=0.2, epochs=2, batch_size=8):
        # Split data
        train_words, test_words, train_pos, test_pos = train_test_split(
            self.words, self.encoded_pos_tags, test_size=test_size
        )

        # Create datasets
        train_dataset = POSTaggingDataset(train_words, train_pos, self.tokenizer, self.max_length)
        test_dataset = POSTaggingDataset(test_words, test_pos, self.tokenizer, self.max_length)

        # Training arguments
        training_args = TrainingArguments(
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_strategy="epoch",
            logging_dir='./pos_logs',
            save_strategy="no",
            report_to="none"
        )

        # Train
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )
        trainer.train()
        results = trainer.evaluate()
        print("POS Tagging Evaluation:", results)
        
        # Save the trained model
        self.save_model()

    def predict(self, text):
        self.model.eval()
        device = next(self.model.parameters()).device

        # Convert input to list of words
        if isinstance(text, str):
            words = text.strip().split()
        elif isinstance(text, list):
            words = text[0].strip().split() if len(text) == 1 else text

        # Tokenize
        inputs = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length =self.max_length
        ).to(device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits.argmax(dim=-1)[0].cpu().numpy()

        # Convert predictions to tags
        tags = []
        word_ids = inputs.word_ids()
        prev_word_id = None
        
        for idx, word_id in enumerate(word_ids):
            if word_id is not None and word_id != prev_word_id:
                tag = self.label_encoder.inverse_transform([predictions[idx]])[0]
                tags.append(str(tag))
                prev_word_id = word_id

        return tags[:len(words)]

    def save_model(self, path=None):
        if path is None:
            path = os.path.expanduser('~/.lowresnltk/POSModel')
        
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        np.save(f"{path}/label_encoder_classes.npy", self.label_encoder.classes_)
    # ... existing imports ...

    @classmethod
    def tag(cls, text, model_path=None):
        """
        Class method to load a saved model and predict POS tags
        Args:
            text: Input text or list of words
            model_path: Path to saved model directory
        Returns:
            List of POS tags
        """
        download_model('pos')
        if model_path is None:
            model_path = os.path.expanduser('~/.lowresnltk/POSModel')

        # Initialize instance without training data
        instance = cls(pd.DataFrame({'Words': [], 'POS': []}))
        
        # Load saved model and tokenizer
        instance.model = AutoModelForTokenClassification.from_pretrained(model_path)
        instance.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load label encoder classes
        classes_path = f"{model_path}/label_encoder_classes.npy"
        if not os.path.exists(classes_path):
            raise ValueError(f"Label encoder classes not found at {classes_path}")
        
        instance.label_encoder = LabelEncoder()
        instance.label_encoder.classes_ = np.load(classes_path)

        # Use existing predict method
        return instance.predict(text)