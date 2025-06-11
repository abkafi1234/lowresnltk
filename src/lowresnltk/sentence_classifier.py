from math import e
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from .main import download_model

class SentenceDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length=64):
        self.encodings = tokenizer(sentences, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class SentenceClassifier:
    def __init__(self, data:pd.DataFrame=None, model_name="csebuetnlp/banglabert" ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sentences = data['Sentence'].tolist()
        self.labels = data['Labels'].tolist()
        self.label_encoder = LabelEncoder()
        self.labels_encoded = self.label_encoder.fit_transform(self.labels)
        num_labels = len(self.label_encoder.classes_)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.X_test = None
        self.ytest = None
    
    def train(self, test_size=0.2, epoch=2, per_device_train_batch_size=8,
            per_device_eval_batch_size=8, random_state=42):
        
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            self.sentences, self.labels_encoded, test_size=test_size, random_state=random_state
        )

        train_dataset = SentenceDataset(X_train, y_train, self.tokenizer)
        test_dataset = SentenceDataset(self.X_test, self.y_test, self.tokenizer)
        training_args = TrainingArguments(
            num_train_epochs=epoch,
            per_device_train_batch_size= per_device_train_batch_size,
            per_device_eval_batch_size= per_device_eval_batch_size,
            eval_strategy="epoch",
            logging_dir='./logs',
            logging_steps=10,
            save_strategy="no",
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )

        trainer.train()
        results = trainer.evaluate()
        return results
    
    def save_model(self, path=None):
        """Save the trained model and tokenizer"""
        if path is None:
            path = os.path.expanduser('~/.lowresnltk/ClassifierModel')
        
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        np.save(f"{path}/label_encoder_classes.npy", self.label_encoder.classes_)
        
    def predict(self, sentences):
        self.model.eval()
        device = next(self.model.parameters()).device
        if isinstance(sentences, str):
            sentences = [sentences]
        inputs = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        return self.label_encoder.inverse_transform(predictions.cpu().numpy())
    
    def get_classification_report(self):
        """
        Generate a classification report on the provided test set.
        X_test: list of sentences
        y_test: list or array of encoded labels (integers)
        """
        test_dataset = SentenceDataset(self.X_test, self.y_test, self.tokenizer)
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir='./results',
                per_device_eval_batch_size=8,
                report_to="none"
            ),
            eval_dataset=test_dataset
        )
        predictions = trainer.predict(test_dataset)
        y_pred = predictions.predictions.argmax(axis=1)
        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("Classification Report:\n", classification_report(
            self.y_test, y_pred, target_names=self.label_encoder.classes_))
    
    def get_validation_report(self,X_val, y_val):
        """
        Generate a classification report on the provided test set.
        X_test: list of sentences
        y_test: list or array of encoded labels (integers)
        """
        test_dataset = SentenceDataset(self.X_test, self.y_test, self.tokenizer)
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir='./results',
                per_device_eval_batch_size=8,
                report_to="none"
            ),
            eval_dataset=test_dataset
        )
        predictions = trainer.predict(test_dataset)
        y_pred = predictions.predictions.argmax(axis=1)
        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("Classification Report (Validation):\n", classification_report(
            self.y_test, y_pred, target_names=self.label_encoder.classes_))
        

    @classmethod
    def classify(cls, sentence, model_dir=None):
        """
        Load model, tokenizer, and label encoder from model_dir and classify the given sentence.
        """
        # Load tokenizer and model
        download_model('classifier')
        if model_dir is None:
            model_dir = os.path.expanduser('~/.lowresnltk/ClassifierModel')

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        # Load label encoder classes
        label_classes = np.load(f"{model_dir}/label_encoder_classes.npy", allow_pickle=True)
        # Reconstruct label encoder
        label_encoder = LabelEncoder()
        label_encoder.classes_ = label_classes

        # Tokenize and predict
        model.eval()
        inputs = tokenizer([sentence], return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).cpu().numpy()[0]
        return str(label_encoder.inverse_transform([pred])[0])