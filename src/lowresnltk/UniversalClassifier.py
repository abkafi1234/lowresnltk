import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModelForTokenClassification, 
    Trainer, 
    TrainingArguments
)

# ---------------------------------------------------------
# Internal Dataset Classes (Hidden from the user)
# ---------------------------------------------------------

class _SequenceDataset(Dataset):
    """Dataset for Binary, Multiclass, and Multilabel Sequence Classification"""
    def __init__(self, texts, labels, tokenizer, max_length=64, is_multilabel=False):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels
        self.is_multilabel = is_multilabel

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Multilabel requires floats for BCEWithLogitsLoss
        if self.is_multilabel:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        else:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

class _TokenDataset(Dataset):
    """Dataset for Token Classification (NER, POS)"""
    def __init__(self, words_list, labels_list, tokenizer, max_length=64):
        self.words_list = words_list
        self.labels_list = labels_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.words_list)

    def __getitem__(self, idx):
        words = self.words_list[idx]
        labels = self.labels_list[idx]
        
        encoding = self.tokenizer(
            words, is_split_into_words=True, truncation=True, 
            padding='max_length', max_length=self.max_length, return_tensors='pt'
        )
        
        label_ids = [-100] * self.max_length
        word_ids = encoding.word_ids(batch_index=0)
        
        previous_word_id = None
        for i, word_id in enumerate(word_ids):
            if word_id is not None and word_id != previous_word_id:
                if word_id < len(labels):
                    label_ids[i] = labels[word_id]
                previous_word_id = word_id

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label_ids, dtype=torch.long)
        return item


# ---------------------------------------------------------
# Main API Class
# ---------------------------------------------------------

class UniversalClassifier:
    def __init__(self, kind: str, model_name="csebuetnlp/banglabert", max_length=64):
        """
        Args:
            kind: 'binary', 'multiclass', 'multilabel', or 'token'
            model_name: HuggingFace model hub name or local path
        """
        valid_kinds = ['binary', 'multiclass', 'multilabel', 'token']
        if kind not in valid_kinds:
            raise ValueError(f"Invalid kind. Must be one of: {valid_kinds}")
            
        self.kind = kind
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.kind == 'multilabel':
            self.label_encoder = MultiLabelBinarizer()
        else:
            self.label_encoder = LabelEncoder()
            
        self.model = None

    def fit(self, X, y, test_size=0.2, epochs=2, batch_size=8, random_state=42, output_dir='./logs'):
        """
        Sklearn-style fit method. Handles label encoding, model routing, and training.
        """
        # 1. Encode labels & Initialize Model based on task
        if self.kind == 'token':
            flat_y = [tag for seq in y for tag in seq]
            self.label_encoder.fit(flat_y)
            encoded_y = [self.label_encoder.transform(seq).tolist() for seq in y]
            num_labels = len(self.label_encoder.classes_)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name, num_labels=num_labels)
            
        elif self.kind == 'multilabel':
            encoded_y = self.label_encoder.fit_transform(y).astype(float) 
            num_labels = len(self.label_encoder.classes_)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=num_labels, problem_type="multi_label_classification" 
            )
            
        else: # binary & multiclass
            encoded_y = self.label_encoder.fit_transform(y)
            num_labels = len(self.label_encoder.classes_)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels)

        # 2. Train-Test Split
        X_train, X_val, y_train, y_val = train_test_split(
            X, encoded_y, test_size=test_size, random_state=random_state
        )

        # 3. Create Datasets
        if self.kind == 'token':
            train_dataset = _TokenDataset(X_train, y_train, self.tokenizer, self.max_length)
            eval_dataset = _TokenDataset(X_val, y_val, self.tokenizer, self.max_length)
        else:
            is_multi = (self.kind == 'multilabel')
            train_dataset = _SequenceDataset(X_train, y_train, self.tokenizer, self.max_length, is_multi)
            eval_dataset = _SequenceDataset(X_val, y_val, self.tokenizer, self.max_length, is_multi)

        # 4. Train
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_strategy="epoch",
            save_strategy="no",
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        trainer.train()
        # print(f"Validation Results: {trainer.evaluate()}")
        return self

    def predict(self, X):
        """
        Sklearn-style predict method. Handles single strings or batches automatically.
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Please run fit() or load a trained model first.")
            
        self.model.eval()
        device = next(self.model.parameters()).device
        
        if self.kind == 'token':
            # Handle single list of words vs list of lists
            is_single = False
            if isinstance(X, str):
                X = [X.split()]
                is_single = True
            elif isinstance(X, list) and isinstance(X[0], str) and not any(' ' in s for s in X):
                X = [X]
                is_single = True

            results = []
            for words in X:
                inputs = self.tokenizer(words, is_split_into_words=True, return_tensors="pt", 
                                        padding=True, truncation=True, max_length=self.max_length).to(device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = outputs.logits.argmax(dim=-1)[0].cpu().numpy()
                
                tags = []
                word_ids = inputs.word_ids()
                prev_word_id = None
                for idx, word_id in enumerate(word_ids):
                    if word_id is not None and word_id != prev_word_id:
                        tag = self.label_encoder.inverse_transform([predictions[idx]])[0]
                        tags.append(str(tag))
                        prev_word_id = word_id
                results.append(tags[:len(words)])
                
            return results[0] if is_single else results

        else: # Sequence tasks
            is_single = isinstance(X, str)
            if is_single:
                X = [X]
                
            inputs = self.tokenizer(X, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.cpu()
            
            if self.kind == 'multilabel':
                probs = torch.sigmoid(logits).numpy()
                predictions = (probs > 0.5).astype(int) # Apply 0.5 threshold
                result = self.label_encoder.inverse_transform(predictions)
            else:
                predictions = torch.argmax(logits, dim=-1).numpy()
                result = self.label_encoder.inverse_transform(predictions)
                
            return result[0] if is_single else list(result)

    def save(self, path):
        """Save the trained model, tokenizer, and label encoder."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        np.save(f"{path}/label_encoder_classes.npy", self.label_encoder.classes_)
        with open(f"{path}/model_kind.txt", "w") as f:
            f.write(self.kind)
        print(f"Model saved successfully to {path}")
    

    @classmethod
    def load(cls, path):
        """Load a trained UniversalClassifier from disk."""
        with open(f"{path}/model_kind.txt", "r") as f:
            kind = f.read().strip()
            
        instance = cls(kind=kind, model_name=path)
        
        # Restore classes
        classes = np.load(f"{path}/label_encoder_classes.npy", allow_pickle=True)
        if kind == 'multilabel':
            instance.label_encoder.classes_ = classes
        else:
            instance.label_encoder.classes_ = classes
            
        # Load correct architecture
        if kind == 'token':
            instance.model = AutoModelForTokenClassification.from_pretrained(path)
        else:
            instance.model = AutoModelForSequenceClassification.from_pretrained(path)
            
        return instance

    def evaluate(self, X_test, y_test):
        """
        Generates a comprehensive evaluation report for researchers.
        """
        print("Running inference on test set...")
        y_pred = self.predict(X_test)
        
        if self.kind == 'token':
            # Flatten lists for NER/POS evaluation
            flat_y_true = [tag for seq in y_test for tag in seq]
            flat_y_pred = [tag for seq in y_pred for tag in seq]
            
            # Ensure lengths match (in case truncation happened during prediction)
            min_len = min(len(flat_y_true), len(flat_y_pred))
            flat_y_true, flat_y_pred = flat_y_true[:min_len], flat_y_pred[:min_len]

            print("\n--- Token Classification Report ---")
            print(classification_report(flat_y_true, flat_y_pred))
            
        elif self.kind == 'multilabel':
            # Multilabel requires a special binarized approach
            y_true_encoded = self.label_encoder.transform(y_test)
            y_pred_encoded = self.label_encoder.transform(y_pred)
            
            print("\n--- Multilabel Classification Report ---")
            print(classification_report(y_true_encoded, y_pred_encoded, target_names=self.label_encoder.classes_))
            
        else: # Binary & Multiclass
            print("\n--- Classification Report ---")
            print(classification_report(y_test, y_pred))
            
            print("\n--- Confusion Matrix ---")
            print(confusion_matrix(y_test, y_pred))