# LowResNLTK

# lowresnltk 🚀
**A lightweight, model-agnostic NLP toolkit for Low-Resource Languages.**

`lowresnltk` is designed for researchers who want to apply State-of-the-Art Transformer models (BERT, T5, BART, etc.) to their data without writing thousands of lines of PyTorch boilerplate. It provides a familiar **Scikit-Learn API** (`fit`, `predict`, `evaluate`) for complex NLP tasks.

## 🌟 Key Features
- **Universal Classifier**: One class for Binary, Multiclass, Multilabel, and Token Classification (NER/POS).
- **Universal Generator**: Seamless Machine Translation, Summarization, and Paraphrasing.
- **Hardware Agnostic**: Automatically detects and uses CUDA (NVIDIA), MPS (Apple Silicon), or CPU.
- **Research Ready**: Built-in methods for Classification Reports, Confusion Matrices, BLEU, and ROUGE scores.

---

## 📦 Installation

```bash
pip install lowresnltk


🛠️ Quick Start
1. Classification (Sentence or Token/NER)
Whether you are doing Sentiment Analysis or Part-of-Speech tagging, the API is the same.

Python
from lowresnltk import UniversalClassifier

# For Sentence Classification (Binary/Multiclass)
clf = UniversalClassifier(kind='binary', model_name="csebuetnlp/banglabert")
clf.fit(X_train, y_train, epochs=3)

# For POS Tagging or NER
ner = UniversalClassifier(kind='token', model_name="csebuetnlp/banglabert")
ner.fit(X_tokens, y_tags)

# Evaluate like a pro
ner.evaluate(X_test, y_test)
2. Machine Translation & Summarization
Generate text using Encoder-Decoder models like T5 or BART.

Python
from lowresnltk import UniversalGenerator

# Initialize for Translation
translator = UniversalGenerator(model_name="csebuetnlp/banglat5")

# Fit on your parallel corpus
translator.fit(english_texts, bengali_texts, epochs=5)

# Generate translation
result = translator.predict("The weather is beautiful today.")
print(result)

# Get academic metrics (BLEU score)
translator.evaluate(X_test, y_test, task="translation")
3. Save and Load
Save your hard-earned trained models for later use.

Python
model.save("./my_saved_model")

# Later...
from lowresnltk import UniversalClassifier
new_model = UniversalClassifier.load("./my_saved_model")


### POS Tagging
```python
from lowresnltk import POSTagger

# Simple usage
tags = POSTagger.tag('আমি ভালো আছি')
```

### Sentence Classification
```python
from lowresnltk import SentenceClassifier

# Simple usage
label = SentenceClassifier.classify('আমি ভালো আছি')
```

## Training Custom Models

## Data Format Requirements

| Column | Description | Example |
|--------|-------------|---------|
| Sentence | Full Bengali sentence | সন্ধ্যায় পাখিরা বাসায় ফেরে |
| Labels | Sentence type | Simple |
| POS | List of POS tags | ['ক্রিয়া', 'বিশেষ্য', 'বিশেষ্য', 'অব্যয়'] |
| Words | List of words | ['সন্ধ্যায়', 'পাখিরা', 'বাসায়', 'ফেরে'] |

Example Dataset: https://huggingface.co/datasets/abkafi1234/POS-Sentence-Type
##### The code is Language Agnostic So Any Language will work. if the proper structure is followed 


### Train POS Tagger
```python
import pandas as pd
from lowresnltk import POSTagger

# Load your data
data = pd.read_csv('Bangla.csv')

# Initialize and train
pt = POSTagger(data)
pt.train()

# Test prediction
result = pt.predict('আমি ভালো আছি')
```

### Train Sentence Classifier
```python
from lowresnltk import SentenceClassifier

# Load your data
data = pd.read_csv('Bangla.csv')

# Initialize and train
sc = SentenceClassifier(data=data)
sc.train()
result = sc.predict('আমি ভালো আছি')
```
### Text Normalization Evaluator

*A Unified Multi-Metric Framework for Evaluating Semantic Fidelity in Text Normalization**  

Evaluate how stemming or lemmatization algorithms affect your text both structurally and semantically, beyond just compression ratios.

## Python Example

```python
from lowresnltk import NormalizationEvaluator
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier

# 1. Prepare your parallel datasets and labels
original_texts = ["The quick brown foxes are jumping.", "Beautifully painted canvases."]
normalized_texts = ["The quick brown fox be jump.", "Beauti paint canvas."]
labels = [0, 1]

# 2. Setup your evaluation models
classifiers = [RandomForestClassifier()]
embedding_model = SentenceTransformer("csebuetnlp/banglabert") 

# 3. Initialize and run the evaluator
evaluator = NormalizationEvaluator(
    texts_original=original_texts,
    texts_normalized=normalized_texts,
    labels=labels,
    classifiers=classifiers,
    embedding_model=embedding_model
)

# Returns CR, IRS, AES, ANLD, and MPD scores
results = evaluator.evaluate_all()
print(results)
```


## Model Configuration

Default model paths:
- POS Tagger: `~/.lowresnltk/POSModel/`
- Classifier: `~/.lowresnltk/ClassifierModel/`


## Installation

```bash
pip install lowresnltk
```

## License
MIT License
