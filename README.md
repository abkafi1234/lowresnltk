# LowResNLTK

A low-resource Natural Language Processing toolkit for Bangla language.

## Quick Inference Without Training
Pretrained model achived an F1 Score of 97% in all classes.

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