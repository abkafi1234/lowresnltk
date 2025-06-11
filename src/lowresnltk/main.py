import os
import subprocess


__version__ = "0.1.0"
__all__ = ['POSTagger', 'SentenceClassifier']

# Model files to download
MODEL_FILES = [
    'config.json',
    'label_encoder_classes.npy',
    'model.safetensors',
    'special_tokens_map.json',
    'tokenizer_config.json',
    'tokenizer.json',
    'vocab.txt'
]

# Define model paths and URLs
MODELS = {
    'pos': {
        'path': '~/.lowresnltk/POSModel',
        'url': 'https://huggingface.co/abkafi1234/bangla-pos-tagger/resolve/main'
    },
    'classifier': {
        'path': '~/.lowresnltk/ClassifierModel',
        'url': 'https://huggingface.co/abkafi1234/bangla-sentence-classifier/resolve/main'
    }
}

def download_model(model_type):
    """Download all model files if they don't exist"""
    model_info = MODELS.get(model_type)
    if not model_info:
        raise ValueError(f"Unknown model type: {model_type}")
    
    base_path = os.path.expanduser(model_info['path'])
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
        print(f"Downloading {model_type} model files...")
        
        for file in MODEL_FILES:
            file_url = f"{model_info['url']}/{file}"
            file_path = os.path.join(base_path, file)
            try:
                subprocess.run(['wget', file_url, '-O', file_path], check=True)
            except subprocess.CalledProcessError:
                print(f"Failed to download {file}")
                raise