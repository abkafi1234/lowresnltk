import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity, paired_cosine_distances
from sklearn.model_selection import train_test_split

import Levenshtein


class NormalizationEvaluator:
    def __init__(self, texts_original, texts_normalized, labels=None, classifiers=None, embedding_model=None):
        """
        Initializes the evaluator with parallel datasets and required models.
        """
        # Ensure texts are lists of strings (handles pandas Series automatically)
        self.texts_original = list(texts_original)
        self.texts_normalized = list(texts_normalized)
        
        self.labels = labels
        self.classifiers = classifiers if classifiers is not None else []
        self.embedding_model = embedding_model
        
        # Basic validation
        if len(self.texts_original) != len(self.texts_normalized):
            raise ValueError("Original and Normalized datasets must have the same number of documents.")
        
    def calculate_cr(self):
            """
            Calculates the Compression Ratio (CR).
            Formula: Unique Words Before Transformation / Unique Words After Transformation
            """
            # Initialize CountVectorizers
            vec_original = CountVectorizer()
            vec_normalized = CountVectorizer()
            
            # Fit the vectorizers to extract the vocabularies
            vec_original.fit(self.texts_original)
            vec_normalized.fit(self.texts_normalized)
            
            # Get the size of the vocabularies
            vocab_size_original = len(vec_original.vocabulary_)
            vocab_size_normalized = len(vec_normalized.vocabulary_)
            
            # Edge case: Avoid division by zero if normalized text is completely empty
            if vocab_size_normalized == 0:
                return 0.0
                
            # Calculate CR
            cr = vocab_size_original / vocab_size_normalized
            
            print(f"Original Vocab Size: {vocab_size_original}")
            print(f"Normalized Vocab Size: {vocab_size_normalized}")
            print(f"Calculated CR: {cr:.4f}")
            
            return cr
        
    def calculate_mpd(self, test_size=0.2, random_state=42, average_method='weighted', verbose=False):
        """
        Calculates the Model Performance Delta (MPD) for each classifier using F1-Score.
        Also generates full classification reports for in-depth analysis.
        """
        if self.labels is None:
            raise ValueError("Ground truth labels must be provided to calculate MPD.")
        if not self.classifiers:
            print("No classifiers provided. Skipping MPD calculation.")
            return None
            
        vectorizer_orig = TfidfVectorizer()
        vectorizer_norm = TfidfVectorizer()
        
        X_orig = vectorizer_orig.fit_transform(self.texts_original)
        X_norm = vectorizer_norm.fit_transform(self.texts_normalized)
        
        X_orig_train, X_orig_test, X_norm_train, X_norm_test, y_train, y_test = train_test_split(
            X_orig, X_norm, self.labels, test_size=test_size, random_state=random_state
        )
        
        mpd_results = {}
        
        for clf in self.classifiers:
            clf_name = type(clf).__name__
            
            # --- Evaluate Original ---
            clf.fit(X_orig_train, y_train)
            preds_orig = clf.predict(X_orig_test)
            f1_orig = f1_score(y_test, preds_orig, average=average_method) 
            # Get the report as a dictionary for storage and string for printing
            report_orig_dict = classification_report(y_test, preds_orig, output_dict=True, zero_division=0)
            report_orig_str = classification_report(y_test, preds_orig, zero_division=0)
            
            # --- Evaluate Normalized ---
            clf.fit(X_norm_train, y_train)
            preds_norm = clf.predict(X_norm_test)
            f1_norm = f1_score(y_test, preds_norm, average=average_method)
            report_norm_dict = classification_report(y_test, preds_norm, output_dict=True, zero_division=0)
            report_norm_str = classification_report(y_test, preds_norm, zero_division=0)
            
            # --- Calculate MPD ---
            mpd = f1_norm - f1_orig
            
            # Store everything systematically
            mpd_results[clf_name] = {
                "F1 Original": f1_orig,
                "F1 Normalized": f1_norm,
                "MPD": mpd,
                "Report Original": report_orig_dict,
                "Report Normalized": report_norm_dict
            }
            
            print(f"[{clf_name}] Original F1: {f1_orig:.4f} | Normalized F1: {f1_norm:.4f} | MPD: {mpd:+.4f}")
            
            # Print full reports if the user wants deep diagnostics
            if verbose:
                print(f"\n--- {clf_name} Original Report ---")
                print(report_orig_str)
                print(f"--- {clf_name} Normalized Report ---")
                print(report_norm_str)
                print("="*50)
            
        return mpd_results
        
    def calculate_irs(self):
        """
        Calculates the Information Retention Score (IRS).
        Formula: Average Cosine Similarity between original and normalized document embeddings.
        """
        if self.embedding_model is None:
            print("No embedding model provided. Skipping IRS calculation.")
            return None
            
        print("Generating embeddings for original texts... (this may take a moment)")
        # We assume the user passes a model with a standard .encode() method 
        # (like SentenceTransformers)
        emb_orig = self.embedding_model.encode(self.texts_original)
        
        print("Generating embeddings for normalized texts...")
        emb_norm = self.embedding_model.encode(self.texts_normalized)
        
        # Calculate paired cosine distances. 
        # This compares doc_orig[0] with doc_norm[0], doc_orig[1] with doc_norm[1], etc.
        distances = paired_cosine_distances(emb_orig, emb_norm)
        
        # Cosine Similarity = 1 - Cosine Distance
        similarities = 1 - distances
        
        # IRS is the average similarity across the entire corpus
        irs = np.mean(similarities)
        
        print(f"Information Retention Score (IRS): {irs:.4f}")
        
        return irs
        
    def calculate_aes(self, cr, irs):
        """
        Calculates the Algorithm Effectiveness Score (AES).
        Formula: Harmonic mean of CR and IRS.
        """
        if cr is None or irs is None or (cr + irs) == 0:
            return 0.0
            
        aes = (2 * irs * cr) / (irs + cr)
        print(f"Algorithm Effectiveness Score (AES): {aes:.4f}")
        
        return aes
        
    def calculate_anld(self):
        """
        Calculates the Average Normalized Levenshtein Distance (ANLD).
        Formula: (1 / |V|) * sum( LD(w, sigma(w)) / |w| )
        """
        vocab_mapping = {}
        
        # Build a dictionary mapping each original word to its normalized form
        for orig_text, norm_text in zip(self.texts_original, self.texts_normalized):
            orig_tokens = orig_text.split()
            norm_tokens = norm_text.split()
            
            # We iterate up to the minimum length in case the user's algorithm 
            # unpredictably dropped or merged tokens, preventing index errors.
            min_len = min(len(orig_tokens), len(norm_tokens))
            for i in range(min_len):
                w = orig_tokens[i]
                sigma_w = norm_tokens[i]
                
                # Only add unique original words to the vocabulary V
                if w not in vocab_mapping:
                    vocab_mapping[w] = sigma_w
                    
        if not vocab_mapping:
            print("No valid vocabulary mapping could be created for ANLD.")
            return 0.0
            
        total_normalized_ld = 0.0
        valid_words = 0
        
        # Calculate the normalized distance for each word in the vocabulary
        for w, sigma_w in vocab_mapping.items():
            if len(w) == 0:
                continue # Prevent division by zero
                
            ld = Levenshtein.distance(w, sigma_w)
            total_normalized_ld += ld / len(w)
            valid_words += 1
            
        anld = total_normalized_ld / valid_words if valid_words > 0 else 0.0
        
        print(f"Average Normalized Levenshtein Distance (ANLD): {anld:.4f}")
        return anld

    def evaluate_all(self, test_size=0.2, random_state=42, verbose=False):
        """
        Executes the full Unified Multi-Metric Framework and returns the results.
        """
        print("--- 1. Macroscopic Evaluation ---")
        cr = self.calculate_cr()
        
        print("\n--- 2. Semantic Preservation ---")
        irs = self.calculate_irs()
        
        print("\n--- 3. Overall Effectiveness ---")
        aes = self.calculate_aes(cr, irs)
        
        print("\n--- 4. Micro-level Fidelity (Safety Gate) ---")
        anld = self.calculate_anld()
        
        print("\n--- 5. Downstream Impact ---")
        mpd = self.calculate_mpd(test_size=test_size, random_state=random_state, verbose=verbose)
        
        results = {
            "Compression Ratio (CR)": cr,
            "Information Retention Score (IRS)": irs,
            "Algorithm Effectiveness Score (AES)": aes,
            "Average Normalized Levenshtein Distance (ANLD)": anld,
            "Model Performance Delta (MPD)": mpd
        }
        
        return results