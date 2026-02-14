import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Ensure NLTK resources are available
def _download_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt',quiet=True)
    
    
_download_nltk()


class TweetPreProcessor:
    """
    Tweet text preprocessing for baseline ML models.
    For BERT later, we'll use a lighter cleaning.
    """
    
    def __init__(self, remove_stopwords: bool = True, min_word_len: int= 3):
        self.remove_stopwords = remove_stopwords
        self.min_word_len = min_word_len
        self.stop_words = set(stopwords.words('english'))
        
    def clean_tweet(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+", "", text)

        # Remove HTML entities
        text = re.sub(r"&\w+;", " ", text)

        # Remove @mentions
        text = re.sub(r"@\w+", " ", text)

        # Keep hashtag text but drop '#'
        text = re.sub(r"#", " ", text)

        # Remove numbers
        text = re.sub(r"\d+", " ", text)

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text
    
    def tokenize(self, text: str):
        # Simple whitespace tokenization after cleaning
        return text.split()
    
    def preprocess_text(self, text: str) -> str:
        text = self.clean_tweet(text)
        tokens = self.tokenize(text)

        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]

        # Remove very short tokens
        tokens = [t for t in tokens if len(t) >= self.min_word_len]

        return " ".join(tokens)
    
    def preprocess_dataframe(self, df:pd.DataFrame, text_col: str = 'text', new_col: str = 'text_clean') -> pd.DataFrame: 
        df = df.copy()
        df[new_col] = df[text_col].astype(str).apply(self.preprocess_text)
        return df
    


def main():
    train_path = "data/raw/train.csv"
    test_path = "data/raw/test.csv"
    out_train = "data/processed/train_clean.csv"
    out_test = "data/processed/test_clean.csv"
    
    print("🔹 Loading raw data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"  Train shape: {train_df.shape}")
    print(f"  Test shape:  {test_df.shape}")
    
    preprocessor = TweetPreProcessor(remove_stopwords=True, min_word_len=3)

    print("🔹 Preprocessing training data...")
    train_clean = preprocessor.preprocess_dataframe(train_df, text_col="text", new_col="text_clean")

    print("🔹 Preprocessing test data...")
    test_clean = preprocessor.preprocess_dataframe(test_df, text_col="text", new_col="text_clean")

# Ensure processed folder exists
    import os

    os.makedirs("data/processed", exist_ok=True)

    print(f"🔹 Saving to {out_train} and {out_test} ...")
    train_clean.to_csv(out_train, index=False)
    test_clean.to_csv(out_test, index=False)

    print("✅ Preprocessing done.")
    print("  New columns: 'text_clean' added to both train and test.")


if __name__ == "__main__":
    main()