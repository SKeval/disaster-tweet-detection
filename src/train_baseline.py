import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path: str = "data/processed/train_clean.csv"):
    print(f"🔹 Loading processed data from {path}")
    df = pd.read_csv(path)
    assert "text_clean" in df.columns, "text_clean column missing – run data_preprocessing first."
    assert "target" in df.columns, "target column missing in train dataset."

    print(f"  Shape: {df.shape}")
    return df

def build_tfidf(X_train_raw, X_val_raw, max_features: int = 5000):
    print("🔹 Building TF-IDF features...")

    # Handle NaN values - fill with empty strings
    X_train_raw = X_train_raw.fillna("")
    X_val_raw = X_val_raw.fillna("")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(X_train_raw)
    X_val = vectorizer.transform(X_val_raw)

    print(f"  Train TF-IDF shape: {X_train.shape}")
    print(f"  Val TF-IDF shape:   {X_val.shape}")
    return vectorizer, X_train, X_val

def train_and_eval_model(name, model, X_train, y_train, X_val, y_val):
    print(f"\n================= {name} =================")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification report:\n")
    print(classification_report(y_val, y_pred, digits=4))

    cm = confusion_matrix(y_val, y_pred)
    plot_confusion_matrix(cm, name)

    return {
        "name": name,
        "model": model,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not disaster", "Disaster"],
        yticklabels=["Not disaster", "Disaster"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix – {title}")
    os.makedirs("docs", exist_ok=True)
    out_path = os.path.join("docs", f"cm_{title.replace(' ', '_').lower()}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Confusion matrix saved to {out_path}")


def main():
    # 1. Load data
    df = load_data("data/processed/train_clean.csv")

    X = df["text_clean"]
    y = df["target"]

    # 2. Train/validation split
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"Train size: {X_train_raw.shape[0]}, Val size: {X_val_raw.shape[0]}")

    # 3. TF-IDF features
    vectorizer, X_train, X_val = build_tfidf(X_train_raw, X_val_raw, max_features=5000)

    # 4. Define models
    models = []

    # Logistic Regression
    lr = LogisticRegression(
        max_iter=1000,
        C=1.0,
    )
    models.append(("Logistic Regression", lr))

    # Multinomial Naive Bayes
    nb = MultinomialNB(alpha=1.0)
    models.append(("Multinomial NB", nb))

    # Linear SVM
    svm = LinearSVC(C=1.0)
    models.append(("Linear SVM", svm))

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    models.append(("XGBoost", xgb))

    results = []

    # 5. Train and evaluate each model
    for name, model in models:
        res = train_and_eval_model(name, model, X_train, y_train, X_val, y_val)
        results.append(res)

    # 6. Compare models
    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    print("\n================= MODEL COMPARISON =================")
    print(results_df[["name", "accuracy", "precision", "recall", "f1"]])

    # 7. Save best model + vectorizer
    best = results_df.iloc[0]
    best_name = best["name"]
    print(f"\n✅ Best model: {best_name} (F1={best['f1']:.4f})")

 

    os.makedirs("models/baseline", exist_ok=True)
    for res in results:
        fname = res["name"].replace(" ", "_").lower()
        joblib.dump(res["model"], f"models/baseline/{fname}.pkl")
        print(f"  Saved {res['name']} to models/baseline/{fname}.pkl")

    joblib.dump(vectorizer, "models/baseline/tfidf_vectorizer.pkl")
    print("  Saved TF-IDF vectorizer to models/baseline/tfidf_vectorizer.pkl")



if __name__ == "__main__":
    main()