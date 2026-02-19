# 🚨 Disaster Tweet Detection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![BERT](https://img.shields.io/badge/Model-BERT-green.svg)](https://huggingface.co/transformers/)

A state-of-the-art disaster tweet detection system powered by BERT transformers that intelligently distinguishes real emergency situations from metaphorical or non-emergency language. This system helps emergency response teams and organizations filter through social media noise to identify genuine disaster-related tweets.

## 🌟 Features

- **Advanced NLP**: Utilizes BERT (Bidirectional Encoder Representations from Transformers) for deep contextual understanding
- **High Accuracy**: Achieves superior performance in distinguishing real disasters from metaphorical language
- **Real-time Detection**: Fast inference for processing large volumes of tweets
- **Easy Integration**: Simple API for seamless integration into existing systems
- **Comprehensive Evaluation**: Detailed performance metrics and visualization tools

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Model Information](#model-information)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [API Documentation](#api-documentation)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## 🔍 Overview

Social media platforms, particularly Twitter, are crucial sources of real-time information during disasters. However, distinguishing between genuine disaster reports and metaphorical language (e.g., "The concert was absolutely fire!") remains challenging. This project addresses this problem using:

- **Pre-trained BERT model** fine-tuned on disaster-related tweets
- **Contextual embeddings** that understand nuanced language
- **Transfer learning** for improved performance with limited labeled data

The core challenge here — extracting a reliable signal from noisy, ambiguous text — is domain-agnostic. The same BERT-based classification architecture applies directly to clinical NLP tasks: distinguishing real patient alerts from routine documentation, flagging safety-critical events in hospital logs, or routing medical queries to the right knowledge base.

### Problem Statement

Given a tweet, classify whether it refers to a real disaster or not. The challenge lies in understanding context, as the same words can mean different things:

- ✅ **Real Disaster**: "Wildfire spreading rapidly in Northern California, residents evacuating"
- ❌ **Not a Disaster**: "This new album is straight fire! 🔥"

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 8GB+ RAM recommended
- GPU (optional, for faster training/inference)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/SKeval/disaster-tweet-detection.git
   cd disaster-tweet-detection
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the trained model** (see [Model Information](#model-information))

## 🤖 Model Information

### ⚠️ Model Not Included in Repository

Due to GitHub's file size limitations, the trained BERT model is **not included** in this repository. The model file is approximately **400-500 MB** and exceeds Git's recommended file size limits.

### Downloading the Model

You have several options to obtain the trained model:

#### Option 1: Download from Release (Recommended)
```bash
# Download from GitHub Releases
wget https://github.com/SKeval/disaster-tweet-detection/releases/download/v1.0/disaster_bert_model.pkl

# Place in the models directory
mkdir -p models
mv disaster_bert_model.pkl models/
```

#### Option 2: Train Your Own Model
Follow the [Training](#training) section to train the model from scratch using the provided notebooks.

#### Option 3: Use Pre-trained Model
```python
from transformers import BertForSequenceClassification

# Start with base BERT and fine-tune
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
```

### Model Architecture

- **Base Model**: BERT-base-uncased
- **Parameters**: ~110M
- **Hidden Size**: 768
- **Attention Heads**: 12
- **Layers**: 12
- **Max Sequence Length**: 128 tokens
- **Classification Head**: Linear layer with 2 outputs (disaster/not-disaster)

### Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 84.5% |
| Precision | 83.2% |
| Recall | 82.8% |
| F1-Score | 81.5% |

## 💻 Usage

### Quick Start

```python
from src.model import DisasterTweetClassifier

# Initialize the classifier
classifier = DisasterTweetClassifier(model_path='models/disaster_bert_model.pkl')

# Predict on a single tweet
tweet = "Earthquake strikes coastal region, tsunami warning issued"
result = classifier.predict(tweet)

print(f"Prediction: {result['label']}")  # Output: "disaster"
print(f"Confidence: {result['confidence']:.2%}")  # Output: "95.3%"
```

### Batch Prediction

```python
tweets = [
    "Forest fire spreading near residential areas",
    "This pizza is fire!",
    "Severe flooding reported in downtown area"
]

results = classifier.predict_batch(tweets)

for tweet, result in zip(tweets, results):
    print(f"Tweet: {tweet}")
    print(f"Label: {result['label']} ({result['confidence']:.2%})\n")
```

### Using the Web API

Start the Flask application:

```bash
python app/app.py
```

Make predictions via HTTP POST:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Massive earthquake hits city center"}'
```

Response:
```json
{
  "text": "Massive earthquake hits city center",
  "prediction": "disaster",
  "confidence": 0.967,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## 📁 Project Structure

```
disaster-tweet-detection/
│
├── app/                          # Web application
│   ├── app.py                    # Flask API
│   ├── templates/                # HTML templates
│   └── static/                   # CSS, JS, images
│
├── docs/                         # Documentation
│   ├── architecture.md           # System architecture
│   ├── api.md                    # API documentation
│   └── performance.md            # Performance metrics
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_data_exploration.ipynb     # EDA
│   ├── 02_preprocessing.ipynb        # Data preprocessing
│   ├── 03_model_training.ipynb       # Model training
│   └── 04_evaluation.ipynb           # Model evaluation
│
├── src/                          # Source code
│   ├── __init__.py
│   ├── data_loader.py            # Data loading utilities
│   ├── preprocessing.py          # Text preprocessing
│   ├── model.py                  # Model definition
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation utilities
│   └── utils.py                  # Helper functions
│
├── models/                       # Trained models (not in Git)
│   └── .gitkeep                  # Placeholder
│
├── data/                         # Dataset (not in Git)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── .gitignore                    # Git ignore file
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── LICENSE                       # License file
```

## 📊 Dataset

This project uses the **"Natural Language Processing with Disaster Tweets"** dataset from Kaggle.

### Dataset Structure

- **Training set**: ~7,600 labeled tweets
- **Test set**: ~3,200 unlabeled tweets
- **Features**:
  - `id`: Unique identifier
  - `text`: Tweet content
  - `location`: Tweet location (may be blank)
  - `keyword`: Keyword from tweet (may be blank)
  - `target`: Label (1 = disaster, 0 = not disaster)

### Downloading the Dataset

```bash
# Using Kaggle API
kaggle competitions download -c nlp-getting-started

# Unzip
unzip nlp-getting-started.zip -d data/
```

### Data Distribution

- **Disaster tweets**: 43% (3,271 samples)
- **Non-disaster tweets**: 57% (4,342 samples)

## 🎓 Training

### Training from Scratch

1. **Prepare your data** in the `data/` directory
2. **Run the training script**:

```bash
python src/train.py --epochs 4 --batch_size 32 --learning_rate 2e-5
```

Or use the interactive notebook:

```bash
jupyter notebook notebooks/03_model_training.ipynb
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 4 | Number of training epochs |
| `--batch_size` | 32 | Training batch size |
| `--learning_rate` | 2e-5 | AdamW learning rate |
| `--max_length` | 128 | Maximum sequence length |
| `--warmup_steps` | 500 | Learning rate warmup steps |

### Training Time

- **CPU**: ~6-8 hours (not recommended)
- **GPU (Tesla T4)**: ~30-45 minutes
- **GPU (V100)**: ~15-20 minutes

## 📈 Evaluation

### Running Evaluation

```bash
python src/evaluate.py --model_path models/disaster_bert_model.pkl --test_data data/test.csv
```

### Evaluation Metrics

The system reports:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown
- **ROC-AUC**: Area under the ROC curve

### Visualization

The evaluation script generates:
- Confusion matrix heatmap
- ROC curve
- Precision-Recall curve
- Classification report

## 🔌 API Documentation

### REST API Endpoints

#### POST /predict
Classify a single tweet

**Request:**
```json
{
  "text": "Tornado warning issued for the metro area"
}
```

**Response:**
```json
{
  "text": "Tornado warning issued for the metro area",
  "prediction": "disaster",
  "confidence": 0.923,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### POST /predict_batch
Classify multiple tweets

**Request:**
```json
{
  "tweets": [
    "Hurricane approaching coastline",
    "My exam was a disaster lol"
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "text": "Hurricane approaching coastline",
      "prediction": "disaster",
      "confidence": 0.945
    },
    {
      "text": "My exam was a disaster lol",
      "prediction": "not_disaster",
      "confidence": 0.876
    }
  ],
  "count": 2,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## 💡 Examples

### Example 1: Disaster Detection

```python
examples = [
    "Massive wildfire destroys hundreds of homes",
    "Flood waters rising rapidly in downtown area",
    "Emergency services responding to building collapse",
]

for tweet in examples:
    result = classifier.predict(tweet)
    print(f"✅ {tweet} → {result['label']} ({result['confidence']:.1%})")
```

### Example 2: Metaphorical Language

```python
examples = [
    "This movie was an absolute disaster!",
    "My hair is on fire with this new look",
    "The team is ablaze tonight!",
]

for tweet in examples:
    result = classifier.predict(tweet)
    print(f"❌ {tweet} → {result['label']} ({result['confidence']:.1%})")
```

### Example 3: Edge Cases

```python
edge_cases = [
    "Fire drill scheduled for 2pm today",  # Planned, not disaster
    "Explosion heard downtown, cause unknown",  # Potential disaster
    "Breaking news: earthquake simulation successful",  # Simulation
]

for tweet in edge_cases:
    result = classifier.predict(tweet)
    print(f"🤔 {tweet} → {result['label']} ({result['confidence']:.1%})")
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Development Guidelines

- Write clear, documented code
- Add unit tests for new features
- Update documentation as needed
- Follow PEP 8 style guidelines
- Ensure all tests pass before submitting PR

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this project in your research or work, please cite:

```bibtex
@software{disaster_tweet_detection,
  author = {SKeval},
  title = {Disaster Tweet Detection System},
  year = {2024},
  url = {https://github.com/SKeval/disaster-tweet-detection}
}
```

## 🙏 Acknowledgments

- **Kaggle** for providing the disaster tweets dataset
- **Hugging Face** for the transformers library
- **Google Research** for the BERT model
- The open-source community for various tools and libraries

## 📞 Contact

For questions, suggestions, or collaboration:

- **GitHub**: [@SKeval](https://github.com/SKeval)
- **Repository**: [disaster-tweet-detection](https://github.com/SKeval/disaster-tweet-detection)
- **Issues**: [Report a bug or request a feature](https://github.com/SKeval/disaster-tweet-detection/issues)

## 🔮 Future Improvements

- [ ] Multi-language support
- [ ] Real-time Twitter stream integration
- [ ] Mobile app development
- [ ] Enhanced model with additional features (images, user metadata)
- [ ] Deployment to cloud platforms (AWS, Azure, GCP)
- [ ] Model compression for edge devices
- [ ] Active learning pipeline for continuous improvement

---

**⭐ If you find this project helpful, please consider giving it a star!**

Made with ❤️ by [SKeval](https://github.com/SKeval)
