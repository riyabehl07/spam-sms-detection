# Spam SMS Detection

## Overview
This project is a **Spam SMS Detection System** that classifies messages as spam or ham (not spam) using **Natural Language Processing (NLP)** techniques and **Machine Learning**. The dataset used contains labeled SMS messages.

## Features
- Preprocesses text data (removes stopwords, tokenization, etc.).
- Implements a Machine Learning model for classification.
- Provides a user interface or CLI for testing new messages.
- Uses `nltk`, `pandas`, and `scikit-learn` for data processing and modeling.

## Technologies Used
- **Python**
- **NLTK (Natural Language Toolkit)**
- **Pandas**
- **Scikit-learn**
- **Flask** (if applicable for web deployment)

## Installation
### Prerequisites
Ensure you have **Python 3.8+** installed. You can check your Python version by running:
```bash
python --version
```

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/spam-sms-detection.git
cd spam-sms-detection
```

### Step 2: Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage
### Running the Application
To start the spam detection script, run:
```bash
python app.py
```

### Testing with Sample Input
You can test the model by inputting an SMS message when prompted.

## Dataset
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
The dataset used in this project consists of **5572 labeled SMS messages**:
- `ham` (Not Spam): 4825 messages
- `spam`: 747 messages

## Training and Model Details
- **Data Preprocessing:** Tokenization, stopword removal, and text vectorization using TF-IDF.
- **Model Used:** Naïve Bayes / Logistic Regression (or any other specified model).
- **Evaluation Metrics:** Accuracy, Precision, Recall, and F1-score.

## Troubleshooting
If you encounter an error related to missing NLTK resources, run:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Future Improvements
- Deploying as a web service using Flask/Django.
- Adding deep learning-based NLP models for improved accuracy.
- Extending support for multilingual SMS detection.

## License
This project is licensed under the MIT License.

## Author
**Riya Behl**

Feel free to contribute by raising issues or submitting pull requests! 

