# SMS Spam Classifier

## Overview
The **SMS Spam Classifier** is a machine learning-based project that automatically detects spam messages in SMS texts. Using Natural Language Processing (NLP) and classification algorithms, this system can distinguish between spam and legitimate (ham) messages.

## Features
- **Text Preprocessing**: Tokenization, stop-word removal, stemming/lemmatization.
- **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) or Count Vectorizer.
- **Machine Learning Models**: Naive Bayes, Logistic Regression, Random Forest, or Deep Learning models.
- **Real-time Prediction**: Classify incoming messages as spam or ham.
- **Web Interface**: A Streamlit-based UI for easy interaction.

## Dataset
The project uses the **SMS Spam Collection Dataset**

## Installation
### Prerequisites
- Python 3.x
- Jupyter Notebook or any Python IDE
- Required libraries: 
  ```sh
  pip install numpy pandas scikit-learn nltk streamlit
  ```

## Usage
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/sms-spam-classifier.git
   cd sms-spam-classifier
   ```
2. Run the Jupyter Notebook for model training:
   ```sh
   jupyter notebook sms_spam_classifier.ipynb
   ```
3. Train the model and evaluate performance.
4. Launch the Streamlit web app:
   ```sh
   streamlit run app.py
   ```
5. Enter SMS text and classify messages as spam or ham.

## Project Structure
```
📂 sms-spam-classifier
├── data
│   ├── spam.csv  # Dataset
├── notebooks
│   ├── sms_spam_classifier.ipynb  # Model training
├── models
│   ├── model.pkl  # Saved trained model
├── app.py  # Streamlit web app
├── README.md  # Project documentation
```

## Model Performance
The classifier is evaluated using accuracy, precision, recall, and F1-score metrics. Sample results:
- **Naive Bayes**: Accuracy ~98%

## Future Improvements
- Improve accuracy using Deep Learning models (LSTMs, Transformers).
- Deploy as an API for real-time spam detection.
- Enhance the web interface with a mobile-friendly design.



