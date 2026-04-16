🎬 Movie Review Sentiment Analysis

A Machine Learning web application that analyzes movie reviews and predicts whether the sentiment is Positive 😊 or Negative 😡 using Natural Language Processing (NLP).

📌 Project Overview

This project uses TF-IDF vectorization and a Logistic Regression model to classify text-based movie reviews. The model is deployed using Streamlit, allowing users to interact with it through a simple web interface.

🧠 Features
 Real-time sentiment prediction
 Confidence score display
 Clean and interactive UI (Streamlit)
 Example inputs for quick testing
 Word count display
 Sidebar with model details
🛠️ Tech Stack
Programming Language: Python
Libraries:
pandas
scikit-learn
streamlit
Machine Learning Model: Logistic Regression
Text Processing: TF-IDF Vectorizer
📂 Project Structure
sentiment_project/
│
├── data/
│   └── imdb.csv
│
├── model/
│   ├── sentiment_model.pkl
│   └── vectorizer.pkl
│
├── app.py
├── train.py
├── requirements.txt
└── README.md
⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/your-username/sentiment-analysis.git
cd sentiment-analysis
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Train the model
python train.py
4️⃣ Run the application
streamlit run app.py
📊 Model Details
Algorithm: Logistic Regression
Vectorization: TF-IDF (max_features=5000)
Dataset: IMDb Movie Reviews
Evaluation Metric: Accuracy Score

🎯 Future Improvements
🔹 Add multiple model comparison (SVM, Naive Bayes)
🔹 Integrate advanced NLP models (BERT)
🔹 Deploy on cloud (Streamlit Cloud / Render)
🔹 Add visualization dashboard (WordCloud, graphs)
💼 Use Case

This project can be used in:

Customer feedback analysis
Product review classification
Social media sentiment tracking
🙌 Acknowledgements
Dataset from IMDb
Built as part of a Machine Learning project

If you have any questions or suggestions, feel free to connect!
