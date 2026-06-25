# 📈 Stock Price Predictor using Machine Learning

A Machine Learning-based Stock Price Prediction application that predicts whether a stock price is likely to move **UP 📈** or **DOWN 📉** on the next trading day.

The project uses technical indicators, feature engineering, Random Forest, Logistic Regression, and an interactive Streamlit web application.

---

# 🚀 Live Features

✅ Predict next day's stock movement

✅ Random Forest Classifier

✅ Logistic Regression

✅ Automatic Feature Engineering

✅ Technical Indicators (RSI, EMA, MACD, Bollinger Bands)

✅ Probability Prediction

✅ Feature Importance Graph

✅ Confusion Matrix

✅ Download Prediction CSV

✅ Interactive Streamlit Web Application

---

# 🖥️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- yFinance API
- Matplotlib
- Streamlit

---

# 📊 Machine Learning Pipeline

```
User Input
     │
     ▼
Download Stock Data (Yahoo Finance)
     │
     ▼
Feature Engineering
     │
     ├── SMA
     ├── EMA
     ├── RSI
     ├── MACD
     ├── Bollinger Bands
     ├── Momentum
     ├── Volatility
     └── Volume Change
     │
     ▼
Train/Test Split
     │
     ▼
Data Standardization
     │
     ▼
Random Forest
        +
Logistic Regression
     │
     ▼
Model Evaluation
     │
     ▼
Best Model Selection
     │
     ▼
Next Day Prediction
     │
     ▼
Probability + Charts + CSV
```

---

# 📂 Project Structure

```
Stock-Price-Predictor/

│── app.py
│── predictor.py
│── requirements.txt
│── .gitignore

│── outputs/
│      ├── confusion_matrix.png
│      ├── feature_importance.png
│      └── predictions.csv
```

---

# 📈 Features Created

The model generates multiple technical indicators automatically.

- SMA (5,20,50)
- EMA (10,30)
- RSI
- MACD
- Daily Return
- Volatility
- Price Position
- Volume Change
- SMA Cross
- Momentum
- Bollinger Bands

These engineered features help the ML models learn market behaviour more effectively.

---

# 🤖 Models Used

## Random Forest

- 300 Decision Trees
- Max Depth = 8
- Balanced Classes
- Random State = 42

---

## Logistic Regression

- Max Iterations = 1000

---

# 📊 Model Evaluation

The application displays

- Accuracy
- Classification Report
- Confusion Matrix
- Feature Importance

It automatically selects the model with the highest accuracy.

---

# 📥 Installation

Clone the repository

```bash
git clone https://github.com/prajjwal555/Stock-Price-Predictor.git
```

Go inside the project

```bash
cd Stock-Price-Predictor
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

# 📸 Application Preview

*(Add screenshots here after uploading them.)*

Example:

```
images/homepage.png

images/result.png
```

---

# 📊 Outputs

The project automatically generates

- Prediction CSV
- Confusion Matrix
- Feature Importance Graph

inside the **outputs/** folder.

---

# ⚠️ Challenges Faced During Development

During development several practical ML and software engineering challenges were solved:

- Handling MultiIndex columns returned by Yahoo Finance.
- Designing reusable prediction functions instead of a standalone script.
- Converting console ML code into a Streamlit application.
- Managing NaN and Infinite values created by technical indicators.
- Standardizing features before prediction.
- Integrating two ML models into one prediction pipeline.
- Automatically selecting the better-performing model.
- Exporting prediction reports.
- Creating visual analytics (Feature Importance & Confusion Matrix).
- Organizing the project into a modular structure.

These challenges improved understanding of Machine Learning deployment and software engineering practices.

---

# 🔮 Future Improvements

- LSTM Deep Learning Model
- XGBoost
- LightGBM
- Multiple Stock Comparison
- Candlestick Charts
- News Sentiment Analysis
- Portfolio Recommendation
- Model Hyperparameter Optimization
- Real-Time Prediction
- Cloud Deployment

---

# 👨‍💻 Author

**Prajjwal Yadav**

B.Tech Computer Science Engineering

Netaji Subhas University of Technology (NSUT)

---

# ⭐ If you found this project useful, consider giving it a Star!
