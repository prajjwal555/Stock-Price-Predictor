# Stock Predictor V2
# Upgraded version with:
# - Random Forest
# - Logistic Regression
# - EMA Features
# - Bollinger Bands
# - Momentum
# - Probability Prediction

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_data(ticker, start_date, end_date):

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.dropna(inplace=True)

    return df

def add_features(df):

    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()

    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA_30"] = df["Close"].ewm(span=30, adjust=False).mean()

    df["Daily_Return"] = df["Close"].pct_change() * 100

    df["Volatility"] = df["Daily_Return"].rolling(10).std()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26

    df["Price_Position"] = (
        (df["Close"] - df["Low"]) /
        (df["High"] - df["Low"] + 1e-9)
    )

    df["Volume_Change"] = df["Volume"].pct_change() * 100

    df["SMA_Cross"] = (df["SMA_5"] > df["SMA_20"]).astype(int)

    df["Momentum_10"] = df["Close"] - df["Close"].shift(10)

    df["BB_Middle"] = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()

    df["BB_Upper"] = df["BB_Middle"] + 2 * std
    df["BB_Lower"] = df["BB_Middle"] - 2 * std

    df["BB_Distance"] = (
        (df["Close"] - df["BB_Middle"])
        / (df["BB_Middle"] + 1e-9)
    )

    df["Target"] = (
        df["Close"].shift(-1) > df["Close"]
    ).astype(int)

    df.dropna(inplace=True)

    return df

def prepare_data(df):

    FEATURES = [
        "SMA_5",
        "SMA_20",
        "SMA_50",
        "EMA_10",
        "EMA_30",
        "RSI",
        "MACD",
        "Daily_Return",
        "Volatility",
        "Price_Position",
        "Volume_Change",
        "SMA_Cross",
        "Momentum_10",
        "BB_Distance"
    ]

    X = df[FEATURES].copy()
    y = df["Target"]

# Replace infinity values
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove rows containing NaN
    valid_rows = X.notna().all(axis=1)

    X = X[valid_rows]
    y = y[valid_rows]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )
    print(X.isnull().sum())
    print(np.isinf(X).sum())
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, FEATURES

def train_random_forest(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)
    return model

def train_logistic(X_train, y_train):

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model

def evaluate_model(name, model, X_test, y_test):

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    print(f"\n{name}")
    print("=" * 40)
    print(f"Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, pred))

    return pred, acc

def plot_feature_importance(model, features):

    importances = model.feature_importances_

    idx = np.argsort(importances)

    plt.figure(figsize=(8,5))
    plt.barh(
        [features[i] for i in idx],
        importances[idx]
    )

    plt.title("Feature Importance")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "feature_importance.png"
        )
    )

    plt.close()

def save_confusion_matrix(y_test, pred):

    cm = confusion_matrix(y_test, pred)

    fig, ax = plt.subplots(figsize=(5,4))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot(ax=ax)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "confusion_matrix.png"
        )
    )

    plt.close()

def predict_stock(ticker, start_date, end_date):

    print("Downloading Data...")

    df = download_data(
        ticker,
        start_date,
        end_date
    )

    print("Creating Features...")

    df = add_features(df)

    X_train, X_test, y_train, y_test, FEATURES = prepare_data(df)

    rf_model = train_random_forest(
        X_train,
        y_train
    )

    log_model = train_logistic(
        X_train,
        y_train
    )

    rf_pred, rf_acc = evaluate_model(
        "Random Forest",
        rf_model,
        X_test,
        y_test
    )

    log_pred, log_acc = evaluate_model(
        "Logistic Regression",
        log_model,
        X_test,
        y_test
    )

    best_model = rf_model if rf_acc >= log_acc else log_model

    latest = X_test[-1:]

    probs = best_model.predict_proba(latest)[0]

    save_confusion_matrix(y_test, rf_pred)

    if hasattr(rf_model, "feature_importances_"):
        plot_feature_importance(
            rf_model,
            FEATURES
        )

    export = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": rf_pred
    })

    csv_path = os.path.join(
        OUTPUT_DIR,
        "predictions.csv"
    )

    export.to_csv(
        csv_path,
        index=False
    )

    return {
        "prediction": "UP" if probs[1] > probs[0] else "DOWN",
        "up_probability": probs[1] * 100,
        "down_probability": probs[0] * 100,
        "accuracy": max(rf_acc, log_acc),
        "confusion_matrix": os.path.join(
            OUTPUT_DIR,
            "confusion_matrix.png"
        ),
        "feature_importance": os.path.join(
            OUTPUT_DIR,
            "feature_importance.png"
        ),
        "csv": csv_path
    }
    return {
        "prediction": "UP" if probs[1] > probs[0] else "DOWN",
        "up_probability": probs[1] * 100,
        "down_probability": probs[0] * 100,
        "accuracy": max(rf_acc, log_acc),
        "confusion_matrix": os.path.join(
            OUTPUT_DIR,
            "confusion_matrix.png"
        ),
        "feature_importance": os.path.join(
            OUTPUT_DIR,
            "feature_importance.png"
        ),
        "csv": csv_path
    }