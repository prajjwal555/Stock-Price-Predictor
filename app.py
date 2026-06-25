import streamlit as st
from predictor import predict_stock

st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Price Predictor")

st.write(
    """
    This application predicts whether a stock price is likely
    to move **UP** or **DOWN** on the next trading day using
    Machine Learning.
    """
)

st.divider()

ticker = st.text_input(
    "Enter Stock Symbol",
    value="RELIANCE.NS"
)

end_date = st.date_input(
    "Select End Date"
)
if st.button("Predict"):

    with st.spinner("Running Machine Learning Model..."):

       result = predict_stock(ticker,"2018-01-01",str(end_date))
   

    st.success("Prediction Complete!")

    st.subheader("Prediction")

    st.metric(
        "Prediction",
        result["prediction"]
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "UP Probability",
            f"{result['up_probability']:.2f}%"
        )

    with col2:
        st.metric(
            "DOWN Probability",
            f"{result['down_probability']:.2f}%"
        )

    st.metric(
        "Accuracy",
        f"{result['accuracy']*100:.2f}%"
    )

    st.subheader("Confusion Matrix")

    st.image(
        result["confusion_matrix"]
    )

    st.subheader("Feature Importance")

    st.image(
        result["feature_importance"]
    )

    with open(result["csv"], "rb") as f:

        st.download_button(
            "Download Predictions CSV",
            f,
            file_name="predictions.csv"
        )