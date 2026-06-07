import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("Sales Forecasting Dashboard")

uploaded_file = st.file_uploader(
    "Upload Sales CSV",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.stop()

df["Day"] = range(1, len(df) + 1)

X = df[["Day"]]
y = df["Sales"]

model = LinearRegression()
model.fit(X, y)

future_days = pd.DataFrame({
    "Day": [16, 17, 18, 19, 20]
})

predictions = model.predict(future_days)

forecast_df = pd.DataFrame({
    "Day": future_days["Day"],
    "Predicted Sales": predictions
})

st.subheader("Historical Sales Data")
st.dataframe(df)

st.subheader("Forecast")
st.dataframe(forecast_df)

fig, ax = plt.subplots()

ax.plot(df["Day"], df["Sales"], marker="o", label="Historical")

ax.plot(
    forecast_df["Day"],
    forecast_df["Predicted Sales"],
    marker="o",
    label="Forecast"
)

ax.set_xlabel("Day")
ax.set_ylabel("Sales")
ax.legend()

st.pyplot(fig)