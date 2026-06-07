import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("sales_data.csv")

df["Day"] = range(1, len(df) + 1)

X = df[["Day"]]
y = df["Sales"]

model = LinearRegression()

model.fit(X, y)

future_days = pd.DataFrame({
    "Day": [16, 17, 18, 19, 20]
})

predictions = model.predict(future_days)

print("Future Sales Forecast")

for day, pred in zip(future_days["Day"], predictions):
    print(f"Day {day}: {pred:.2f}")