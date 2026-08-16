import numpy as np
from sklearn.linear_model import LinearRegression

hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
scores = np.array([35, 40, 50, 55, 60, 65, 70, 80, 90])

model = LinearRegression()
model.fit(hours, scores)

study_hours = float(input("Enter study hours: "))

prediction = model.predict([[study_hours]])

print("Predicted Score:", round(prediction[0], 2))
