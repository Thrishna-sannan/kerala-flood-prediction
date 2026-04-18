# Kerala Flood Prediction (2026)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report


# load data
data = pd.read_csv("C:/Users/thrishna/Downloads/kerala_flood_prediction_2026.csv")

data['flood_severity'] = data['flood_severity'].fillna('No Flood')
data = data.drop_duplicates()


# keep only numeric cols for ML
data_ml = data.drop(columns=['date','month_name','season','district','river_basin','wind_direction','land_use_type','imd_warning_level','alert_level','flood_severity'])


# scaling
scaler = MinMaxScaler()
cols = ['daily_rainfall_mm','monthly_rainfall_mm','flood_probability_score']
data_ml[cols] = scaler.fit_transform(data_ml[cols])


#Visualisation
plt.figure(figsize=(8,5))
plt.scatter(data_ml['daily_rainfall_mm'], data_ml['flood_probability_score'], alpha=0.3)
plt.title("Rainfall vs Flood Prob")
plt.grid()
plt.show()


plt.figure(figsize=(8,5))
monthly_avg = data.groupby('month')['daily_rainfall_mm'].mean()
plt.plot(monthly_avg.index, monthly_avg.values, marker='o')
plt.xticks(range(1,13), ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
plt.title("Monthly Rainfall Trend")
plt.grid()
plt.show()


plt.figure(figsize=(8,5))
plt.hist(data_ml['daily_rainfall_mm'], bins=30)
plt.title("Rainfall Distribution")
plt.show()


plt.figure(figsize=(10,8))
features = ['daily_rainfall_mm','monthly_rainfall_mm','antecedent_rainfall_7day_mm','soil_moisture_pct','river_level_m','reservoir_level_pct','flood_probability_score','flood_occurred']
sns.heatmap(data_ml[features].corr(), annot=True, fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()


plt.figure(figsize=(7,5))
sns.boxplot(x='flood_occurred', y='river_level_m', data=data_ml)
plt.title("River Level vs Flood")
plt.show()


plt.figure(figsize=(10,5))
avg_rain = data.groupby('month')['daily_rainfall_mm'].mean()
plt.bar(avg_rain.index, avg_rain.values)
plt.xticks(range(1,13), ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
plt.title("Monthly Avg Rainfall")
plt.grid(axis='y')
plt.show()


plt.figure(figsize=(7,7))
counts = data['flood_severity'].value_counts()
plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%')
plt.title("Flood Severity Distribution")
plt.show()


#Linear regression

x = data_ml[['daily_rainfall_mm']]
y = data_ml[['flood_probability_score']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=34)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("MSE:", mean_squared_error(y_test, y_pred))


plt.figure(figsize=(8,5))
plt.scatter(x, y, alpha=0.2)
plt.plot(x, model.predict(x), color='red')
plt.title("Regression Line")
plt.show()


test_val = pd.DataFrame({'daily_rainfall_mm':[0.5]})
print("Predicted Score:", model.predict(test_val))


#Logical regression

feature_cols = ['daily_rainfall_mm','monthly_rainfall_mm','antecedent_rainfall_7day_mm','soil_moisture_pct','river_level_m','reservoir_level_pct','flood_probability_score']

X = data_ml[feature_cols]
Y = data_ml['flood_occurred']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=34)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, Y_train)

preds = clf.predict(X_test)

print("Accuracy:", accuracy_score(Y_test, preds))
print(classification_report(Y_test, preds))


cm = confusion_matrix(Y_test, preds)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()


#Prediction

pred_score = model.predict(test_val)[0][0]
level = "LOW" if pred_score < 0.3 else "MEDIUM" if pred_score < 0.6 else "HIGH"

print(f"Flood Risk: {level} | Score: {round(pred_score, 4)}")

sample = pd.DataFrame([[0.5,0.5,0.4,0.6,0.5,0.5,pred_score]], columns=feature_cols)
print("Flood Occurrence:", "YES (HIGH)" if clf.predict(sample)[0] == 1 else "NO (LOW)")