import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv(r'finaOutput.csv')

# Shuffle the rows
df_shuffled = df.sample(frac=1)

# Save the shuffled DataFrame back to a CSV file
df_shuffled.to_csv('finaOutput-2.csv', index=False)

final_nahab_2 = pd.read_csv(r'final-Dataset-2.csv')

# محاسبه سن افراد از تاریخ تولد
current_year = datetime.now().year
final_nahab_2['BIRTHDATE'] = pd.to_datetime(final_nahab_2['BIRTHDATE'], errors='coerce')
final_nahab_2['age'] = current_year - final_nahab_2['BIRTHDATE'].dt.year

# حذف ستون BIRTHDATE و اضافه کردن ستون age
final_nahab = final_nahab_2.drop(columns=['BIRTHDATE'])


# Split dataset into train and test sets
X, Y = final_nahab.drop(columns=['requiredInquiry']), final_nahab['requiredInquiry']
X_train, X_test, y_train, y_test = train_test_split(
    X,Y, test_size=0.3, random_state=42)

# Instantiate an XGBoost classifier
model = xgb.XGBClassifier(n_estimators=40, learning_rate=0.7, max_depth=6)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions
predictions = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print('predictions:', predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")
