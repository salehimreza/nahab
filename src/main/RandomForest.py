import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

final_nahab = pd.read_csv(r'finaOutput.csv')

current_year = datetime.now().year
final_nahab['BIRTHDATE'] = pd.to_datetime(final_nahab['BIRTHDATE'], errors='coerce')
final_nahab['age'] = current_year - final_nahab['BIRTHDATE'].dt.year

final_nahab = final_nahab.drop(columns=['BIRTHDATE'])
from sklearn.model_selection import train_test_split

x = final_nahab.drop('requiredInquiry', axis=1)
y = final_nahab['requiredInquiry']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

dtree = RandomForestClassifier()

dtree.fit(X_train,y_train)
prediction = dtree.predict(X_test)
from sklearn.metrics import classification_report , confusion_matrix

print(confusion_matrix(y_test,prediction))
print('\n')
print(classification_report(y_test,prediction))
