import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

final_nahab = pd.read_csv(r'C:\Users\Dps\Desktop\Review\csvFile\finalRecord\finaOutput.csv')

# رسم نمودار countplot برای بررسی توزیع کد تراکنش بر اساس وضعیت استعلام
plt.figure(figsize=(12, 6))
sns.countplot(x='TRANSACTIONCODE', hue='requiredInquiry', data=final_nahab)
plt.title('Distribution of Transaction Codes by Inquiry Status')
plt.xlabel('Transaction Code')
plt.ylabel('Count')
plt.legend(title='Inquiry Status', loc='upper right')
plt.show()


# محاسبه سن افراد از تاریخ تولد
current_year = datetime.now().year
final_nahab['BIRTHDATE'] = pd.to_datetime(final_nahab['BIRTHDATE'], errors='coerce')
final_nahab['age'] = current_year - final_nahab['BIRTHDATE'].dt.year

# حذف ستون BIRTHDATE و اضافه کردن ستون age
final_nahab = final_nahab.drop(columns=['BIRTHDATE'])

# تبدیل ستونهای غیر عددی به دادههای عددی
final_nahab['CUSTOMERTYPE'] = final_nahab['CUSTOMERTYPE'].astype(float)
final_nahab['TRANSACTIONCODE'] = final_nahab['TRANSACTIONCODE'].astype(float)
final_nahab['CIF'] = final_nahab['CIF'].astype(float)
final_nahab['POSTALCODE'] = final_nahab['POSTALCODE'].astype(float)
final_nahab['requiredInquiry'] = final_nahab['requiredInquiry'].astype(float)

# محاسبه ماتریس همبستگی
corr_matrix = final_nahab.corr()

# رسم ماتریس همبستگی
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

#مشاهده نمودار هیستوگرام داده ها به صورت یکجا 
final_nahab.hist(figsize=(12,8) , grid=False)
plt.show()

print("Missing Value:\n" , final_nahab.isnull().sum())

#پر کردن داده های مفقودی
def handle_missing_values(final_nahab):
    #پر کردن داده های کمی با میانه
    for col in final_nahab.select_dtypes(include=['number']).columns:
        final_nahab[col].fillna(final_nahab[col].median(), inplace=True)
    #پر کردن داده های کیفی با مد 
    for col in final_nahab.select_dtypes(include=['object']).columns:
        final_nahab[col].fillna(final_nahab[col].mode()[0], inplace=True)

    return final_nahab

final_nahab = handle_missing_values(final_nahab)
print(final_nahab.isnull().sum())
