import pandas as pd

df_nahab2 = pd.read_csv(r'C:\Users\Dps\Desktop\Review\csvFile\finalRecord\updateNahab2.csv')
df_nahab4 = pd.read_csv(r'C:\Users\Dps\Desktop\Review\csvFile\finalRecord\updateNahab4.csv')


merged_df = pd.merge(df_nahab2, df_nahab4, on=['CIF'], how='outer', suffixes=('_df1', '_df2'))
#fill null postalcode with 0
merged_df['POSTALCODE'] = merged_df['POSTALCODE'].fillna('0')
#removes the decimal part from the POSTALCODE values
merged_df['POSTALCODE'] = merged_df['POSTALCODE'].map(lambda x: str(int(x)))

#removes ..:..:.. from the date format in the BIRTHDATE_df2 column
merged_df['BIRTHDATE_df2'] = merged_df['BIRTHDATE_df2'].str.split(' ').str[0]
#جایگزینی مقادیر خالی با صفر برایcustomerType
merged_df['CUSTOMERTYPE_df2'] = merged_df['CUSTOMERTYPE_df2'].fillna('0')
merged_df['CUSTOMERTYPE_df2'] = merged_df['CUSTOMERTYPE_df2'].map(lambda x: str(int(x)))

result_df = merged_df[['CIF', 'TRANSACTIONCODE', 'BIRTHDATE_df2', 'CUSTOMERTYPE_df2', 'POSTALCODE']]
transactioinCode =set(merged_df['TRANSACTIONCODE'])
my_dict = {value: index for index, value in enumerate(transactioinCode)}

result_df['TRANSACTIONCODE'] = result_df['TRANSACTIONCODE'].map(my_dict)
#ذخیره خروجی در فایل csv
result_df.to_csv(r'C:\Users\Dps\Desktop\Review\csvFile\finalRecord\output.csv', index=False)

print(result_df)