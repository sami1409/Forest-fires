import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

df=pd.read_csv('forestfires.csv')
# print(df.head())
# print(df.shape)
# print(df.isna().sum())     checking null values in each column
# print(df.info())           checking null values in each column and datatype
# print(df.describe().T)
# print(df.describe(include='O'))


num_cols=df.select_dtypes(include='number').columns
fig, axs=plt.subplots(nrows=4,ncols=2,figsize=(15,20))
axs=np.ravel(axs)
for i,col in enumerate(num_cols[2:]):
    plt.sca(axs[i])
    sns.histplot(data=df, x=col, kde=True, line_kws={'linewidth':2, 'linestyle':'--'}, color='orange')
plt.tight_layout()
plt.show()


mask= (df.FFMC<80)|(df.ISI>25)
df=df.loc[~mask]
# print(df.shape)


# num_cols = df.select_dtypes(include='number').columns
# fig, axs =  plt.subplots(nrows=5, ncols=2, figsize=(15,20))
# axs = np.ravel(axs)
# for i, col in enumerate(num_cols[2:]):
#     plt.sca(axs[i])
#     sns.histplot(data=df, x=col, kde=True, line_kws={'linewidth':2, 'linestyle':'--'}, color='orange')  
# plt.tight_layout()
# plt.show()


# print(df.month.value_counts())
# print(df.day.value_counts())


month_map={'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
day_map = {'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5, 'sat':6, 'sun':7}
df.month=df.month.map(month_map)
df.day=df.day.map(day_map)
# print(df.info())


X, y = df.drop('area', axis=1).values, df['area'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


linear_reg = LinearRegression()
linear_reg.fit(X_train_scaled,y_train)
y_hat=linear_reg.predict(X_train_scaled)
mse=mean_squared_error(y_train,y_hat)
rmse=np.sqrt(mse)
print(f"RMSE:{rmse:.3f}\n")


print("Actual value         predicted value")
for i in range(10):
    print(f"{y_train[i]}    ,  {y_hat[i]}")


y_hat_test=linear_reg.predict(X_test_scaled)
print("\n\nActual value         predicted value")
for i in range(10):
    print(f"{y_test[i]}    ,  {y_hat_test[i]}")




