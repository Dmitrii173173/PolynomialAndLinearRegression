import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
# 线性回归模型
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

tab = pd.read_csv("D:\HomeWorkCar90\Coffe\coffe2.csv")
tab.head()

print(tab)
coffe_shop_List = tab['coffe_shop'].tolist()
number_of_vending_machines_List = tab['number_of_vending_machines'].tolist()
coffee_sales_List = tab['coffee_sales'].tolist()
print(coffe_shop_List) 
print(number_of_vending_machines_List) #售货机数量
print(coffee_sales_List) #咖啡销售量
x = np.array(number_of_vending_machines_List).reshape((-1, 1))
y = np.array(coffee_sales_List)
print(x)
print(y)
model = LinearRegression() #性回归模型
model.fit(x, y)
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
# y_pred = model.predict(x)
# print('predicted response:', y_pred, sep='\n')
y_pred = model.intercept_ + model.coef_ * x
print('predicted response:', y_pred, sep='\n')




plt.figure(figsize=(10, 6))
plt.scatter(number_of_vending_machines_List, coffee_sales_List, color='blue', label='Actual Data')
plt.plot(number_of_vending_machines_List, y_pred, color='red', label='Linear Regression')
plt.xlabel('Number of Vending Machines',fontsize=22)
plt.ylabel('Coffee Sales',fontsize=22)
plt.title('Linear Regression',fontsize=26)
plt.legend()
plt.show()

