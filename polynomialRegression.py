# 多项式回归模型
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline



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
y_pred = model.intercept_ + model.coef_ * x
print('predicted response:', y_pred, sep='\n')
print('---------------')
print('intercept:', model.intercept_)
print('slope:', model.coef_)
print('---------------')


#多项式回归模型
Input=[('polynomial',PolynomialFeatures(degree=2)),('modal',LinearRegression())] 
pipe=Pipeline(Input)
pipe.fit(x.reshape(-1,1),y.reshape(-1,1))

poly_pred=pipe.predict(x.reshape(-1,1))
sorted_zip = sorted(zip(x,poly_pred))
x_poly, poly_pred = zip(*sorted_zip)


plt.figure(figsize=(10, 6))
plt.scatter(number_of_vending_machines_List, coffee_sales_List,s=15, color='blue', label='Actual Data')
plt.plot(number_of_vending_machines_List, y_pred, color='red', label='Linear Regression')
plt.plot(x_poly,poly_pred,color='g',label='Polynomial Regression')
plt.xlabel('Number of Vending Machines, X',fontsize=22) #售货机数量
plt.ylabel('Coffee Sales, Y',fontsize=22) #咖啡销售量
plt.title('Linear and Polynomial Regression',fontsize=26)
plt.legend()
plt.show()



