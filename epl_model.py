import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Letting pandas read our CSV
data = pandas.read_csv('epl_payroll_clean.csv')

# Assigning each column from the table to a variable
x = DataFrame(data, columns=['squad_value'])
y = DataFrame(data, columns=['points'])


# Creating the chart to compare X and Y values
plt.scatter(x, y, alpha=0.3)
plt.title('Squad value vs Points accrued - EPL 2020/2021 ')
plt.xlabel('Squad value $')
plt.ylabel('Points accrued')

# Remove comment to show graph
# plt.show()

regression = LinearRegression()
regression.fit(x,y)

# Creating the chart with our linear regression line plotted
plt.plot(x, regression.predict(x), color='green', linewidth=3)
plt.scatter(x, y, alpha=0.3)
plt.title('Squad value vs Points accrued - EPL 2020/2021 ')
plt.xlabel('Squad value $')
plt.ylabel('Points accrued')

# Remove comment to show graph
# plt.show()

# How accurate is the model?

# R squared is equal to 0.5241, so ~ 52%
print(regression.score(x, y))



