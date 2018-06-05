import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt

data_train = pd.read_csv("./train.csv")

fig = plt.figure()
fig.set(alpha=0.2)
plt.subplot2grid( (2,3),(0,0) )
data_train.Survived.value_counts().plot(kind='bar')
plt.title(u"whether survived (1 means survived)")
plt.ylabel(u"counting")

plt.subplot2grid( (2,3),(0,1) )
data_train.Pclass.value_counts().plot(kind='bar')
plt.title("distribution of class")
plt.ylabel("counting")

plt.subplot2grid( (2,3),(0,2) )
plt.scatter(data_train.Survived,data_train.Age)
plt.title("avg vs. survived")
plt.grid(b=True,which='major',axis='y')
plt.ylabel("counting")

plt.subplot2grid( (2,3),(1,0) ,colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel("age")
plt.title("age vs. class")
plt.legend("first','second','third")

plt.subplot2grid( (2,3),(1,2) )
data_train.Embarked.value_counts().plot(kind='bar')
plt.title("embard")
plt.ylabel("counting")


plt.show()

