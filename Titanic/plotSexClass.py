import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2

data_train = pd.read_csv("./train.csv")

# fig = plt.figure()
# fig.set(alpha=0.2)
#
# Survived_1 = data_train.Survived[data_train.Pclass== 1].value_counts()
# Survived_2 = data_train.Survived[data_train.Pclass== 2].value_counts()
# Survived_3 = data_train.Survived[data_train.Pclass== 3].value_counts()
#
# df = pd.DataFrame({'first':Survived_1,'second':Survived_2,'third':Survived_3} )
# df.plot(kind='bar',stacked=True)
# plt.title('survive in different class')
# plt.xlabel("class")
# plt.ylabel("counting")
# plt.show()
#
#
# fig2 = plt2.figure()
# fig2.set(alpha=0.2)
#
# Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
# Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
#
# df2 = pd.DataFrame({'female':Survived_f,'male':Survived_m})
# df2.plot(kind='bar',stacked=True)
# plt2.title("survived vs. sex")
# plt2.xlabel("sex")
# plt2.ylabel("counting")
# plt2.show()


fig = plt.figure()
fig.set(alpha=0.65)
plt.title("survived in different class and sex")

# plt.subplot2grid((1,4),(0,0))
# data_train.Survived[data_train.Sex=='female'][data_train.Pclass!=3].value_counts().plot(kind='bar')
# plt.legend(["female,high carbin"],loc='best')
#
# plt.subplot2grid((1,4),(0,1))
# data_train.Survived[data_train.Sex=='female'][data_train.Pclass==3].value_counts().plot(kind='bar')
# plt.legend(["female,low carbin"],loc='best')
#
# plt.subplot2grid((1,4),(0,2))
# data_train.Survived[data_train.Sex=='male'][data_train.Pclass!=3].value_counts().plot(kind='bar')
# plt.legend(["male,high carbin"],loc='best')
#
# plt.subplot2grid((1,4),(0,3))
# data_train.Survived[data_train.Sex=='male'][data_train.Pclass==3].value_counts().plot(kind='bar')
# plt.legend(["male,low carbin"],loc='best')

ax1 = fig.add_subplot(141)
data_train.Survived[data_train.Sex=='female'][data_train.Pclass!=3].value_counts().plot(kind='bar',color='red')
ax1.set_xticklabels(["not survived","survived"],rotation=0)
ax1.legend(["female,high carbin"],loc='best')

ax2 = fig.add_subplot(142,sharey=ax1)
data_train.Survived[data_train.Sex=='female'][data_train.Pclass==3].value_counts().plot(kind='bar',color='orange')
ax2.set_xticklabels(["not survived","survived"],rotation=0)
ax2.legend(["female,low carbin"],loc='best')

ax3 = fig.add_subplot(143,sharey=ax1)
data_train.Survived[data_train.Sex=='male'][data_train.Pclass!=3].value_counts().plot(kind='bar',color='yellow')
ax3.set_xticklabels(["not survived","survived"],rotation=0)
ax3.legend(["male,high carbin"],loc='best')

ax4 = fig.add_subplot(144,sharey=ax1)
data_train.Survived[data_train.Sex=='male'][data_train.Pclass==3].value_counts().plot(kind='bar',color='green')
ax4.set_xticklabels(["not survived","survived"],rotation=0)
ax4.legend(["male,low carbin"],loc='best')


plt.show()
