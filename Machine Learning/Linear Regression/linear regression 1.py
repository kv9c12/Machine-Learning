"""
Linear Regression model using imported data from quandl
"""

import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, train_test_split
from matplotlib import style
import pickle 

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close','HL_PCT','PCT_Change','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)
forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)
df['label'] = df[(forecast_col)].shift(-forecast_out)

x = np.array(df.drop(['label'],1))
x = preprocessing.scale(x)
x=x[:-forecast_out]
x_lately=x[-forecast_out:]
df.dropna(inplace=True)
y = np.array(df['label'])
y=np.array(df['label'])

# =============================================================================
# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
# clf = LinearRegression(n_jobs=any)
# clf.fit(x_train,y_train)
# with open('linearregression.pickle','wb') as f:
#     pickle.dump(clf,f)
# =============================================================================


# =============================================================================
#     Single line comment
# 
#     Ctrl + 1
# 
#     Multi-line comment select the lines to be commented
# 
#     Ctrl + 4
# 
#     Unblock Multi-line comment
# 
#     Ctrl + 5
# 
# 
# =============================================================================

    
pickle_in = open('linearregression.pickle','rb')
clf=pickle.load(pickle_in)

accuracy=clf.score(x_test,y_test)

forecast_set = clf.predict(x_lately)

print(forecast_set,accuracy,forecast_out)

df['Forecast']=np.nan

last_date=df.iloc[-1].name
last_unix=last_date.timestamp()
one_day=86400
next_unix=last_unix+one_day

for i in forecast_set:
    next_date=datetime.datetime.fromtimestamp(next_unix)
    next_unix+=one_day
    df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)]+[i]
    
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=2)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()