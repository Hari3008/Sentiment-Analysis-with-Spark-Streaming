import pandas as pd 
data=pd.reac_csv("train.csv")
data.head()
pd.set_option('display.max_colwidth',-1)
data= data[['Sentiment','Tweet']]
data.head()
data['Sentiment'].value_counts()
