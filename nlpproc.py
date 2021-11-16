import pandas as pd 
data=pd.reac_csv("train.csv")
data.head()
pd.set_option('display.max_colwidth',-1)
data= data[['Sentiment','Tweet']]
data.head()
data['Sentiment'].value_counts()
#removing punctuation from tweet data
import string
string.punctuation
def rem_punct(text):
    cleanedtext="".join([i for i in text if i not in string.punctuation])
    return cleanedtext
data['clean_data']=data['Tweet'].apply(lambda x:rem_punct(x))
data.head()
#consistency in data by lowering case
data['lower_case']=data['clean_data'].apply(lambda x: x.lower())
data.head()