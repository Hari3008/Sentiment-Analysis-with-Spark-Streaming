from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sparknlp.base import DocumentAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, lower, regexp_replace
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from nltk.stem.snowball import SnowballStemmer
from pyspark.sql.types import *
import re 
spark = SparkSession.builder.master("local[2]").getOrCreate()
df = spark.read.csv("test.csv", header=True,inferSchema='True')
text_col = 'Tweet'
Tweet = df.select(text_col).filter(F.col(text_col).isNotNull())
# Clean text
df_clean = df.select('Sentiment', (lower(regexp_replace('Tweet', "[^a-zA-Z\\s]", "")).alias('Tweet')))
# Tokenize text
tokenizer = Tokenizer(inputCol='Tweet', outputCol='Tokenized_words')
df_words_token = tokenizer.transform(df_clean).select('Sentiment', 'Tokenized_words')
# df_words_token.show(truncate=False)
remover = StopWordsRemover(inputCol='Tokenized_words', outputCol='Clean_Tweets')
df_cleaned = remover.transform(df_words_token).select('Sentiment', 'Clean_Tweets')
# Stem text
stemmer = SnowballStemmer(language='english')
stemmer_udf = udf(lambda tokens: [stemmer.stem(token) for token in tokens], ArrayType(StringType()))
df_stemmed = df_cleaned.withColumn("Stemmed_Tweets", stemmer_udf("Clean_Tweets")).select('Sentiment', 'Stemmed_Tweets')
#df_stemmed.show(truncate=False)
#words=['u','ur']
#clean = filter(lambda word: word not in words,df_stemmed('Stemmed_Tweets')) 
#lemmatize text
lemmatizer = Lemmatizer(inputCol='Stemmed_Tweets', outputCol='Lemm_Tweets',) 
df_lemm = lemmatizer.transform(df_stemmed).select('Sentiment','Lemm_Tweets')
