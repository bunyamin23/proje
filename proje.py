

import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding
from nltk.corpus import stopwords
import string
import nltk
from gensim import corpora, models, similarities 
import re 
import demoji

from tensorflow.keras.optimizers import Adam



#cols=["Tweet Id","Text","Name","Screen Name","UTC","Created At","Favorites","Reet"]
df = pd.read_excel("C:/Users/BÜNYAMİN/Desktop/twitter.xlsx")
df = df["Text"].to_frame()


def word_tokenize(sentence):    
    acronym_each_dot = r"(?:[a-zğçşöüı]\.){2,}"
    acronym_end_dot = r"\b[a-zğçşöüı]{2,3}\."
    suffixes = r"[a-zğçşöüı]{3,}' ?[a-zğçşöüı]{0,3}"
    numbers = r"\d+[.,:\d]+"
    any_word = r"[a-zğçşöüı]+"
    punctuations = r"[a-zğçşöüı]*[.,!?;:]"
    word_regex = "|".join([acronym_each_dot,acronym_end_dot,suffixes,numbers,any_word,punctuations])
    sentence = re.compile("%s"%word_regex, re.I).findall(sentence)
    return sentence

def initial_clean(text):
     text = text.lower()
     text = re.sub("[^a-zğüışçö ]", "", text)
     text = nltk.word_tokenize(text)
     return text


with open('C:/Users/BÜNYAMİN/Desktop/stopwords-tr.txt', 'r') as f:
    myList = [line.strip() for line in f]

def remove_stop_words(df):
    stop_words = myList 
    return [word for word in df if word not in stop_words]

"""
import codecs
stop_words = set(stopwords.words('turkish')) 
file1 = codecs.open('soccer.csv','r','utf-8') 
line = file1.read() 
words = line.split()
for r in words: 
    if not r in stop_words: 
        appendFile = open('stopwords_soccer.csv','a', encoding='utf-8') 
        appendFile.write(" "+r)
        appendFile.close()        
        
        appendFile.write(r)
        appendFile.write("\n")
        appendFile.close()

import nltk
nltk.download('stopwords')
turkish_stopwords = stopwords.words("turkish")
"""

def deEmoji(df):

    emoji_pattern = re.compile("["
          u"\U0001F600-\U0001F64F" 
          u"\U0001F300-\U0001F5FF"  
          u"\U0001F680-\U0001F6FF" 
          u"\U0001F1E0-\U0001F1FF"  
                            "]+", flags=re.UNICODE)
    return str(emoji_pattern.sub('', df) )


def apply_all(df):
    return remove_stop_words(initial_clean(deEmoji(df)))
        
import time

t1 = time.time()
df['tokenized_texts'] = df["Text"].apply(apply_all) 
#kirli veri seti normalize edilmiş veri seti
t2 = time.time()
print("prerocess ve tokenize için geçen süre", len(df), "texts:", (t2-t1)/60, "min") 
#Preprocess ve tokenize uyguladığımız verilerimize bir bakalım.



#LDA
tokenized = df['tokenized_texts']
dictionary = corpora.Dictionary(tokenized)
dictionary.filter_extremes(no_below=1, no_above=0.8)
corpus = [dictionary.doc2bow(tokens) for tokens in tokenized]
print(corpus[:1])        

import gensim
import pyLDAvis
import pyLDAvis.gensim as gensimvis

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 8, id2word=dictionary, passes=15)
ldamodel.save('mOdel.gensim')#save model
topics = ldamodel.print_topics(num_words=30)
        
for topic in topics:
   print(topic)
   
get_document_topics = ldamodel.get_document_topics(corpus[0])
print(get_document_topics)

lda_viz = gensim.models.ldamodel.LdaModel.load('mOdel.gensim')#load lda model
lda_display = pyLDAvis.gensim.prepare(lda_viz, corpus, dictionary, sort_topics=True)
pyLDAvis.display(lda_display)   

def dominant_topic(ldamodel,corpus,content):
     sent_topics_df = pd.DataFrame() 
     for i, row in enumerate(ldamodel[corpus]):
         row = sorted(row, key=lambda x: (x[1]), reverse=True)
         for j, (topic_num, prop_topic) in enumerate(row):
             if j == 0: 
                 wp = ldamodel.show_topic(topic_num,topn=30)
                 topic_keywords = ", ".join([word for word, prop in wp])
                 sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
             else:
                 break
     sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
     contents = pd.Series(content)#noisy data
     sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
     return(sent_topics_df)

df_dominant_topic = dominant_topic(ldamodel=ldamodel, corpus=corpus, content=df['Text'])

df_dominant_topic.head(10)

df_dominant_topic.to_csv('outputFile.csv')
"""
cleaning = pd.read_csv("C:/Users/BÜNYAMİN/Desktop/outputFile.csv")
cleaning=cleaning.drop(["Text","Topic_Keywords"],axis=1)

df_data = pd.read_excel("C:/Users/BÜNYAMİN/Desktop/dolar_verileri.xlsx")
df_data = df_data["dolar_verileri"].to_frame()       


#pd.unified_dolar(cleaning,df_data)
"""
from sklearn.preprocessing import LabelEncoder

df_dolar=pd.read_excel("C:/Users/BÜNYAMİN/Desktop/dolar_verileri.xlsx")        

labels_encoder= LabelEncoder().fit(df_dolar.dolar_verileri)
labels=labels_encoder.transform(df_dolar.dolar_verileri)
classes=list(labels_encoder.classes_)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df_dolar=sc.fit_transform(df_dolar) 
y=labels

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_dolar, y,test_size=0.2)

x_train, x_test, y_train, y_test =np.array(x_train),np.array(x_test),np.array(y_train),np.array(y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
"""
model =Sequential()
model.add(LSTM(16, input_dim=4, activation="relu"))
model.add((LSTM(12, activation="relu"))
model.add(Dense(4, activation="softmax"))
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train,y_train, validation_data=(x_test, y_test), epochs=150)
"""

model = Sequential()
model.add(Embedding(input_dim=4,
                    output_dim=50,
                    input_length=len(x_train),
                    name='embedding_layer'))


## 16 nöronlu LSTM (16 outputlu , return_sequences=True demek output'un tamamını ver demek)
model.add(LSTM(units=16,  return_sequences=True))
## 8 nöronlu LSTM (8 outputlu , return_sequences=True demek output'un tamamını ver demek)
model.add(LSTM(units=8, return_sequences=True))
## 4 nöronlu LSTM (4 outputlu , return_sequences=False yani default değer, tek bir output verecek)
model.add(LSTM(units=4))
## output layer'ı , görsel olarak gösterilirken dense layer kullanılır.  Tek bir nörondan oluştuğu için 1 yazılır.
model.add(Dense(1, activation='sigmoid'))

#modeli derlemek, loss fonksiyonu binary_crossentropy -> sadece 2 sınıf ama daha fazla sınıflar için categorical_crossentropy kullanılır.
#metrics -> modelin başarısını görmek için.
optimizer = Adam(lr=1e-3)


model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(x_train, y_train,validation_split=0.25, epochs=5, batch_size=256)


import matplotlib.pyplot as plt

plt.plot(model.history.history["accuracy"])
plt.plot(model.history.history["val_accuracy"])
plt.title("model başarımı")
plt.ylabel("başarım")
plt.xlabel("epok sayısı")
plt.legend(["eğitim", "test"], loc="upper left") 
plt.show()      




        