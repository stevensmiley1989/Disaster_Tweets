import re
from nltk.tokenize import word_tokenize
import gensim
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D
from keras.layers import Bidirectional
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
import pandas as pd
import string
import os
import altair as alt
import warnings
import operator
import numpy as np
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.models import load_model
warnings.filterwarnings("ignore")
from nltk.corpus import stopwords

stop=set(stopwords.words('english'))
path_main = "nlp-getting-started"
files = os.listdir(path_main)
train_csv = os.path.join(path_main,"train.csv")
test_csv = os.path.join(path_main,"test.csv")
sample_submission_csv = os.path.join(path_main,"sample_submission.csv")

df_sample_submission = pd.read_csv(sample_submission_csv)
def create_corpus(df):
    corpus=[]
    for tweet in tqdm(df['clean_text']):
        words=[word.lower() for word in word_tokenize(tweet) if((word.isalpha()==1) & (word not in stop))]
        corpus.append(words)
    return corpus
        
glove_file = 'glove.twitter.27B.200d.txt'
NUMT = 200

def load_glove_model(glove_file):
    glove_model = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float32)
            glove_model[word] = embedding
    return glove_model

# # Load the GloVe embeddings
glove_embeddings = load_glove_model(glove_file)        
df_train = pd.read_csv("df_train_clean.csv",index_col=None)
df_test = pd.read_csv("df_test_clean.csv",index_col=None)

df_combined = pd.concat([df_train, df_test], axis=0, ignore_index=True)
corpus = create_corpus(df_combined)

MAX_LEN=75
tokenizer_obj=Tokenizer()
tokenizer_obj.fit_on_texts(corpus)
sequences=tokenizer_obj.texts_to_sequences(corpus)

tweet_pad=pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')

word_index=tokenizer_obj.word_index
print('Number of unique words:',len(word_index))

num_words=len(word_index)+1
embedding_matrix=np.zeros((num_words,NUMT))

for word,i in tqdm(word_index.items()):
    if i > num_words:
        continue
    
    emb_vec=glove_embeddings.get(word)
    if emb_vec is not None:
        embedding_matrix[i]=emb_vec
            
model=Sequential()
embedding=Embedding(num_words,NUMT,embeddings_initializer=Constant(embedding_matrix),
                   input_length=MAX_LEN,trainable=False)
model.add(embedding)
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(LSTM(150, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(1, activation='sigmoid'))
optimzer=Adam(learning_rate=1e-5)
model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])

train=tweet_pad[:df_train.shape[0]]
test=tweet_pad[df_train.shape[0]:]

X_train,X_test,y_train,y_test=train_test_split(train,df_train['target'],test_size=0.15)
print('Shape of train',X_train.shape)
print("Shape of Validation ",X_test.shape)
import datetime
date_i = str(datetime.datetime.now()).replace(" ","_").replace(":","-").replace(".","p")
MODEL_OUTPUT = os.path.join("MODELS",date_i+"_"+glove_file.split(".txt")[0].replace(".","p"))
if os.path.exists(MODEL_OUTPUT)==False:
    os.makedirs(MODEL_OUTPUT)

early_stopping_callback = EarlyStopping(
    monitor='val_loss',        
    patience=3,               
    mode='min',                
    restore_best_weights=True, 
    verbose=1                  
)

checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(MODEL_OUTPUT,'model_epoch_{epoch:02d}_val_acc_{val_accuracy:.2f}.h5'),  
    monitor='val_accuracy',       
    save_best_only=True,        
    mode='max',                   
    save_freq='epoch',             
    verbose=1                     
)


history = model.fit(
    X_train, y_train,
    batch_size=4,
    epochs=40,
    validation_data=(X_test, y_test),
    verbose=2,
    callbacks=[checkpoint_callback,early_stopping_callback]  
)

history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join(MODEL_OUTPUT,'training_history.csv'), index=False)
model.save(os.path.join(MODEL_OUTPUT,'model.h5'))
model = load_model(os.path.join(MODEL_OUTPUT,'model.h5'))

y_pre=model.predict(test)
y_pre=np.round(y_pre).astype(int).reshape(3263)
sub=pd.DataFrame({'id':df_sample_submission['id'].values.tolist(),'target':y_pre})
sub.to_csv(os.path.join(MODEL_OUTPUT,date_i+"_"+glove_file.split(".txt")[0].replace(".","p")+'_submission.csv'),index=False)
