import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import EarlyStoppingCallback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tqdm import tqdm
import datetime
from keras.callbacks import ModelCheckpoint, EarlyStopping

path_main = "nlp-getting-started"
train_csv = os.path.join(path_main, "train.csv")
test_csv = os.path.join(path_main, "test.csv")
sample_submission_csv = os.path.join(path_main, "sample_submission.csv")

df_sample_submission = pd.read_csv(sample_submission_csv)
df_train = pd.read_csv("df_train_clean.csv", index_col=None)
df_test = pd.read_csv("df_test_clean.csv", index_col=None)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def encode_texts(texts, max_len):
    return tokenizer(
        texts.tolist(),
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

MAX_LEN = 75
train_encodings = encode_texts(df_train['clean_text'], MAX_LEN)
test_encodings = encode_texts(df_test['clean_text'], MAX_LEN)

X_train = train_encodings['input_ids']
X_train_attention = train_encodings['attention_mask']
y_train = df_train['target'].values

X_test = test_encodings['input_ids']
X_test_attention = test_encodings['attention_mask']

X_train_ids_np = X_train.numpy()
X_train_attention_np = X_train_attention.numpy()

X_train_ids, X_val_ids, X_train_attention, X_val_attention, y_train, y_val = train_test_split(
    X_train_ids_np, X_train_attention_np, y_train, test_size=0.15, random_state=42
)

X_train_ids = tf.convert_to_tensor(X_train_ids)
X_val_ids = tf.convert_to_tensor(X_val_ids)
X_train_attention = tf.convert_to_tensor(X_train_attention)
X_val_attention = tf.convert_to_tensor(X_val_attention)

model = TFBertForSequenceClassification.from_pretrained(
    'bert-base-uncased', 
    num_labels=1,
    use_safetensors=False  
)

optimizer = Adam(learning_rate=2e-5)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

date_i = str(datetime.datetime.now()).replace(" ", "_").replace(":", "-").replace(".", "p")
MODEL_OUTPUT = os.path.join("MODELS", date_i + "_BERT")
os.makedirs(MODEL_OUTPUT, exist_ok=True)

checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(MODEL_OUTPUT, 'model_epoch_{epoch:02d}_val_acc_{val_accuracy:.2f}'),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    save_freq='epoch',
    verbose=1,
    save_format='tf' 
)

early_stopping_callback = EarlyStopping(
    monitor='val_loss', 
    patience=3, 
    mode='min', 
    restore_best_weights=True, 
    verbose=1
)

history = model.fit(
    {'input_ids': X_train_ids, 'attention_mask': X_train_attention},
    y_train,
    validation_data=(
        {'input_ids': X_val_ids, 'attention_mask': X_val_attention}, 
        y_val
    ),
    epochs=10,
    batch_size=8,
    callbacks=[checkpoint_callback, early_stopping_callback],
    verbose=2
)

model.save_pretrained(MODEL_OUTPUT)

history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join(MODEL_OUTPUT, 'training_history.csv'), index=False)

y_pred_logits = model.predict({'input_ids': X_test, 'attention_mask': X_test_attention})['logits']
y_pred = (tf.sigmoid(y_pred_logits).numpy() > 0.5).astype(int).reshape(-1)

sub = pd.DataFrame({'id': df_sample_submission['id'].values.tolist(), 'target': y_pred})
sub.to_csv(os.path.join(MODEL_OUTPUT, date_i + '_BERT_submission.csv'), index=False)
