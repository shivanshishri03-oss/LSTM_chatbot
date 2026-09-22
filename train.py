## libraries 
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense , Dropout

# data load 
df = pd.read_csv("data/chat_data.csv")
df.head()
# data preprocessing
X = df["question"].values
y = df["answer"].values

# label encoding
le = LabelEncoder()
y = le.fit_transform(y)

# save the label encoder
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# tokenize the questions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)

# save the tokenizer
with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# convert questions to sequences
X_seq = tokenizer.texts_to_sequences(X)

# pad sequences
max_len = max(len(seq) for seq in X_seq)
X_pad = pad_sequences(X_seq, maxlen=max_len, padding="post")

# model 
model = Sequential()


#embedding layer
vocab_size = len(tokenizer.word_index) + 1
model.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=max_len))

# lstm
model.add(LSTM(128, return_sequences=False))

# dropout
model.add(Dropout(0.2))


#hidden layer 
model.add(Dense(64, activation='relu'))

# output layer

model.add(Dense(len(np.unique(y)), activation='softmax'))

# compile 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model fit
model.fit(X_pad, y, epochs=200, batch_size=4)

# save the model
model.save("models/lstm_chatbot_model.h5")

# max_len 
pickle.dump(max_len, open("models/max_len.pkl", "wb"))
