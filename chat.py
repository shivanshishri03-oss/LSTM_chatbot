import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
# load the model
model = load_model("models/lstm_chatbot_model.h5")
# load the tokenizer
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
# load the label encoder
with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
# load max_len
with open("models/max_len.pkl", "rb") as f:
    max_len = pickle.load(f)


# tittle 
print("=" * 50)
print("college admission chatbot started")
print("type exit to quit")
print("=" * 50)

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break


    # preprocess the input
    seq = tokenizer.texts_to_sequences([user_input])
    padded_seq = pad_sequences(seq, maxlen=max_len, padding="post")


    # predict the response
    pred = model.predict(padded_seq)
    confidence = np.max(pred)
    if confidence < 0.5:
        print("Chatbot: I'm not sure how to respond to that.")
        continue
    idx = np.argmax(pred)
    response = le.inverse_transform([idx])[0]
    print(f"Chatbot: {response}")

