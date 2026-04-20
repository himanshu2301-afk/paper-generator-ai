import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

model = load_model("question_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def generate_question(seed_text, num_words=8):
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=model.input_shape[1], padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        seed_text += " " + output_word

        if "?" in output_word:
            break
        words = seed_text.split()
        if len(words) >= 3 and words[-1] == words[-2]:
            break

    words = seed_text.split()
    cleaned = []
    for word in words:
        if word not in cleaned:
            cleaned.append(word)
    return " ".join(cleaned).capitalize()

print(generate_question("what is"))
print(generate_question("define the"))
print(generate_question("explain the"))
print(generate_question("true or false"))