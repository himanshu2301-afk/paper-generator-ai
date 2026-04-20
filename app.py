from flask import Flask, render_template, request, jsonify
from groq import Groq
import PyPDF2
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

client = Groq(api_key="")

model = load_model("question_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/lstm')
def lstm_page():
    return render_template('lstm.html')

@app.route('/llm')
def llm_page():
    return render_template('llm.html')

@app.route('/generate-lstm', methods=['POST'])
def generate_lstm():
    data = request.json
    starter = data['starter']
    num_questions = int(data['num_questions'])
    starters = ["what is", "define", "explain", "true or false", "what causes", "where is", "how does", "name the"]
    questions = []
    for i in range(num_questions):
        seed = starters[i % len(starters)] + " " + starter
        result = seed
        for _ in range(8):
            token_list = tokenizer.texts_to_sequences([result])[0]
            token_list = pad_sequences([token_list], maxlen=model.input_shape[1], padding='pre')
            predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            result += " " + output_word
            words = result.split()
            if len(words) >= 3 and words[-1] == words[-2]:
                break
        words = result.split()
        cleaned = []
        for word in words:
            if word not in cleaned:
                cleaned.append(word)
        questions.append(f"Q{i+1}. {' '.join(cleaned).capitalize()}?")
    return jsonify({'result': '\n\n'.join(questions)})

@app.route('/generate-llm', methods=['POST'])
def generate_llm():
    file = request.files['pdf']
    num_mcq = request.form.get('num_mcq', 5)
    num_tf = request.form.get('num_tf', 3)
    num_fill = request.form.get('num_fill', 3)
    num_direct = request.form.get('num_direct', 4)
    difficulty = request.form.get('difficulty', 'Medium')
    text = ""
    reader = PyPDF2.PdfReader(file)
    for i, page in enumerate(reader.pages):
        if i >= 3:
            break
        text += page.extract_text()
    text = text[:3000]
    prompt = f"""
    You are an expert teacher. Based on the following text, generate a question paper with:
    - {num_mcq} Multiple Choice Questions (with 4 options and answer)
    - {num_tf} True or False questions (with answer)
    - {num_fill} Fill in the blank questions (with answer)
    - {num_direct} Direct questions
    Difficulty level: {difficulty}
    Format each section clearly with headings.
    Text: {text}
    """
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return jsonify({'result': response.choices[0].message.content})

if __name__ == '__main__':
    app.run(debug=True)