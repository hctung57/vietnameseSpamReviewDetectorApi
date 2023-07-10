import re
from vncorenlp import VnCoreNLP
from flask import Flask , request, redirect, url_for
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from constants import *

app = Flask(__name__)
vncorenlp = VnCoreNLP("./vncorenlp/VnCoreNLP-1.1.1.jar",
                      annotators="wseg", max_heap_size='-Xmx500m')
#load model
#tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
#task 1
model_task_1 = AutoModelForSequenceClassification.from_pretrained(model_task_1_dir)
#task 2
model_task_2 = AutoModelForSequenceClassification.from_pretrained(model_task_2_dir)

def filter_stop_words(train_sentences, stop_words):
    new_sent = [word for word in train_sentences.split()
                if word not in stop_words]
    train_sentences = ' '.join(new_sent)

    return train_sentences


def deEmojify(text):
    regrex_pattern = re.compile(pattern="["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'', text)


def preprocess(text, tokenized=True, lowercased=True):
    text = filter_stop_words(text, stopwords)
    text = deEmojify(text)
    text = text.lower() if lowercased else text
    if tokenized:
        pre_text = ""
        sentences = vncorenlp.tokenize(text)
        for sentence in sentences:
            pre_text += " ".join(sentence)
        text = pre_text
    return text

@app.route('/api/task1', methods=['POST'])
def handle_comment_task_1():
    if request.method == "POST":
        review = request.json['review']
        review = preprocess(review, tokenized=True, lowercased=False)
        tokenized_input = tokenizer(str(review), truncation=True, padding=True, max_length=100, return_tensors="pt")
        outputs = model_task_1(**tokenized_input)
        predicted_class_id = outputs.logits.argmax().item()
        return str(predicted_class_id), 200

@app.route('/api/task2', methods=['POST'])
def handle_comment_task_2():
    if request.method == "POST":
        review = request.json['review']
        review = preprocess(review, tokenized=True, lowercased=False)
        tokenized_input = tokenizer(str(review), truncation=True, padding=True, max_length=100, return_tensors="pt")
        outputs = model_task_2(**tokenized_input)
        predicted_class_id = outputs.logits.argmax().item()
        return str(predicted_class_id), 200
    
app.run(host='0.0.0.0',debug=False, port=int(os.environ.get('PORT', 8081)))