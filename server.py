import re
from vncorenlp import VnCoreNLP
import json
import psycopg2
from flask import Flask , request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from constants import *

#ket noi voi co so du lieu 
DATABASE_URL = 'postgres://spam_user:Yh41T27MKvNg5x8NaXSgHbaqRfqGg4Ef@dpg-ciooartgkuvh5ghr6tvg-a.singapore-postgres.render.com/spam'
conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

# Khoi tao flask server
app = Flask(__name__)
CORS(app)


# khoi tao vncorenlp
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
        data = review = request.json['review']
        productID = request.json['product_id']
        review = preprocess(review, tokenized=True, lowercased=False)
        tokenized_input = tokenizer(str(review), truncation=True, padding=True, max_length=100, return_tensors="pt")
        outputs = model_task_2(**tokenized_input)
        predicted_class_id = outputs.logits.argmax().item()
        if predicted_class_id in [0,1,2,3]:
            cur.execute("""INSERT INTO reviews (product_id, review, task2)
                    VALUES (%s, %s, %s);""", (productID, data, predicted_class_id))
            # Xác nhận việc thêm dữ liệu
            conn.commit()
        return str(predicted_class_id), 200


@app.route('/api/get/<product_id>', methods=['GET'])
def get_all_data(product_id):
    cur.execute("""SELECT *
                    FROM reviews
                    WHERE product_id = %s
                    ORDER BY task2 ASC, time DESC
                """, (product_id,))

    # Lấy kết quả trả về từ câu lệnh SELECT
    comments = cur.fetchall()
    json_data = []
    for item in comments:
        if item != None:
            item_dict = {
                'product_id': item[1],
                'review': item[2],
                'task2': item[3]
            }
            json_data.append(json.loads(json.dumps(item_dict)))
    return jsonify(json_data)
  
    
app.run(host='0.0.0.0',debug=False, port=int(os.environ.get('PORT', 8081)))