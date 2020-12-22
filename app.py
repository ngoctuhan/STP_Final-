from flask import (Flask, session, g, json, Blueprint, flash, jsonify, redirect, render_template, request,
                   url_for, send_from_directory)
from flask_socketio import SocketIO, emit
import flask
import requests
import logging
import sys
import argparse
import os
import cgi
import numpy as np
import pandas as pd
import json
import threading
from src.nlu.mapping import Selena
from utils.params import Params
from utils.upload_weight import get_top_intent, __init_weight, update
from werkzeug.utils import secure_filename
from src.ner.ner_extract import NER
from utils.covertPDF2TXT import pdf2txt
from utils.covertPDF2TXT import docx2txt
from nltk import word_tokenize
from src.lm.suggestor import Suggestion
from utils.dataaccount import Login
from utils.datacv import CV
from utils.dataintent import Intent
from utils.save_log import save_log
from utils.datalog import Log

prs = Params()
bot = Selena('models/MLP_model_nlu.pickle')
logging.basicConfig(level=logging.INFO)
ner = NER()
sugg = Suggestion('model_LM', ver=2)
weight = __init_weight(prs)

app = Flask(__name__)
app.config["SECRET_KEY"] = 'secrettt'
socketio = SocketIO(app)

os.environ["GEVENT_SUPPORT"] = 'True'

UPLOAD_FOLDER = 'cv_uploads'
ALLOWED_EXTENSIONS = {'docx', 'pdf'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app.config.from_object(__name__)  # load config from this file , flaskr.py

# Load default config and override config from an environment variable
app.config.update(dict(
    USERNAME='admin',
    PASSWORD='admin',
    SECRET_KEY='development key',
))


@app.route('/chat')
def hello():

    return render_template('chat_off.html')


def messageRecived():
    print('message was received!!!')


@app.route('/', methods=['GET', 'POST'])
def start():
    return render_template('signin.html')


@app.route('/home/<username>')
def home(username):
    data = {}
    return render_template("chat_view.html", data=data)


@app.route('/admin/<username>', methods=['GET', 'POST'])
def admin(username):
    dataCV = CV('dataset/infor_cv/cv.csv')
    data = dataCV.get_cv()
    return render_template("home_admin.html", data=data)


@app.route('/timcv/<username>', methods=['GET', 'POST'])
def timcv(username):
    data = CV('dataset/infor_cv/cv.csv')
    res = data.search_cv(request.form['tencv'], None)
    print(res)
    return jsonify(res)


@app.route('/checklog/<username>', methods=['GET', 'POST'])
def checklog(username):
    log = Log('dataset/log_chat.csv')
    data = log.get_log()
    return render_template("check_log.html", data=data)


'''Intent'''


@app.route('/chinhintent/<username>', methods=['GET', 'POST'])
def chinhintent(username):
    dataIntent = Intent('dataset/nlu_answer')
    data = dataIntent.get_intent()
    global bot
    bot = Selena('models/MLP_model_nlu.pickle')
    return render_template('ans_intent.html', data=data)


@app.route('/timintent/<username>', methods=['GET', 'POST'])
def timintent(username):
    ress = {}
    dataIntent = Intent('dataset/nlu_answer')
    res = dataIntent.get_Name(request.form['tenintent'])
    if len(res) == 0:
        ress['status'] = "no"
    else:
        ress['status'] = "yes"
    ress['data'] = res
    print(ress)
    return jsonify(ress)


@app.route('/luuintent', methods=['POST'])
def luuintent():
    dataIntent = Intent('dataset/nlu_answer')
    ress = {}
    res_ten = request.form['tenintent']
    res_text = request.form['textintent']
    dataIntent.sua_Intent(res_ten, res_text)
    ress['status'] = "yes"
    return jsonify(ress)


'''login'''


@app.route('/login', methods=['GET', 'POST'])
def login():
    checkac = Login('dataset/account.csv')
    error = None
    if request.method == 'POST':
        kt = checkac.check_login(
            request.form['username'], request.form['pass'])
        if kt:
            flash('You were logged in')
            tmp = checkac.get_status(request.form['username'])
            if tmp == 1:
                return redirect(url_for('home', username=request.form['username']))
            else:
                return redirect(url_for('admin', username=request.form['username']))
        else:
            error = 'Invalid username or password'
            return render_template('signin.html', error=error)


@app.route('/dataAccount', methods=['POST'])
def dataAccount():
    checkac = Login('dataset/account.csv')
    if request.method == 'POST':
        data = request.get_json()
        res = {}
        if data['password'] != data['re_password']:
            res['status'] = "Password và repassword không trùng nhau"
        else:
            if checkac.check_user(data['username']) or checkac.check_email(data['email']):

                res['status'] = "Email hoặc tài khoản đã tồn tài trong CSDL"

            else:
                res['status'] = 'Tạo tài khoản: ' + \
                    data['username'] + " thành công"
                checkac.create_acount(
                    data['username'], data['password'], data['email'])
        return jsonify(res)


@app.route('/cv_uploads/<path:filename>')
def custom_static(filename):
    return send_from_directory('./cv_uploads', filename)


def extract_cv(filename):
    global UPLOAD_FOLDER
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    txt = pdf2txt(file_path, "outputF.txt")
    global ner
    resul = ner.predict(txt)
    # save a file

    for key in resul.keys():
        list_item = resul[key]
        new_list = []
        for text in list_item:
            text = text.rstrip("\n")
            # normalize text
            new_list.append(text)

        new_list = np.unique(new_list)
        xxx = ""
        for text in new_list:
            xxx += text + "\n"
        resul[key] = xxx

    # save to the file csv
    columns = ['Name', 'Designation', 'Skills',
               'College Name', 'Companies worked at', 'Email Address']

    # tao ra 1 ban ghi:
    key_have = resul.keys()
    row = []
    for key in columns:
        if key in key_have:
            row.append(resul[key])
        else:
            row.append('')
    columns.append('Filename')
    row.append(filename)
    row = np.array(row).reshape(1, -1)
    path = 'dataset/infor_cv/cv.csv'
    if os.path.exists(path):

        df = pd.read_csv(path, encoding='utf-8').values
        new_df = np.concatenate((df, row), axis=0)
        # print(new_df.shape)
        df = pd.DataFrame(data=new_df, columns=columns)
        df.to_csv(path, index=False)

    else:
        df = pd.DataFrame(data=row, columns=columns)
        df.to_csv(path, index=False)


@app.route('/demo')
def demo():

    return render_template('demo_modal.html')


@app.route('/next_word', methods=['POST'])
def next_word():

    global sugg

    sentence = request.form['word']
    words = word_tokenize(sentence.lower())
    list_word = sugg.find_next_word(words)
    data = {}
    data['num'] = len(list_word)
    for i, word in enumerate(list_word):
        data[str(i)] = word[0]
    return jsonify(data)


@app.route('/submit_cv', methods=['POST'])
def submit_cv():

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        x = threading.Thread(target=extract_cv, args=(file.filename,))
        x.start()
        data = {}

        data['msg'] = 'Chúng tôi đã nhận được CV của bạn'
        data['msg2'] = 'Cv sẽ được duyệt và xs cao là bạn bị trượt haha.'
        return jsonify(data)


@socketio.on('my event')
def handle_my_custom_event(json_data):

    global bot
    global weight
    global prs

    data = {"sender": "trung", "message": "đồng ý"}
    data["sender"] = json_data.get("sender")
    data["message"] = json_data.get("message")

    intent, answer, prob = bot.predict(data["message"])

    if prob < 60:
        json_data["message"] = "Tôi không hiểu câu hỏi của bạn"
        json_data["intent"] = 'non'
    else:
        mores = bot.relation_answer(intent).tolist()
        more_intents = get_top_intent(intent, weight, prs)
        json_data["intent"] = intent
        mores += more_intents
        weight = update(json_data.get("intent_bf"), intent, weight, prs)
        json_data["message"] = answer
        if len(mores) > 0:
            json_data['num'] = str(len(mores))

            for i, more in enumerate(mores):
                key = 'more_' + str(i)
                json_data[key] = more

    json_data["user_name"] = "Bot: "

    x = threading.Thread(target=save_log, args=(
        'dataset/log_chat.csv', data["message"], json_data['intent'], flask.request.remote_addr, ))
    x.start()
    socketio.emit('my response', json_data, callback=messageRecived)


@socketio.on('intent click')
def handle(json_data):

    global bot
    global weight
    global prs

    print(json_data)
    data = {"sender": "trung", "message": "đồng ý"}
    data["sender"] = json_data.get("sender")
    data["message"] = json_data.get("message")

    # print('Goi dc vao day ròi')
    json_data = {}
    # check 2 hướng nếu chọn intent
    if data["message"] in prs.itent:
        print("Là INTENT")
        intent = data["message"]
        json_data["message"] = bot.answer(data["message"])
        mores = bot.relation_answer(intent).tolist()
        more_intents = get_top_intent(intent, weight, prs)
        json_data["intent"] = intent
        mores += more_intents

        if len(mores) > 0:
            json_data['num'] = str(len(mores))

            for i, more in enumerate(mores):
                key = 'more_' + str(i)
                json_data[key] = more
    elif data["message"] in prs.extention:
        # nó là extention
        print("Là EXTENTTION")
        mores = bot.details_extention(data["message"])
        if mores[0] == '0':
            # la cau hoi chi tiết hơn
            json_data["message"] = "Hãy lựa chọn bên dưới"
            if len(mores) > 0:
                json_data['num'] = str(len(mores) - 1)
                json_data["intent"] = 'non'
                for i, more in enumerate(mores):
                    if i > 0:
                        key = 'more_' + str(i - 1)
                        json_data[key] = more
        else:
            # là câu trả lời:
            ans = 'Bao gồm: '
            for i in range(1, len(mores)):
                ans += mores[i] + ' '
            json_data["message"] = ans
            json_data["intent"] = 'non'
            json_data['num'] = str(0)

    else:
        print("Là DETAILS")
        # nếu chọn câu hỏi details
        ans = bot.answer_details(data["message"])
        json_data["intent"] = 'non'
        if len(ans) <= 0:
            json_data["message"] = "Hiện tại tôi không thể giải đáp được vấn đề này"
            json_data['num'] = str(0)
        else:
            json_data["message"] = ans[0]
            json_data['num'] = str(0)

    json_data['sender'] = data["sender"]
    json_data["user_name"] = "Bot :"

    print(json_data)
    socketio.emit('response intent', json_data, callback=messageRecived)


if __name__ == '__main__':

    socketio.run(app, debug=True)

    # extract_cv('test3.pdf')
