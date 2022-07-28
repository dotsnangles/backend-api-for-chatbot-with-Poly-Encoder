from flask import Flask, request
import json
from chatbot import chatbot

app = Flask(__name__)

data = {
    'response': ''
}

@app.route('/', methods=['POST'])
def chatbot_():
    context = request.json['context']
    data['response'] = chatbot(context)
    return json.dumps(data, ensure_ascii=False)

if __name__ == '__main__':
    app.run(debug=True)