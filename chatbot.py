from modules import response_table, get_idx
from db import Chatbot, session

def chatbot(query):
    idx = get_idx(query)
    result = session.query(Chatbot).where(Chatbot.order == idx).all()
    return result[0].response

def chatbot_colab(query):
    idx = get_idx(query)
    return response_table['response'][idx]