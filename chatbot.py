from modules import get_idx
from db import Chatbot, session

def chatbot(query):
    idx = get_idx(query)
    result = session.query(Chatbot).where(Chatbot.order == idx).all()
    return result[0].response