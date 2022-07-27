import pickle
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session

db = {
    'user'     : 'user',		# 1)
    'password' : 'password',		# 2)
    'host'     : 'localhost',	# 3)
    'port'     : 3306,			# 4)
    'database' : 'chatbot'		# 5)
}

DB_URL = f"mysql+mysqlconnector://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['database']}"

engine = create_engine(DB_URL, echo=False)

with open('./data/response_data.pickle', "rb") as f:
    response_table = pickle.load(f)

response_table['response'] = response_table['response'].apply(lambda x: x[0])
response_table = pd.DataFrame(response_table['response'])
response_table['order'] = response_table.reset_index().index

response_table.to_sql(name='chatbot', con=engine, if_exists='replace', index=False)
engine.execute('ALTER TABLE chatbot ADD id BIGINT PRIMARY KEY AUTO_INCREMENT')

Base = automap_base()
Base.prepare(engine, reflect=True)
Chatbot = Base.classes.chatbot
session = Session(engine)