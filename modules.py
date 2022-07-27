import pickle, os, gdown, shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertConfig, BertModel, BertTokenizer, AutoModel
from encoder.encoder import PolyEncoder
from encoder.transform import SelectionJoinTransform, SelectionSequentialTransform

if os.path.exists('./data'):
    shutil.rmtree('./data')

if os.path.exists('./model'):
    shutil.rmtree('./model')

gdown.download_folder(id='1Ipr-aNF5ELMY0HTXAmeV26LlgktKUfmG', quiet=True, use_cookies=False)
gdown.download_folder(id='1RH7laK4WlucCw68ZeExFvyg7vs-kB_x3', quiet=True, use_cookies=False)
os.rename('./감성대화챗봇데이터/', './data')
os.rename('./chatbot_output/', './model')

# try:
#     response_table = pd.read_pickle('./data/response_data.pickle')
# except:
    # with open('./data/response_data.pickle', "rb") as f:
    #     response_table = pickle.load(f)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

PATH = 'model/poly_16_pytorch_model.bin'

bert_name = 'klue/bert-base'
bert_config = BertConfig.from_pretrained(bert_name)

tokenizer = BertTokenizer.from_pretrained(bert_name)
tokenizer.add_tokens(['\n'], special_tokens=True)

context_transform = SelectionJoinTransform(tokenizer=tokenizer, max_len=256)
response_transform = SelectionSequentialTransform(tokenizer=tokenizer, max_len=128)

bert = BertModel.from_pretrained(bert_name, config=bert_config)

model = PolyEncoder(bert_config, bert=bert, poly_m=16)
model.resize_token_embeddings(len(tokenizer))
try:
    model.load_state_dict(torch.load(PATH))
except RuntimeError as e:
    # print(e)
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

model.to(device)

# data examples
# context = ['This framework generates embeddings for each input sentence', 
#             'Sentences are passed as a list of string.', 
#             'The quick brown fox jumps over the lazy dog.']

# candidates = ['This framework generates embeddings for each input sentence', 
#             'Sentences are passed as a list of string.', 
#             'The quick brown fox jumps over the lazy dog.']

def context_input(context):
    context_input_ids, context_input_masks = context_transform(context)
    contexts_token_ids_list_batch, contexts_input_masks_list_batch = [context_input_ids], [context_input_masks]
    long_tensors = [contexts_token_ids_list_batch, contexts_input_masks_list_batch]
    contexts_token_ids_list_batch, contexts_input_masks_list_batch = (torch.tensor(t, dtype=torch.long, device=device) for t in long_tensors)
    return contexts_token_ids_list_batch, contexts_input_masks_list_batch

def response_input(candidates):
    responses_token_ids_list, responses_input_masks_list = response_transform(candidates)
    responses_token_ids_list_batch, responses_input_masks_list_batch = [responses_token_ids_list], [responses_input_masks_list]
    long_tensors = [responses_token_ids_list_batch, responses_input_masks_list_batch]
    responses_token_ids_list_batch, responses_input_masks_list_batch = (torch.tensor(t, dtype=torch.long, device=device) for t in long_tensors)
    return responses_token_ids_list_batch, responses_input_masks_list_batch

def embs_gen(contexts_token_ids_list_batch, contexts_input_masks_list_batch):
    with torch.no_grad():
        model.eval()
        ctx_out = model.bert(contexts_token_ids_list_batch, contexts_input_masks_list_batch)[0]  # [bs, length, dim]
        poly_code_ids = torch.arange(model.poly_m, dtype=torch.long).to(contexts_token_ids_list_batch.device)
        poly_code_ids = poly_code_ids.unsqueeze(0).expand(1, model.poly_m)
        poly_codes = model.poly_code_embeddings(poly_code_ids) # [bs, poly_m, dim]
        embs = model.dot_attention(poly_codes, ctx_out, ctx_out) # [bs, poly_m, dim]
        return embs

def cand_emb_gen(responses_token_ids_list_batch, responses_input_masks_list_batch):
    with torch.no_grad():
        model.eval()
        batch_size, res_cnt, seq_length = responses_token_ids_list_batch.shape # res_cnt is 1 during training
        responses_token_ids_list_batch = responses_token_ids_list_batch.view(-1, seq_length)
        responses_input_masks_list_batch = responses_input_masks_list_batch.view(-1, seq_length)
        cand_emb = model.bert(responses_token_ids_list_batch, responses_input_masks_list_batch)[0][:,0,:] # [bs, dim]
        cand_emb = cand_emb.view(batch_size, res_cnt, -1) # [bs, res_cnt, dim]
        return cand_emb

def score(embs, cand_emb):
    with torch.no_grad():
        model.eval()
        ctx_emb = model.dot_attention(cand_emb, embs, embs) # [bs, res_cnt, dim]
        dot_product = (ctx_emb*cand_emb).sum(-1)
        return dot_product

def get_cand_embs():
    if os.path.exists('./data/cand_embs.pickle'):
        with open('./data/cand_embs.pickle', 'rb') as f:
            cand_embs = pickle.load(f)
        return cand_embs.to(device)
    else:
        with open('./data/train_data_source.pickle', 'rb') as f:
            train = pickle.load(f)

        data = {
            'context' : [],
            'response': []
        }

        for sample in train:
            data['context'].append(sample['context'])
            data['response'].append([sample['responses'][0]])


        df = pd.DataFrame(data)

        ## generate cand_embs & create tensor table on device
        response_input_srs = df['response'].apply(response_input)
        response_input_lst = response_input_srs.to_list()

        cand_embs_lst = []
        for sample in response_input_lst:
            cand_embs_lst.append(cand_emb_gen(*sample).to('cpu'))

        cand_embs = cand_embs_lst[0]
        for idx in range(1, len(cand_embs_lst)):
            y = cand_embs_lst[idx]
            cand_embs = torch.cat((cand_embs, y), 1)
        return cand_embs.to(device)

cand_embs = get_cand_embs()

with open('./data/response_data.pickle', "rb") as f:
    response_table = pickle.load(f)

def chatbot(query):
    query = [query]
    poly_embs = embs_gen(*context_input(query))
    score_lst = score(poly_embs, cand_embs)
    idx = score_lst.argmax(1)
    idx = int(idx[0])
    return response_table['response'][idx][0]