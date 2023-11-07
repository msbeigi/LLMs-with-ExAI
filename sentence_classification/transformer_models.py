# !pip install transformers
import transformers
import torch
from transformers import AdamW
from transformers import BertTokenizer,\
BertForSequenceClassification

def predict_with_tranfomers():
	pretrained_model_name = 'bert-base-uncased'

	tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
	model = BertForSequenceClassification.from_pretrained(pretrained_model_name)
