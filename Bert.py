from torch import nn
from transformers import BertModel,AutoModel
import torch
import numpy as np
from transformers import BertTokenizer, BertConfig,AutoConfig

#const
label_dict_shl = {0: 'Still', 1: 'Walking', 2: 'Run', 3: 'Bike', 4: 'Car', 5: 'Bus', 6: 'Train', 7: 'Subway'}
label_dict_gl = {0:'Walk', 1:'Bike', 2:'Bus', 3:'Drive', 4:'Train'}

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
config = AutoConfig.from_pretrained('bert-base-uncased')# set the configuration for the model

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = df['label'].values
        # This block of code is used to implement BERT-based model taking single feature as input
        # self.texts = [tokenizer(text, 
        #                         padding='max_length', 
        #                         max_length=512, 
        #                         truncation=True,
        #                         return_tensors="pt") 
        #                     for text in df['id']] 

        # This block of code is used to implement BERT-based model taking joint features as input
        self.texts = [tokenizer(row['id'], row['speed'],  
                                padding='max_length', 
                                max_length=512, 
                                truncation=True,
                                return_tensors="pt") 
                            for _, row in df.iterrows()]  # Add closing bracket
        pass
    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer
    

class BertClassifier_npre(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier_npre, self).__init__()
        #self.bert = BertModel(BertConfig().from_pretrained('bert-base-uncased'))
        self.bert =  AutoModel.from_config(config) # to load the BERT with 'bert-base-uncased' configuration but without pre-trained weights
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer
