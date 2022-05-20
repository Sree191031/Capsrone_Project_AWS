import os
import gzip
import itertools
import logging
import torch
from typing import List, Any
import torch.nn.functional as F
import numpy as np
from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AutoModel
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from collections import defaultdict
from typing import Set
from overrides import overrides
from allennlp.training.metrics.metric import Metric
import argparse
import time
from pytorch_lightning import seed_everything
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
import warnings
from tqdm import tqdm
import random as rn
from collections import OrderedDict

os.environ["TOKENIZERS_PARALLELISM"] = "false"

conll_iob = {'B-ORG': 0, 'I-ORG': 1, 'B-MISC': 2, 'I-MISC': 3, 'B-LOC': 4, 
             'I-LOC': 5, 'B-PER': 6, 'I-PER': 7, 'O': 8}
wnut_iob = {'B-CORP': 0, 'I-CORP': 1, 'B-CW': 2, 'I-CW': 3, 'B-GRP': 4, 
            'I-GRP': 5, 'B-LOC': 6, 'I-LOC': 7, 'B-PER': 8, 'I-PER': 9, 
            'B-PROD': 10, 'I-PROD': 11, 'O': 12}
SEED = 42
rn.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def get_ner_reader(data):
    
    # 'fields' contains 4 lists 
    # The first list is the list of words present in the sentence
    # The last list is the list of ner tags of the words.
    
    fin = gzip.open(data, 'rt') if data.endswith('.gz') else open(data, 'rt')
    
    for is_divider, lines in itertools.groupby(fin, _is_divider):
        
        if is_divider:
            continue
        
        fields = [line.strip().split() for line in lines]
        fields = [list(field) for field in zip(*fields)]
        
        yield fields

# Function to assign the new tags 
def _assign_ner_tags(ner_tag, rep_):
    
    ner_tags_rep = []
    token_masks = []

    sub_token_len = len(rep_)
    token_masks.extend([True] * sub_token_len)
    
    if ner_tag[0] == 'B':
        
        in_tag = 'I' + ner_tag[1:]
        ner_tags_rep.append(ner_tag)
        ner_tags_rep.extend([in_tag] * (sub_token_len - 1))
    
    else:
        ner_tags_rep.extend([ner_tag] * sub_token_len)
    
    return ner_tags_rep, token_masks

# Function to extract spans (BI spans) and store in a dictionary
def extract_spans(tags):
    
    cur_tag = None
    cur_start = None
    gold_spans = {}

    def _save_span(_cur_tag, _cur_start, _cur_id, _gold_spans):
        
        if _cur_start is None:
            return _gold_spans
        
        _gold_spans[(_cur_start, _cur_id - 1)] = _cur_tag  # inclusive start & end, accord with conll-coref settings
        
        return _gold_spans

    # iterate over the tags
    for _id, nt in enumerate(tags):
        
        indicator = nt[0]
        
        if indicator == 'B':
            gold_spans = _save_span(cur_tag, cur_start, _id, gold_spans)
            cur_start = _id
            cur_tag = nt[2:]
            pass
        
        elif indicator == 'I':
            # do nothing
            pass
        
        elif indicator == 'O':
            gold_spans = _save_span(cur_tag, cur_start, _id, gold_spans)
            cur_tag = 'O'
            cur_start = _id
            pass
    
    _save_span(cur_tag, cur_start, _id + 1, gold_spans)
    
    return gold_spans


def _is_divider(line: str) -> bool:
    
    empty_line = line.strip() == ''
    
    if empty_line:
        return True

    first_token = line.split()[0]
    if first_token == "-DOCSTART-" or line.startswith('# id'):  # pylint: disable=simplifiable-if-statement
        return True

    return False

class CoNLLReader(Dataset):
    
    def __init__(self, max_instances = -1, max_length = 50, target_vocab = None, 
                 pretrained_dir = '', encoder_model = 'xlm-roberta-large'):
        
        self._max_instances = max_instances
        self._max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_dir + encoder_model)

        self.pad_token = self.tokenizer.special_tokens_map['pad_token']
        self.pad_token_id = self.tokenizer.get_vocab()['pad']
        self.sep_token = self.tokenizer.special_tokens_map['sep_token']

        self.label_to_id = {} if target_vocab is None else target_vocab
        self.instances = []

    def get_target_size(self):
        return len(set(self.label_to_id.values()))

    def get_target_vocab(self):
        return self.label_to_id

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        return self.instances[item]

    def read_data(self, data):
        
        dataset_name = data if isinstance(data, str) else 'dataframe'

        print("Reading file {}".format(dataset_name))
        instance_idx = 0
        
        for fields in tqdm(get_ner_reader(data = data)):
            
            if self._max_instances != -1 and instance_idx > self._max_instances:
                break
            
            sentence_str, tokens_sub_rep, token_masks_rep, coded_ner_, gold_spans_ = self.parse_line_for_ner(fields = fields)
            
            tokens_tensor = torch.tensor(tokens_sub_rep, dtype = torch.long)
            tag_tensor = torch.tensor(coded_ner_, dtype = torch.long).unsqueeze(0)
            token_masks_rep = torch.tensor(token_masks_rep)

            self.instances.append((tokens_tensor, token_masks_rep, gold_spans_, tag_tensor))
            instance_idx += 1
                    
        print("Finished reading {:d} instances from file {}".format(len(self.instances), dataset_name))
    
    def parse_line_for_ner(self, fields):
        
        tokens_, ner_tags = fields[0], fields[-1]

        sentence_str, tokens_sub_rep, ner_tags_rep, token_masks_rep = self.parse_tokens_for_ner(tokens_, ner_tags)
        gold_spans_ = extract_spans(ner_tags_rep)
        coded_ner_ = [self.label_to_id[tag] for tag in ner_tags_rep]

        return sentence_str, tokens_sub_rep, token_masks_rep, coded_ner_, gold_spans_

    def parse_tokens_for_ner(self, tokens_, ner_tags):
        
        sentence_str = ''
        tokens_sub_rep, ner_tags_rep = [self.pad_token_id], ['O']
        
        for idx, token in enumerate(tokens_):
            
            if self._max_length != -1 and len(tokens_sub_rep) > self._max_length:
                break
            
            sentence_str += ' ' + ' '.join(self.tokenizer.tokenize(token.lower()))
            rep_ = self.tokenizer(token.lower())['input_ids']
            rep_ = rep_[1:-1]
            tokens_sub_rep.extend(rep_)

            # if we have a NER here, in the case of B, the first NER tag is the B tag, the rest are I tags.
            ner_tag = ner_tags[idx]
            tags, masks = _assign_ner_tags(ner_tag, rep_)
            ner_tags_rep.extend(tags)

        tokens_sub_rep.append(self.pad_token_id)
        ner_tags_rep.append('O')
        token_masks_rep = [True] * len(tokens_sub_rep)
        
        return sentence_str, tokens_sub_rep, ner_tags_rep, token_masks_rep

def get_tagset(tagging_scheme):
    if 'conll' in tagging_scheme:
        return conll_iob
    return wnut_iob

def get_reader(file_path, max_instances=-1, max_length=50, target_vocab=None, encoder_model='xlm-roberta-large'):
    if file_path is None:
        return None
    reader = CoNLLReader(max_instances=max_instances, max_length=max_length, target_vocab=target_vocab, encoder_model=encoder_model)
    reader.read_data(file_path)

    return reader

class Args():
    
    def __init__(self):
        #PLACE THE APPROPRIATE DATA PATH
        self.train = '/content/CS60075_Course_Project__Multi_CoNER-main/Data/DE-German/de_train.conll'
        self.test = '/content/CS60075_Course_Project__Multi_CoNER-main/Data/DE-German/de_dev.conll'
        self.dev = '/content/CS60075_Course_Project__Multi_CoNER-main/Data/DE-German/de_dev.conll'
        
        self.out_dir = './'
        self.iob_tagging = 'wnut'
        
        self.max_instances = -1
        self.max_length = 50
        # you can add any number of layers here, just make sure you write the neurons in dense layer in a list format
        self.hidden_layer_sizes = [512, 256, 128]
        
        # encoder_model can be bert-base-multilingual-cased or xlm-roberta-base or ai4bharat/indic-bert 
        self.encoder_model = 'xlm-roberta-base'
        self.model = './'
        self.model_name = 'xlm-roberta-base'
        self.stage = 'fit'
        self.prefix = 'test'

        self.batch_size = 8
        self.gpus = 1
        self.device = 'cuda'
        self.epochs = 3
        self.lr = 1e-5
        self.dropout = 0.1
        self.max_grad_norm = 1.0

sg = Args()
class SpanF1(Metric):
    
    def __init__(self, non_entity_labels = ['O']) -> None:
        
        self._num_gold_mentions = 0
        self._num_recalled_mentions = 0
        self._num_predicted_mentions = 0
        self._TP, self._FP, self._GT = defaultdict(int), defaultdict(int), defaultdict(int)
        self.non_entity_labels = set(non_entity_labels)

    @overrides
    def __call__(self, batched_predicted_spans, batched_gold_spans, sentences = None):
        
        non_entity_labels = self.non_entity_labels

        for predicted_spans, gold_spans in zip(batched_predicted_spans, batched_gold_spans):
            gold_spans_set = set([x for x, y in gold_spans.items() if y not in non_entity_labels])
            pred_spans_set = set([x for x, y in predicted_spans.items() if y not in non_entity_labels])

            self._num_gold_mentions += len(gold_spans_set)
            self._num_recalled_mentions += len(gold_spans_set & pred_spans_set)
            self._num_predicted_mentions += len(pred_spans_set)

            for ky, val in gold_spans.items():
                if val not in non_entity_labels:
                    self._GT[val] += 1

            for ky, val in predicted_spans.items():
                if val in non_entity_labels:
                    continue
                if ky in gold_spans and val == gold_spans[ky]:
                    self._TP[val] += 1
                else:
                    self._FP[val] += 1

    @overrides
    def get_metric(self, reset: bool = False) -> float:
        
        all_tags: Set[str] = set()
        all_tags.update(self._TP.keys())
        all_tags.update(self._FP.keys())
        all_tags.update(self._GT.keys())
        all_metrics = {}

        for tag in all_tags:
            precision, recall, f1_measure = self.compute_prf_metrics(true_positives=self._TP[tag],
                                                                     false_negatives=self._GT[tag] - self._TP[tag],
                                                                     false_positives=self._FP[tag])
            all_metrics['P@{}'.format(tag)] = precision
            all_metrics['R@{}'.format(tag)] = recall
            all_metrics['F1@{}'.format(tag)] = f1_measure

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self.compute_prf_metrics(true_positives=sum(self._TP.values()),
                                                                 false_positives=sum(self._FP.values()),
                                                                 false_negatives=sum(self._GT.values())-sum(self._TP.values()))
        all_metrics["micro@P"] = precision
        all_metrics["micro@R"] = recall
        all_metrics["micro@F1"] = f1_measure

        if self._num_gold_mentions == 0:
            entity_recall = 0.0
        else:
            entity_recall = self._num_recalled_mentions / float(self._num_gold_mentions)

        if self._num_predicted_mentions == 0:
            entity_precision = 0.0
        else:
            entity_precision = self._num_recalled_mentions / float(self._num_predicted_mentions)

        all_metrics['MD@R'] = entity_recall
        all_metrics['MD@P'] = entity_precision
        all_metrics['MD@F1'] = 2. * ((entity_precision * entity_recall) / (entity_precision + entity_recall + 1e-13))
        all_metrics['ALLTRUE'] = self._num_gold_mentions
        all_metrics['ALLRECALLED'] = self._num_recalled_mentions
        all_metrics['ALLPRED'] = self._num_predicted_mentions
        if reset:
            self.reset()
        return all_metrics

    @staticmethod
    def compute_prf_metrics(true_positives: int, false_positives: int, false_negatives: int):
        
        precision = float(true_positives) / float(true_positives + false_positives + 1e-13)
        recall = float(true_positives) / float(true_positives + false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measure

    @overrides
    def reset(self):
        
        self._num_gold_mentions = 0
        self._num_recalled_mentions = 0
        self._num_predicted_mentions = 0
        self._TP.clear()
        self._FP.clear()
        self._GT.clear()

class NERModel(nn.Module):
    
    def __init__(self,
                 lr = 1e-5,
                 dropout_rate = 0.1,
                 batch_size = 16,
                 tag_to_id = None,
                 stage = 'fit',
                 hidden_layer_sizes = [512, 256, 128],
                 pad_token_id = 1,
                 device = 'cuda',
                 encoder_model = 'xlm-roberta-large',
                 num_gpus = 1):
        super(NERModel, self).__init__()

        self.id_to_tag = {v: k for k, v in tag_to_id.items()}
        self.tag_to_id = tag_to_id
        self.batch_size = batch_size
        self.device = device

        self.stage = stage
        self.num_gpus = num_gpus
        self.target_size = len(self.id_to_tag)
        
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_layer_list = OrderedDict()
            
        # set the default baseline model here
        self.pad_token_id = pad_token_id

        self.encoder_model = encoder_model
        self.encoder = AutoModel.from_pretrained(encoder_model, return_dict = True)

        self.hidden_layer_list['linear0'] = nn.Linear(in_features = self.encoder.config.hidden_size, 
                                     out_features = self.hidden_layer_sizes[0])
        
        for i in range(len(hidden_layer_sizes) - 1):
            self.hidden_layer_list['linear' + str(i+1)] = nn.Linear(in_features = self.hidden_layer_sizes[i],
                                         out_features = self.hidden_layer_sizes[i+1], device = self.device)
        
        self.hidden_layer_list['linear' + str(len(hidden_layer_sizes))] = nn.Linear(in_features = self.hidden_layer_sizes[-1],
                                        out_features = self.target_size)
        
        self.sequential_layers = nn.Sequential(self.hidden_layer_list)
        
        self.crf_layer = ConditionalRandomField(num_tags = self.target_size, 
                                                constraints = allowed_transitions(constraint_type = "BIO", 
                                                                                  labels = self.id_to_tag))

        self.lr = lr
        self.dropout = nn.Dropout(dropout_rate)

        self.span_f1 = SpanF1()        

    def forward(self, batch):
        
        tokens, tags, token_mask, metadata = batch
        batch_size = tokens.size(0)

        embedded_text_input = self.encoder(input_ids=tokens, attention_mask=token_mask)
        embedded_text_input = embedded_text_input.last_hidden_state
        embedded_text_input = self.dropout(F.leaky_relu(embedded_text_input))

        # project the token representation for classification
#         token_scores = self.feedforward(embedded_text_input)
#         for hidden_layer in self.hidden_layer_list:
#             token_scores = hidden_layer(token_scores)
        
#         token_scores = self.lastfeedforward(token_scores)
        
        token_scores = self.sequential_layers(embedded_text_input)

        # compute the log-likelihood loss and compute the best NER annotation sequence
        output = self._compute_token_tags(token_scores=token_scores, tags=tags, token_mask=token_mask, metadata=metadata, batch_size=batch_size)
        return output
    
    def _compute_token_tags(self, token_scores, tags, token_mask, metadata, batch_size):
        
        # compute the log-likelihood loss and compute the best NER annotation sequence
        loss = -self.crf_layer(token_scores, tags, token_mask) / float(batch_size)
        best_path = self.crf_layer.viterbi_tags(token_scores, token_mask)

        pred_results = []
        for i in range(batch_size):
            tag_seq, _ = best_path[i]
            pred_results.append(extract_spans([self.id_to_tag[x] for x in tag_seq if x in self.id_to_tag]))

        self.span_f1(pred_results, metadata)
        output = {"loss": loss, "results": self.span_f1.get_metric()}
        return output

def dataloading():
    train_data = get_reader(file_path=sg.train, target_vocab=get_tagset(sg.iob_tagging), 
                            encoder_model=sg.encoder_model, max_instances=sg.max_instances,
                            max_length=sg.max_length)
    dev_data = get_reader(file_path=sg.dev, target_vocab=get_tagset(sg.iob_tagging), 
                          encoder_model=sg.encoder_model, max_instances=sg.max_instances, 
                          max_length=sg.max_length)

    return train_data, dev_data

train_data, dev_data = dataloading()

def collate_batch(batch):
        
        batch_ = list(zip(*batch))
        tokens, masks, gold_spans, tags = batch_[0], batch_[1], batch_[2], batch_[3]

        max_len = max([len(token) for token in tokens])
        token_tensor = torch.empty(size = (len(tokens), max_len), 
                                   dtype = torch.long).fill_(1)
        tag_tensor = torch.empty(size = (len(tokens), max_len), 
                                 dtype = torch.long).fill_(model.tag_to_id['O'])
        mask_tensor = torch.zeros(size = (len(tokens), max_len), dtype = torch.bool)

        for i in range(len(tokens)):
            
            tokens_ = tokens[i]
            seq_len = len(tokens_)

            token_tensor[i, :seq_len] = tokens_
            tag_tensor[i, :seq_len] = tags[i]
            mask_tensor[i, :seq_len] = masks[i]

        return token_tensor, tag_tensor, mask_tensor, gold_spans

def train_dataloader():
    loader = DataLoader(train_data, batch_size = sg.batch_size, collate_fn = collate_batch, num_workers = 1)
    return loader

def val_dataloader():
    loader = DataLoader(dev_data, batch_size = sg.batch_size, collate_fn = collate_batch, num_workers = 1)
    return loader

training_dataloader = train_dataloader()
validation_dataloader = val_dataloader()

model = NERModel(tag_to_id = train_data.get_target_vocab(), dropout_rate = sg.dropout, 
                 batch_size = sg.batch_size, hidden_layer_sizes = sg.hidden_layer_sizes, stage = sg.stage, lr = sg.lr,
                         encoder_model = sg.encoder_model, num_gpus = sg.gpus, device = sg.device)

model.to(sg.device)
optimizer = torch.optim.Adam(model.parameters(), lr=sg.lr)

def train_and_evaluate():
    
    print("----------------------- Training ----------------------------")
    print()
    
    # Training loop
    for epoch_i in tqdm(range(sg.epochs)):

        epoch_iterator = tqdm(training_dataloader, desc = "Iteration", position = 0, leave = True)

        # TRAIN loop
        model.train()
        training_loss = 0

        for step, batch in enumerate(epoch_iterator):
            #print(batch)
            batch = (batch[0].to(sg.device), batch[1].to(sg.device), batch[2].to(sg.device), batch[3])
            # forward pass
            output = model.forward(batch)

            # backward pass
            loss = output['loss']
            loss.backward()

            # track train loss
            training_loss += loss.item()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = sg.max_grad_norm)

            # update parameters
            optimizer.step()

        # print train loss per epoch
        training_loss = training_loss / len(training_dataloader)
        print()
        print('Epoch: {} \tTraining Loss: {:.5f}'.format(epoch_i + 1, training_loss))
    
        metric_scores = model.span_f1.get_metric()
        model.span_f1.reset()
        
        print()
        print("Epoch: {} metrics".format(epoch_i+1))
        print()
        for key, value in metric_scores.items():
            print("{}: {:.5f},".format(key, value), end = " ")
        print()
    
    print()
    print("--------------------- Evaluation ---------------------")
    print()
    
    # Loop for evaluation on validation set
    
    epoch_iterator = tqdm(validation_dataloader, desc = "Iteration", position = 0, leave = True)
    
    validation_loss = 0
    for step, batch in enumerate(epoch_iterator):
    
        batch = (batch[0].to(sg.device), batch[1].to(sg.device), batch[2].to(sg.device), batch[3])

        with torch.no_grad():
            output = model.forward(batch)

        loss = output['loss']
        validation_loss += loss.item()

    validation_loss = validation_loss / len(validation_dataloader)
    print()
    print('Validation Loss: {:.5f}'.format(validation_loss))
    print()
    metric_scores = model.span_f1.get_metric()
    model.span_f1.reset()
    print()
    print("Metrics on validation set")
    print()
    for key, value in metric_scores.items():
        print("{}: {:.5f},".format(key, value), end = " ")
    print()
    print()
    torch.save(model, "./" + sg.model_name + "_" + str(sg.batch_size) + "_" + str(sg.lr) + ".pt")
    print("Saved the model")

