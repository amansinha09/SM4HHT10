# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 18:12:29 2022

@author: Cristina GH
"""

import pandas as pd
import csv
import os
import glob
import flair
import re
import sklearn
import numpy as np
import sys
import flair, torch 
import gensim
import argparse
from typing import List
from flair.data import Corpus
from flair.visual.training_curves import Plotter
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.adam import Adam
from numpy.ma.core import default_fill_value
from flair.datasets import ColumnCorpus
from torch.utils.data import dataset
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings, TransformerWordEmbeddings, BertEmbeddings, CharacterEmbeddings, BytePairEmbeddings, TransformerWordEmbeddings

print("********************")
print("CURRENT FLAIR VERSION:")
print(flair.__version__)
print()

def load_corpus():
    ## get the files and prepared them for flair into a dedicated folder
    os.makedirs(os.path.dirname(args.save_to), exist_ok=True)
    
    for file in [args.train, args.val, args.test]:
        print()
        print("Input files:\n", file)
        print()
        # change to // in linux
        name = args.save_to + file.split('/')[-1].replace('.conll', '_flair.tsv').replace('training','')
        df = pd.read_csv(file, sep="\t", quoting=csv.QUOTE_NONE, encoding='utf-8', dtype='object')
        if "Tag" not in df.columns: # for actual test there is no 'tag' col so we make a fake one
            df['Tag'] = "O"
        df = df[['Word', 'Tag']]
        df.to_csv(name, sep="\t", quoting=csv.QUOTE_NONE, encoding='utf-8', index=False, header=None)
    
    
    columns = {0:'text',1:'ner'}
    corpus = ColumnCorpus(args.save_to, columns)
    dataset_size = [[len(corpus.train), len(corpus.test), len(corpus.dev)]]
    print()
    print("-----------------------------------------------------------------")
    print(pd.DataFrame(dataset_size, columns=["Train", "Test", "Development"]))
    print()
    print()
    
    label_dict = corpus.make_label_dictionary(label_type='ner')
    print("Detected labels for NER:\n", label_dict)
    
    return corpus, label_dict


def prepare_tagger(embeddings_model, label_dict):
    # initialize fine-tuneable transformer embeddings WITH document context
    embeddings = TransformerWordEmbeddings(model = embeddings_model,
                                           max_length = 512,
                                           layers = "-1",
                                           subtoken_pooling = "first",
                                           fine_tune = True,
                                           use_context = True,
                                           )
    
    tagger = SequenceTagger(hidden_size=256,
                            embeddings=embeddings,
                            tag_dictionary=label_dict,
                            tag_type='ner',
                            use_crf=True,
                            use_rnn=True,
                            reproject_embeddings=False,
                            )
    return tagger

def train(tagger, corpus):
    # initialize trainer
    trainer = ModelTrainer(tagger, corpus)
    
    # 7. run fine-tuning
    trainer.fine_tune(args.save_to,
                      max_epochs=args.EPOCHS,
                      learning_rate=args.LR,
                      mini_batch_size=args.BATCH_SIZE,
                      mini_batch_chunk_size=1,  # remove this parameter to speed up computation if you have a big GPU
                      )

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-train', '--train', required=True, help='training conll file')
    parser.add_argument('-val', '--val', required=True, help='val conll file')
    parser.add_argument('-test', '--test', required=True, help='test conll file')
    
    
    parser.add_argument('-save_to' ,'--save_to', required=True, help='save preds and model to path')
    
    parser.add_argument('--lm', default=None, 
                        choices = ['llange/xlm-roberta-large-spanish-clinical',
                                   'xlm-roberta-large',
                                   'PlanTL-GOB-ES/roberta-base-biomedical-clinical-es',
                                   'flair/ner-spanish-large',
                                   'mrm8488/bert-spanish-cased-finetuned-ner',
                                   'Davlan/bert-base-multilingual-cased-ner-hrl',
                                   'Babelscape/wikineural-multilingual-ner',
                                   'flair/ner-english-large',
                                   'samrawal/bert-base-uncased_clinical-ner',
                                   'chizhikchi/Spanish_disease_finder'],
                        help='Embedding name. \n **large-spanish-clinical: requires Flair v0.10',
                        required=True)
    
    parser.add_argument('--EPOCHS', default=1, type=int,
                     help='Number of epochs. Default 1')
    
    parser.add_argument('--LR', default=5.0e-6, type=int,
                     help='Learning rate')
    
    parser.add_argument('--BATCH_SIZE', default=4, type=int,
                     help='Batch size')

    
    args = parser.parse_args()
    print("arguments given:\n",args,'\n')
    print("*"*40)
    corpus, labels = load_corpus()
    
    tagger = prepare_tagger(args.lm, labels)
    
    train(tagger, corpus)
    
    print(f"Files saved to {args.save_to}")
