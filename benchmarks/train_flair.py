# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 12:21:53 2022

@author: Cristina GH
"""

import pandas as pd
import csv
import os
import sklearn
import numpy as np
import argparse
import flair
from flair.datasets import ColumnCorpus
from torch.utils.data import dataset
from typing import List
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings, FastTextEmbeddings
from numpy.ma.core import default_fill_value




def load_corpus():
    ## get the files and prepared them for flair into a dedicated folder
    os.makedirs(os.path.dirname(args.save_to), exist_ok=True)
    
    for file in [args.train, args.val, args.test]:
        print()
        print("Input files:\n", file)
        print()
        # change to // in linux
        name = args.save_to + file.split('\\')[-1].replace('.conll', '_flair.tsv').replace('training','')
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

def train_flair(corpus, tag, saving_path):

    tag_type = tag

    tag_dictionary = corpus.make_label_dictionary(label_type= tag)
    
    if args.lm == "back_forw_clinical":
        embedding_types: List[TokenEmbeddings] = [
            FlairEmbeddings('es-forward-fast'),
            FlairEmbeddings('es-backward-fast'),
            FlairEmbeddings('es-clinical-forward')
        ]
    if args.lm == "back_forw_glove":
        embedding_types: List[TokenEmbeddings] = [
            WordEmbeddings('glove'),
            FlairEmbeddings('es-backward-fast'),
            FlairEmbeddings('es-clinical-forward')
        ]
    if args.lm == "back_clinical_es_glove":
        embedding_types: List[TokenEmbeddings] = [
            WordEmbeddings('glove'),
            WordEmbeddings('es'),
            FlairEmbeddings('news-backward'),
            FlairEmbeddings('es-clinical-forward')
            ]
    
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
    
    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=True)
    
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)
    
    
    trainer.train(str(saving_path + args.lm),
                  learning_rate = args.LR,
                  mini_batch_size= args.BATCH_SIZE,
                  max_epochs = args.EPOCHS,
                  embeddings_storage_mode='cpu')
 
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-train', '--train', required=True, help='training conll file')
    parser.add_argument('-val', '--val', required=True, help='val conll file')
    parser.add_argument('-test', '--test', required=True, help='test conll file')
    
    parser.add_argument('-save_to' ,'--save_to', required=True, help='save preds and model to path')
    
    parser.add_argument('-lm' ,'--lm', required=True, help='embeddings')
    
    parser.add_argument('--EPOCHS', default=2, type=int,
                     help='Number of epochs. Default 2')
    
    parser.add_argument('--LR', default=0.1, type=int,
                     help='Learning rate')
    
    parser.add_argument('--BATCH_SIZE', default=2, type=int,
                     help='Batch size')

    
    args = parser.parse_args()
    
    corpus, label_dict = load_corpus()
    
    train_flair(corpus, "ner", args.save_to)