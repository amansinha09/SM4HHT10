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




def make_corpus(folder):
    '''
    Load data, make flair corpus

    Parameters
    ----------
    folder : TYPE
        DESCRIPTION.

    Returns
    -------
    corpus : TYPE
        DESCRIPTION.

    '''
    columns = {0:'token',1:'ner'}
    corpus = ColumnCorpus(folder, columns) # folder containing train, dev, test. Requires this setting
    dataset_size = [[len(corpus.train), len(corpus.test), len(corpus.dev)]]
    stats = pd.DataFrame(dataset_size, columns=["Train", "Test", "Development"])
    print(stats)
    
    corpus.train[4].to_tagged_string('ner') # print sample NER from training to see all is OK
    
    return corpus

def train_flair(corpus, tag, saving_path):

    tag_type = tag

    tag_dictionary = corpus.make_label_dictionary(label_type= tag)
    
    embedding_types: List[TokenEmbeddings] = [
        FlairEmbeddings('es-forward-fast'),
        FlairEmbeddings('es-backward-fast'),
        FlairEmbeddings('es-clinical-forward')
    ]
    
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
    
    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=True)
    
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)
    
    trainer.train(saving_path,
                  learning_rate = args.LR,
                  mini_batch_size= args.BATCH_SIZE,
                  max_epochs = args.EPOCHS,
                  embeddings_storage_mode='cpu')
 
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-folder', '--folder', required=True, help='input folder')
    
    parser.add_argument('-save_to' ,'--save_to', required=True, help='save preds and model to path')
    
    parser.add_argument('--EPOCHS', default=2, type=int,
                     help='Number of epochs. Default 2')
    
    parser.add_argument('--LR', default=0.1, type=int,
                     help='Learning rate')
    
    parser.add_argument('--BATCH_SIZE', default=2, type=int,
                     help='Batch size')

    
    args = parser.parse_args()
    
    flair_corpus = make_corpus(args.folder)
    
    train_flair(flair_corpus, "ner", args.save_to)