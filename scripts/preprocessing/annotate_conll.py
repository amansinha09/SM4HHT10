# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 14:49:11 2022

@author: Cristina GH
"""


import nltk
import stanza
import pandas as pd
import csv
import argparse
from ast import literal_eval
nltk.download('stopwords')
nltk.download('punkt')
stanza.download('es', package='ancora', processors='tokenize,mwt,pos,lemma,depparse', verbose=False)
stNLP = stanza.Pipeline(processors='tokenize, mwt, pos, lemma, depparse', lang='es', package="ancora", use_gpu=True, tokenize_pretokenized=True)



def annotate_conll(file, name, save_to):
    '''
    Insert pos and dependency annotations to conll files

    Parameters
    ----------
    file : str
        path to train, dev or test conll files. End with slash
    name : str
        data set name
    save_to : str
        output folder

    Returns
    -------
    None.

    '''
    df = pd.read_csv(file, sep="\t", encoding="utf-8", quoting=csv.QUOTE_NONE, dtype="object")[:-1] # skip last row if it's nan to avoid index unmatch
    
    text = " ".join(df.Word.astype(str).to_list()).split(' nan ')
    
    tagged_sents = []
    for sent in text:
        pos_tweet = stNLP(sent)
        tagged_sents.append(str([f'{word.text}\t{word.upos}\t{word.deprel}' for s in pos_tweet.sentences for word in s.words]))
    
    tg = [sent for sent in tagged_sents]
    tagged_tokens = pd.DataFrame(data = [*tg], columns=['token'])
    tagged_tokens['token'] = tagged_tokens['token'].apply(lambda x: "splitat".join(literal_eval(x)) + "splitat")
    tagged_tokens = tagged_tokens['token'].str.split("splitat").explode()
    tagged_tokens = tagged_tokens.reset_index()
    tagged_tokens = tagged_tokens[:-1]
    tagged_tokens[["token","UPOS","DEPREL"]] = tagged_tokens["token"].str.split("\t", expand=True)
    tagged_tokens.loc[tagged_tokens["UPOS"].isna(), ["token","UPOS","DEPREL"]] = ""
    
    df['UPOS'] = tagged_tokens.UPOS.to_list()
    df['DEPREL'] = tagged_tokens.DEPREL.to_list()
    df = df[["Doc_Id", "Sent_Id", "Token_Id", "Word", "UPOS", "DEPREL", "Tag", "Begin", "End"]]

    out = save_to + name + ".conll"
    df.to_csv(out, sep="\t", encoding="utf-8", quoting=csv.QUOTE_NONE, index=False)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-file', '--file', required=True,
                         help='path to conll file')
    parser.add_argument('-name', '--name', required=True,
                         help='dataset name')
    parser.add_argument('-save_to', '--save_to', required=True,
                         help='path to conll file')
  
    args = parser.parse_args()
    
    annotate_conll(args.file, args.name, args.save_to)