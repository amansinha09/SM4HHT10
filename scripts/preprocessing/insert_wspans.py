# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 16:13:34 2022

@author: Cristina GH

Insert word spans into the conll file + split training data into ctrain-cdev

"""
import pandas as pd
import csv
import numpy as np

type_ = "VALIDATION" # folder name

file = f"data\\data_preprocessed\\training_{type_}_nochars.conll"
mentions = "data\\mentions.tsv"

df = pd.read_csv(file, sep="\t", quoting=csv.QUOTE_NONE, dtype='object').fillna("new_sent")
df.loc[(df.Word == "new_sent") & (df.Sent_Id != "new_sent"), "Word"] = '_'
df['Doc_Id'] = df["Doc_Id"].astype(str)


spans = [0]         # get spans of every word in each tweet
for w in df.Word:
    if w != "new_sent":
        spans.append(len(str(w)) + spans[-1])
        spans.append(spans[-1] + 1)
    else:
        spans.append(0)
        spans.append(0)
        
     
word_spans = [spans[i:i+2] for i in range(0, len(spans), 2)]    # group into 2 [begining, end] of word spans

df_spans = pd.DataFrame(word_spans, columns = ["begin", "end"]) # pass to df
df_spans = df_spans[df_spans["end"] != 0]                       # remove additional noisy spans

data = pd.concat([df, df_spans], axis=1)                        # concat the word spans to data df

for col in data.columns:
    data.loc[data[col] == "new_sent", col] = ""

for col in ["begin", "end"]:
    data[col] = data[col].fillna(999999)
    data[col] = data[col].astype(int).astype(str)
    data.loc[data[col] == "999999", col] = np.nan

data = data[:len(data) -1]

data.loc[(data.Doc_Id.str.len() > 1) & (data.Word == ""), "Word"] = "_"



if type_ == "TRAINING":                                             # Split training into train-dev (80/20)
    sents = data.Sent_Id.dropna().drop_duplicates().to_list()
    split_at_sent = sents[round(80 * len(sents)/100)] # get sent num at position (80% of sents)
    split_data_index = data.index[data.Sent_Id == split_at_sent][0:1][0]
    train_split = data[:int(split_data_index)]
    print("Sents in train:", len(train_split.Sent_Id.dropna().drop_duplicates()))
    print(train_split)
    dev_split = data[split_data_index:]
    print("Sents in dev:", len(dev_split.Sent_Id.dropna().drop_duplicates()))
    print(dev_split)
    
    train_split.to_csv("conll-spans-ctrain.conll", sep="\t", encoding="utf-8", index=None, quoting=csv.QUOTE_NONE)
    dev_split.to_csv("conll-spans-cdev.conll", sep="\t", encoding="utf-8", index=None, quoting=csv.QUOTE_NONE)

else:
    data.to_csv(f"conll-spans-{type_}.conll", sep="\t", encoding="utf-8", index=None, quoting=csv.QUOTE_NONE)