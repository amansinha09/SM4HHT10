# -*- coding: utf-8 -*-
"""
Created on Tue May 10 21:29:29 2022

@author: Cristina GH


Convert TRAINING/ VALIDATION/ tweets with NER mentions to conll format

"""

import pandas as pd
import csv
import glob
import re
import emoji
import os
import argparse


def tagSents(tweets_folder, mentions_file):# merge two files (tweets with id, to mentions)
    data = []
    sent_num = 0 # give sent num to every word in each file (tweet)
    for f in glob.glob(tweets_folder):
        with open(f, 'r', encoding="utf-8") as tweet_file:
            file_tweet_id = f.split("\\")[-1].replace('.txt','')
            file_read = tweet_file.read().strip()
            sent_num += 1
            data.append(str(file_tweet_id + "_SEP_" + file_read + "_SEP_" + str(sent_num)))
            
    data_df = pd.DataFrame(list(zip([int(i.split('_SEP_')[0]) for i in data],
                                    [i.split('_SEP_')[1] for i in data],
                                    [i.split("_SEP_")[-1] for i in data])), 
                            columns=["tweets_id", "text", "sent_id"])
    
    data_df['text'] = data_df['text'] + " EndOfSent"
    data_df = data_df.drop_duplicates(subset="text") # remove tweet duplicates if any
    
    mentions = pd.read_csv(mentions_file, sep="\t", quoting=csv.QUOTE_NONE)
    
    merged_data = mentions.merge(data_df, how='outer', on="tweets_id")

    merged_data['text'] = merged_data['text'].str.replace("\n", " ", regex=True)
    
    
    # sort data by sentence id & group values
    sorted_mdata = merged_data.sort_values(by="sent_id", ascending=False)
    grouped = sorted_mdata.groupby(['tweets_id','type','text', 'sent_id']).agg({'extraction': 
                                                                                  lambda x: "|".join(sorted(x.to_list(), key=len)[::-1]),
                                                                                  'begin': lambda x: x.to_list(),
                                                                                  'end': lambda x: x.to_list()}).reset_index()
    

    # Pass text and extractions to tuple to add extraction to ever sentence in text
    tagged_text = []
    
    for key, value in list(zip(grouped['extraction'], grouped['text'])):
        key_tag = ["INITTAG" + w + "ENDTAG" for w in key.split("|")]
        repls = dict(zip(key.split("|"), key_tag))
        pattern = re.compile("|".join([re.escape(k) for k in repls]), flags=re.DOTALL)
        tg = pattern.sub(lambda x: repls[x.group(0)], value)
        tagged_text.append(tg)
    
    # New col with tagged text
    grouped['tagged_text'] = tagged_text
    grouped['tagged_text'] = grouped['tagged_text'].apply(lambda x: re.sub("INITTAG(.*?)ENDTAG", lambda x:x.group(0).replace(' ',' MIDTAG'), x))
    
    return grouped

def df_to_conll(labeled_data_df):
    df = labeled_data_df

    # space_at = list(",;.:_?¿¡!€$~/\{}[]=><()'\"”“")
    
    # for char in space_at:
    #     df['tagged_text'] = df['tagged_text'].astype(str).str.strip()
    #     df.loc[df['tagged_text'].str.contains(str(char), regex=False), 
    #             'tagged_text'] = df['tagged_text'].apply(lambda x: " ".join([w.replace(str(char), str(" " + char + " ")) for w in x.split(" ") if "http" not in w]))
    
    df['tagged_text'] = df['tagged_text'].astype(str).str.strip()

    df['tagged_text'] = df['tagged_text'].str.replace("INITTAG", " INITTAG")
    df['tagged_text'] = df['tagged_text'].str.replace("ENDTAG", "ENDTAG ") # theses espaces create +1 space counting when #[begin]cataratas[end] (len 10) -> # [begin]cataratas[end] (len 11)
    # df['tagged_text'] = df['tagged_text'].apply(lambda x: x.replace(". . .", "..."))
    # df['tagged_text'] = df['tagged_text'].apply(lambda x: x.replace(".  .  .", "..."))
    df['tagged_text'] = df['tagged_text'].str.split(" ")
    
    df = df.explode('tagged_text').reset_index(drop=True)
    df['tagged_text'] = df['tagged_text'].str.strip()
    df = df[df.tagged_text != ""]
    df['token_id'] = df.groupby('sent_id').cumcount() + 1
    
    df['tag'] = "O"
    df.loc[(df.tagged_text.str.startswith('INITTAG')) & (~df.tagged_text.str.endswith('ENDTAG')), 'tag'] = "B-MISC"
    df.loc[(df.tagged_text.str.endswith('ENDTAG')) & (~df.tagged_text.str.startswith('INITTAG')), 'tag'] = "I-MISC"
    df.loc[(df.tagged_text.str.endswith('ENDTAG')) & (df.tagged_text.str.startswith('INITTAG')), 'tag'] = "B-MISC"
    df.loc[(df.tagged_text.str.startswith('MIDTAG')) & (~df.tagged_text.str.endswith('ENDTAG')), 'tag'] = "I-MISC"
    
    for key, value in emoji.UNICODE_EMOJI['es'].items():
        df.loc[~df['tagged_text'].str.contains("TAG"), 'tagged_text'] = df['tagged_text'].apply(lambda x: x.replace(key, str(key + "SPLIT")))
    df['tagged_text'] = df['tagged_text'].str.strip()
    df['tagged_text'] = df['tagged_text'].str.split("SPLIT")
    df = df.explode('tagged_text').reset_index(drop=True)
    df = df[df.tagged_text != ""]
    
    for tag in ["INITTAG", "ENDTAG", "MIDTAG"]:
        df['tagged_text'] = df['tagged_text'].str.replace(tag,"")
    
    df = df[["tweets_id", "sent_id", "tagged_text", "tag"]].reset_index()
    df.columns = ["Id", "Doc_Id", "Sent_Id", "Word", "Tag"]
    
    df = df.fillna("_")
    df.loc[df["Word"] == "️", "Word"] = "_"
    
    annoying_chars = ["️","⁩","⁦","2⃣"," ", " ", " ", "⁩", "⁦", "‍", " "]
    for char in annoying_chars:
        df.loc[df.Word == char, "Word"] = "_"
        df.loc[df.Word.str.contains(char), "Word"] = df["Word"].str.replace(char, "")
    
    df.fillna("_")
    df.loc[df.Word == "", "Word"] = "_"
    df.loc[df.Word == "EndOfSent", df.columns] = ""
    
    return df


save_to = "data\\data_preprocessed"
tweets_folder = "data\\training\\*.txt"
mentions = "data\\mentions.tsv"



labeled_data = tagSents(tweets_folder, mentions)
conll_df = df_to_conll(labeled_data)    
    ## save to flair/normal input format
output_conll = save_to + "\\training_TRAINING_nochars.conll"
conll_df.to_csv(output_conll, sep="\t", encoding="utf-8", index=False, quoting=csv.QUOTE_NONE)

output_flair = save_to + "\\training_TRAINING_flair_nochars.tsv"
conll_flair = conll_df[['Word', 'Tag']]
conll_flair.loc[conll_flair['Word'] == "EndOfSent", ['Word', 'Tag']] = ""
conll_flair.columns = ["text", "ner"]
conll_flair.to_csv(output_flair, sep="\t", encoding="utf-8", index=False, header=None, quoting=csv.QUOTE_NONE)
