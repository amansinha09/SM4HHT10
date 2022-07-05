# -*- coding: utf-8 -*-
"""
Created on Tue May 10 21:29:29 2022

@author: Cristina GH


Convert TRAINING/ VALIDATION/ tweets with NER mentions to conll and flair formats

"""


import pandas as pd
import csv
import glob
import re
import emoji
import os
import argparse
import numpy as np




def tagSents(tweets_folder, mentions_file):
    '''
    Merge tweets with mentions and convert data to conll

    Parameters
    ----------
    tweets_folder : str
        Path to tweets folder
        
    mentions_file : str
        Path to mentions.tsv


    '''
    data = []
    sent_num = 0 # give sent num to every word in each file (tweet)
    for f in glob.glob(tweets_folder):
        with open(f, 'r', encoding="utf-8") as tweet_file:
            f = f.replace("_utf8","")
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
    
        

    annoying_chars = ["️","⁩","⁦","2⃣"," ", " ", " ", "⁩", "⁦", "‍", " " , " "]
    for char in annoying_chars:
        grouped.loc[grouped.text == char, "text"] = "_"
        grouped.loc[grouped.text.str.contains(char), "text"] = grouped["text"].str.replace(char, " ")
        
    # Pass text and extractions to tuple to add extraction to ever sentence in text
    tagged_text = []
    

    for key, value in list(zip(grouped['extraction'], grouped['text'])):
        key_tag = ["INITTAG" + w + "ENDTAG" for w in key.split("|")]
        key_tag = [w.replace(" ", " MIDTAG") for w in key_tag]
        repls = dict(zip(key.split("|"), key_tag))
        pattern = re.compile("|".join([re.escape(k) for k in repls]), flags=re.DOTALL)
        tg = pattern.sub(lambda x: repls[x.group(0)], value)
        tagged_text.append(tg)
    
    # New col with tagged text
    grouped['tagged_text'] = tagged_text
    grouped['tagged_text'] = grouped['tagged_text'].apply(lambda x: re.sub("INITTAG(.*?)ENDTAG", lambda x:x.group(0).replace(' ',' MIDTAG'), x))
    
    
    grouped['tagged_text'] = grouped['tagged_text'].str.split(" ") # split at explicit space so it wont omit double spaces (they are taken into account in the char count for the spans)
    # grouped['tagged_text'] = grouped['tagged_text'].apply(lambda x: [element or '_' for element in x])
    grouped = grouped.explode(['tagged_text'])
    
    grouped['spans'] = grouped['begin'] + grouped['end'] ## add spans to a single col
    grouped['spans'] = grouped['spans'].apply(lambda x: [sorted(x)[i:i+2] for i in range(0, len(sorted(x)), 2)])

    del grouped['begin']
    del grouped['end']
    
    ## Add token id to every word in a tweet
    grouped['token_id'] = grouped.groupby('sent_id').cumcount() + 1
    
    df = grouped
    ## Insert NER tags column
    df['tag'] = "O"
    # INITTAGArtritis -> B-MISC
    df.loc[(df.tagged_text.str.contains('INITTAG')) & (~df.tagged_text.str.contains('ENDTAG')), 'tag'] = "B-MISC"
    # MIDTAGreumatoideENDTAG -> I-MISC
    df.loc[(~df.tagged_text.str.contains('INITTAG') & (df.tagged_text.str.contains('ENDTAG'))), 'tag'] = "I-MISC"
    # INITTAGansiedadENDTAG -> B-MISC
    df.loc[(df.tagged_text.str.contains('ENDTAG')) & (df.tagged_text.str.contains('INITTAG')), 'tag'] = "B-MISC"
    df.loc[(df.tagged_text.str.contains('MIDTAG')) & (~df.tagged_text.str.contains('ENDTAG')), 'tag'] = "I-MISC"
    
    for tag in ["INITTAG", "ENDTAG", "MIDTAG"]: # remove preprocessing tags from tokenized text
        df['tagged_text'] = df['tagged_text'].str.replace(tag,"")
    
    df.loc[df['tag'] == "O", 'spans'] = "_"
    
    spans = [0]         # get spans of every word in each tweet
    for w in df.tagged_text.to_list():
        if w != "EndOfSent":
            spans.append(len(str(w)) + spans[-1])
            spans.append(spans[-1] + 1)
        if w == "EndOfSent":
            spans.append(0)
            spans.append(0)
            
    word_spans = [spans[i:i+2] for i in range(0, len(spans), 2)]    # group into 2 [begining, end] of word spans

    df_spans = pd.DataFrame(word_spans, columns = ["begin", "end"]) # pass to df
    df_spans = df_spans.dropna()
    
    data_spans = pd.concat([df.reset_index(), df_spans], axis=1)
    
    data_spans.loc[data_spans['end'] == 0, data_spans.columns] = 0
    
    ## Add exceptions on spans
    special_chars = "¿?+;][\/ºª)(,.-!¡_#º=&%$@<>”:*+^'🟢💉💪🔺🦠👇🏼👇🧠🔵💥🔴😱👂💜💪🏻💙🧏‍♂️👆🏻🐝💄🎗🆘➡❌✊🏼✅⚕️😳…🤦‍♀️🙋😥👉🙃💗❤"

    for char in special_chars:
        data_spans.loc[(data_spans['tag'].str.contains('MISC')) & 
                        (data_spans['tagged_text'].apply(lambda x: str(x).endswith(char))), "end"] = data_spans['end'] - 1
        data_spans.loc[(data_spans['tag'].str.contains('MISC')) & 
                        (data_spans['tagged_text'].apply(lambda x: str(x).startswith(char))), "begin"] = data_spans['begin'] + 1
    data_spans.loc[(data_spans['tag'].str.contains('MISC')) & 
                    (data_spans['tagged_text'].apply(lambda x: str(x).endswith('"'))), "end"] = data_spans['end'] - 1
    data_spans.loc[(data_spans['tag'].str.contains('MISC')) & 
                    (data_spans['tagged_text'].apply(lambda x: str(x).endswith('ESP'))), "end"] = data_spans['end'] - 3
    data_spans.loc[(data_spans['tag'].str.contains('MISC')) & 
                    (data_spans['tagged_text'].apply(lambda x: str(x).endswith('".'))), "end"] = data_spans['end'] - 1
   
    for word in ['#DiaMundialdel', '#DiaMundialContraLa', '#DiaMundialContraEl', '#DíaMundialDel',
                  '#DíaMundialContraLa', '#DíaMundialContraEl', '#PrevenciónDe', '#PersonasCon', '#VivirCon',
                  '#Vacuna', '#Vacunación', '#SaludMentaly', '#Prevenir', '#DiaMundialDeLa', '#DíaMundialDeLa',
                  '#comunidad', '#diamundialdela', '#díamundialdela', '#díamundialdel', '#diamundialcontrael',
                  '#Díadel', '#SemanaMundial', '#TodasContraEl', '#TodosContraEl', '#TodasContraLa', '#TodosContraLa',
                  '#TodosSomos', '#TodasSomos', '#Díadela', '#Diadel', '#DiaDeLa', '#DíaDel']:
        
        data_spans.loc[(data_spans['tag'].str.contains('MISC')) & 
                            (data_spans['tagged_text'].apply(lambda x: str(str(x).strip()).startswith(word))), "begin"] = data_spans['begin'] + len(word) - 1  # -1 because of the hashtag, that was considered before
    
    data_spans.loc[data_spans['tagged_text'] == "", 'tagged_text'] = "_"
    data_spans.loc[data_spans.tweets_id == 0, data_spans.columns] = ""
    
    final_data = data_spans[['tweets_id', 'sent_id','token_id', 'tagged_text', 'tag', 'begin', 'end']]
    final_data.columns = ['Doc_Id', 'Sent_Id', 'Token_Id', 'Word', 'Tag', 'Begin', 'End']
    
    
    return final_data




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-tweetsfolder', '--tweetsfolder', required=True, help='input folder')
    
    parser.add_argument('-mentions' ,'--mentions', required=True, help='mentions file')
    
    args = parser.parse_args()

    data = tagSents(args.tweetsfolder, args.mentions)
    
    if "train" in args.tweetsfolder:
        split_at_sent = round(80 * len(data.Sent_Id.dropna().drop_duplicates()) / 100)
        split_idx = data.index[data.Sent_Id.astype(str) == str(split_at_sent)][0:1][0] - 1
    
        ctrain = data[: split_idx]
        cval = data[(split_idx + 1):]
    
        ctrain.to_csv("data\\conll\\conll-spans-_ctrain_enc_final.conll", sep="\t", encoding='utf-8', quoting=csv.QUOTE_NONE, index=False)
        cval.to_csv("data\\conll\\conll-spans-_cdev_enc_final.conll", sep="\t", encoding='utf-8', quoting=csv.QUOTE_NONE, index=False)


    if "val" in args.tweetsfolder:
    
        data.to_csv("data\\conll\\conll-spans-validation_ctest_final.conll", sep="\t", encoding='utf-8', quoting=csv.QUOTE_NONE, index=False)

 
   
    
   
    
   
    
   

   
# data_flair = data[['Word', 'Tag']]

# for char in special_chars:
#     data_flair['Word'] = data_flair['Word'].apply(lambda x: str(x).replace(char, " ").replace('"',' ').strip())
# data_flair.loc[(data_flair['Tag'].astype(str) != '') & (data_flair['Word'].astype(str) == ''), 'Word'] = "_"
# data_flair['Word'] = data_flair['Word'].apply(lambda x: re.sub("\s+", "", x))
# data_flair.to_csv(f"SM4HHT10\\SM4HHT10\\data\\flair\\{name}_flair_nochars_enc.tsv", sep="\t", encoding='utf-8', quoting=csv.QUOTE_NONE, index=False, header=None)
   
   
    
    
# def remove_chars(df):
    ####
    # special_chars = "¿?+;][\/ºª)(,.-!¡_#º=&%$@<>”“:*+^’»'"
    # #🟢💉💪🔺🦠👇🏼👇🧠🔵💥🔴😱👂💜💪🏻💙🧏‍♂️👆🏻🐝💄🎗🆘➡❌✊🏼✅⚕️😳…🤦‍♀️🙋😥👉🙃💗❤"
    
    # data['Word_noChar'] = data['Word']
    # split_at_char = [key for key, value in emoji.UNICODE_EMOJI['es'].items()] + list(special_chars)
    
    # for c in split_at_char:
    #     data.loc[~data['Word_noChar'].str.contains("http"), 'Word_noChar'] = data['Word_noChar'].apply(lambda x: x.replace(c, str("SPLIT" + c + "SPLIT")))
    
    # data['Word_noChar'] = data['Word_noChar'].apply(lambda x: list(filter(None, x.split("SPLIT"))))
    # data = data.explode('Word_noChar')
    # data['Word_noChar'] = data['Word_noChar'].str.strip()
    # data.loc[(data['Word_noChar'] == ""), "Word_noChar"] = "_"
    # data.loc[(data['Token_Id'] == ""), "Word_noChar"] = ""
    # for c in split_at_char:
    #     data.loc[data['Word_noChar'] == c, 'Tag'] = 'O'
    
    # data = data[['Id', 'Sent_Id', 'Token_Id', 'Word', 'Word_noChar', 'Tag', 'Begin', 'End']]
    # ####
    
    
    # for char in special_chars:
    #     data['Word_noChar'] = data['Word_noChar'].apply(lambda x: str(x).replace(str(char), ""))
    #     data['Word_noChar'] = data['Word_noChar'].apply(lambda x: str(x).replace(str('"'), ""))
    # data.loc[(data['Word_noChar'].astype(str) == "") & (data['Word'].astype(str) != ""), 'Word_noChar'] = "_"
