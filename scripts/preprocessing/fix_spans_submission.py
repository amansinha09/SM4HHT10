# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 12:31:42 2022

@author: Cristina GH
"""

import pandas as pd
import csv
import glob
from tqdm import tqdm


diseases_list_file = "SM4HHT10\\SM4HHT10\\data\\diseases_list.tsv"

def fix_spans(submission_file):
    with open(diseases_list_file, 'r', encoding="utf-8") as f:
        diseases_list = [line.strip() for line in f.readlines()]
        df_dis = pd.DataFrame(data = diseases_list, columns=['disease'])  # df with external list of diseases
        
        df_dis['disease'] = df_dis['disease'].str.lower()
        df_dis['disease'] = df_dis['disease'].drop_duplicates()
        df_dis['disease'] = df_dis['disease'].str.split()
        df_dis = df_dis.explode('disease')
        df_dis = df_dis[df_dis['disease'].str.len() > 3].drop_duplicates()
        df_dis['disease'] = df_dis['disease'].str.strip()
        df_dis = df_dis.sort_values(by="disease", key=lambda x: x.str.len(), ascending=False)
        
        # list with dissease list + agglometared disease list (dolor neuropatico, dolorneuropatico) as those disseases can
        # appear as well under a hashtag (#dolorneuropatico)
        diseases = sorted(diseases_list + df_dis['disease'].to_list(), key=len, reverse=True)
    
        df = pd.read_csv(submission_file, sep="\t", encoding="utf-8", quoting=csv.QUOTE_NONE, dtype='str')
        
        # df['gold_begin'] = df.begin   # check original spans
        # df['gold_end'] = df.end
        
        df['noise'] = "_ _"
        

        diseases = ["".join(d.strip()).lower() for d in diseases]
        for dis in diseases:    
            df.loc[(df['extraction'].apply(lambda x: dis in str(x).lower())) &
                (df.noise == "_ _"), 
                'noise'] = df['extraction'].apply(lambda x: str(str(x).lower()).split(dis))

        df['noise2'] = df.noise
        df.loc[df.noise == "_ _", 'noise'] = df['noise'].str.split('_')
        df.loc[df['noise'] != "_", 'noise_left'] = df['noise'].apply(lambda x: len(x[0]))
        df.loc[df['noise'] != "_", 'noise_right'] = df['noise'].apply(lambda x: len(x[1]))
        
        df['begin'] = df['begin'].astype(int) + df['noise_left']
        df['end'] = df['end'].astype(int) - df['noise_right']
        
        # now, if right and left are 0 in the spans (unkown disease or no noise), if this word starts or ends
        # with special char, add or rest char in right or left
        df['new_word'] = "999"
        df.loc[(df.noise_right == 0) & (df.noise_left == 0), 'new_word'] = df.extraction
        
        li_char = ",.;:-_+´`^¨{[}!ºª¿?*]¡'()/&¬$~·#@|!%<>”“:*+^’»\🟢💉💪🔺🦠👇🏼👇🧠🔵💥🔴😱👂💜💪🏻💙🧏‍♂️👆🏻🐝💄🎗🆘➡❌✊🏼✅⚕️😳…🤦‍♀️🙋😥👉🙃💗❤"
        for char in list(li_char):
            df.loc[(df.new_word.apply(lambda x: str(x).startswith(char))) & (df.new_word != "999"), "noise_left"] = 1
            df.loc[(df.new_word.apply(lambda x: str(x).endswith(char))) & (df.new_word != "999"), "noise_right"] = 1
            
        df.loc[(df.noise_left == 1) & (df.new_word != "999"), "begin"] = df.begin + 1
        df.loc[(df.noise_right == 1) & (df.new_word != "999"), "end"] = df.end - 1
    
        for col in ['noise_right','noise','noise_left', 'new_word']:
            del df[col]
        
        name = submission_file.split('\\')[-1].replace(".tsv", "_char_postprocessing.tsv")
    
        df.to_csv(f"SM4HHT10\\SM4HHT10\\logs\\fixed_sub\\{name}_no_diseases", sep="\t", encoding='utf-8', index=None, quoting=csv.QUOTE_NONE)
        return df
    

# data = fix_spans("SM4HHT10\\SM4HHT10\\Flair_official\\transformers\\submissions\\dev_bbmcn_submission__.tsv")

for file in tqdm(glob.glob("SM4HHT10\\SM4HHT10\\logs\\testing_submission_GOLD_official_dev_validation-fix.tsv")):
    print()
    print(file)
    print()
    data = fix_spans(file)