#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import csv
import numpy as np
import argparse
from tqdm import tqdm

from sklearn.metrics import classification_report


def peek_stack(stack):
    if stack:
        return stack[-1]    # this will get the last element of stack
    else:
        return None

def merge_spans2(spans=[(113, 125), (126, 137), (185, 194), (196, 219)], debug=False):
    
    if debug : print(spans)
    spans = np.asarray(spans)
    stack = []
    for s in spans.reshape(1,-1)[0]:
        if debug : print(s)
        if len(stack) == 0:
            stack.append(s)
        else:
            if s - peek_stack(stack) == 1:
                stack.pop()
            else:
                stack.append(s)
        if debug : print("current stack: ", stack)
    stack = np.asarray(stack)
    return stack.reshape(-1,2)
    
#merge_spans2(debug=True)    

def create_submission_file(args):

    path = args.folder

    # '../data/conll/validation_v2.conll'
    validation = pd.read_csv(args.pathtotestconll, sep='\t',quoting=csv.QUOTE_NONE, encoding='utf-8')
    # '/home/amansinha/Downloads/test.tsv' 
    # check format sep
    predictions = pd.read_csv(args.pathtoprediction, sep='\s', names=['word', 'gold', 'pred'])

    validation['pred'] = predictions.pred

    print(classification_report(predictions.gold, predictions.pred, labels=['B-MISC', 'I-MISC']))

    print('converting to submission format')
    with open(f"{args.save_to}blah.tsv", "w") as fp:
        print('tweets_id','begin','end','type','extraction',sep='\t', file=fp)
        for i in tqdm(range(1,args.nb_testsamples+1)):
            info = validation[validation.Sent_Id == i]
            #print(info)
            l = info[info.pred != 'O']
            if len(l) == 0: continue
            docid = l.Doc_Id.values[0]
            #print(docid)
            f = open(f'{path}{docid}.txt', 'r')
            # replace tab with space then strip extra spaces
            text = f.read().replace('\n',' ').strip()
            #print(text)
            #print('*'*20)
            #print(l.Doc_Id.values[0], l.begin.values[0], l.end.values[0], l.pred.values[0])
            spans = [[s,e] for s,e in zip(l.begin, l.end)]
            #print(docid, spans)
            spans = merge_spans2(spans)
            #print(spans)
            for span in spans:
                print(docid,span[0], span[1], 'ENFERMEDAD', text[span[0]: span[1]], sep='\t', file=fp)

            "print("#"*20)
            #break
        
    print(f'submission file saved to {args.save_to}')




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-folder', '--folder', default='data/socialdisner_v3/train-valid-txt-files/validation/',
                         help='test tweet folder')

    parser.add_argument('--pathtotestconll', required=True, help = 'Complete path to test conll file')
    parser.add_argument('--pathtoprediction', required=True, help = 'Complete path to prediction file')

    
    parser.add_argument('-save_to' ,'--save_to', default='submissions/', help='save tsv file to?')

    parser.add_argument('-nb_testsamples' ,'--nb_testsamples', default=2500, help='Numbre of test samples')
    
    # Note : create a submissions/ folder in the SM4HHT10/ directory
    
    args = parser.parse_args()

    create_submission_file(args)

    # python scripts/prediction2submissionformat.py --pathtotest data/conll/old-data/validation_v2.conll --pathtopred ~/Downloads/test.tsv