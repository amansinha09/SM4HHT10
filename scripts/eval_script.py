#!/usr/bin/env python
# coding: utf-8
import argparse
import pandas as pd


def run_eval(arsg):

    mention = pd.read_csv('/home/amansinha/SM4HHT10/data/socialdisner_v3/mentions.tsv', sep='\t')
    pathtopred = args.pathtopred
    # '/home/amansinha/SM4HHT10/submissions/blahblah.tsv'
    predictions = pd.read_csv(args.pathtopred, sep='\t')
    #print(predictions)

    preddict = {}
    golddict = {}
    for tid in set(predictions.tweets_id):
        inpred = predictions[predictions.tweets_id == tid].reset_index()
        ingold = mention[mention.tweets_id == tid].reset_index()
        if args.debug:
            print("gold:")
            print(ingold)
            print("pred:")
            print(inpred)
            print("#"*50)
        
        for i,l in inpred.iterrows():
            if l.tweets_id not in preddict:
                preddict[l.tweets_id] = {(l.begin, l.end): l.extraction}
            else:
                preddict[l.tweets_id][(l.begin, l.end)] = l.extraction
                
        for i,l in ingold.iterrows():
            if l.tweets_id not in golddict:
                golddict[l.tweets_id] = {(l.begin, l.end): l.extraction}
            else:
                golddict[l.tweets_id][(l.begin, l.end)] = l.extraction
                

    total = 0
    strict = 0
    lenient = 0
    for docid in preddict:

        if args.debug: print(preddict[docid], golddict[docid])
        
        for span in golddict[docid]:
            total +=1
            if span in preddict[docid]:
                strict +=1
                lenient += 1
            else:
                pspans = [pspan for pspan in preddict[docid].keys()]
                for ps in pspans:
                    if (ps[0] > span[0] and ps[0] < span[1] ) or (ps[1] > span[0] and ps[0] < span[1] ):
                        lenient +=1
            
             
    print(strict, lenient, total)
    print("Strict Accuracy:", strict/total)
    print("Lenient Accuracy", lenient/total)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-pathtopred', '--pathtopred', required=True,
                         help='complete path to submission tsv file')
    parser.add_argument('-debug', '--debug', action='store_true', help='Print flag')
    
    # Note : create a submissions/ folder in the SM4HHT10/ directory
    
    args = parser.parse_args()


    run_eval(args)
