# SM4HHT10

Code repository for the paper : 
Team IAI @ SocialDisNER : Catch me if you can! Capturing complex disease mentions in tweets


## 🥘 Steps quick overview
- Generate the conll file `to_conll.py`
- Insert POS/deprel if needed `annotate_conll.py`
- Check no special chars on genearted files: ⁩|⁦|⁦|⁦|⁦|⁦|2⃣| | |
‍|⁦| 

- Save files with fixed chars if any.
- For flair emb/transformers, use conll files. A flair formatted file will be created from them. `train_flair.py` `flair_transformers.py`
- Create the submission file using the predictions from FLAIR and the used tes file `prediction2submissionformat.py` or `submission.ipynb`
- Fix the spans `fix_spans_on_submission.py`
- Run eval scripts `eval_script.py` or `official_eval.py`

## Steps to prepare the data

1. Use `to_conll.py` to convert the documents and the mentions into a conll file. For that, pass the mention file and the path to the folder containing the text documents. 
```
python scripts\to_conll.py -tweetsfolder data\socialdisner_v3\train-valid-txt-files\validation\* -mentions data\socialdisner_v3\mentions.tsv
```
What preprocessing is done:  
- Every work is tokenized on space
- Word spans are added based on the tokenization, so spans will need to be fixed afterwars on the generated submission file

2. The next step is to add POS and DEPREL information. Use `annotate_conll.py` and pass the conll file from previous step, a name for the set and the destination folder.
```
python scripts\preprocessing\annotate_conll.py -file data\conll\training_cdev_final.tsv -name validation -save_to data\conll\
```

## Sample instruction run the models

1. Flair-S
```
python ~/SM4HHT10/benchmarks/train_flair.py  --train ../data/conll/official_train.conll \
                                                    --val ../data/conll/official_dev_validation.conll \
                                                    --test  ../data/conll/official_test.conll \
                                                    --EPOCHS 50\
                                                    --save_to ./sf_spainish_50/ \
                                                    --lm back_forw_clinical
```

2. Flair-T

```
python ~/SM4HHT10/benchmarks/flair_transformers.py  --train ../data/conll/official_train.conll \
                                                    --val ../data/conll/official_dev_validation.conll \
                                                    --test  ../data/conll/official_test.conll \
                                                    --EPOCHS 15\
                                                    --save_to ./resources_wmn/ \
                                                    --lm Babelscape/wikineural-multilingual-ner

```

## To run evaluation using official script

```
python official_eval.py ./../submissions/testgold_mana_without_mergingfix.tsv ./../data/socialdisner_v3/mentions.tsv ./
```


** notes
- Remove special chars on generated conll files using regex
- When running flair changing the slash // to \\ is linux. Also, when passing the argument save_to, folder must end with slash
