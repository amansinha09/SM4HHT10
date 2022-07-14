# SM4HHT10

SPanish NER : Reconocimiento de Entidad Nombrada


## ToDo


- [x] Add the dataset
- [x] Make list of the benchmarks
- [x] create conll3 dataformat converter 
- [x] eval script
- [ ] Try out benchmarks and update readme
- [ ] Try out graphner and update
- [x] Extract POS 

## Data Version description

|version|desciption|
|---|---|
|v1|C ; ??|
|v2|A ; space based tokenization |

## To run evaluation using official script

```
python official_eval.py ./../submissions/testgold_mana_without_mergingfix.tsv ./../data/socialdisner_v3/mentions.tsv ./
```
Steps to prepare the data

1. Use `to_conll.py` to convert the documents and the mentions into a conll file. For that, pass the mention file and the path to the folder containing the text documents. What preprocessing is done:  
- Every work is tokenized on space
- Word spans are added based on the tokenization, so spans will need to be fixed afterwars on the generated submission file

2. The next step is to add POS and DEPREL information. Use `annotate_conll.py` and pass the conll file from previous step, a name for the set and the destination folder.

