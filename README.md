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
