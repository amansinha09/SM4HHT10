# How to run Graph Algorithms for NER



## Todo

- [ ] DeepWALK-nn
- [ ] TransE-nn
- [x] GraphSAGE
- [x] GCN
- [x] GAT 


```
SAGE: python3 train_sage.py --dataset mypubmed --gpu 0 --n-ep 500
GAT : python3 train_gat.py --dataset=mypubmed --gpu=0 --num-out-heads=8 --weight-decay=0.001 --early-stop --num-layer 2 --epochs 500
GCN : python3 train_gcn.py --dataset mypubmed --gpu 0 --self-loop --n-ep 500

```
