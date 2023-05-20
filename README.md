# Knowledge Graph Embeddings using ontology
### JOIE and TransO with ReasonKGE Negative sampling
Train and learn JOIE, TransO and ReasonKGE embedding models on DBpedia and YAGO datasets <br>
All implementations are based on ReasonKGE framework.
### Source article:
https://dl.acm.org/doi/10.1145/3292500.3330838 (JOIE 2019 ver) <br>
https://arxiv.org/abs/2103.08115 (JOIE 2021 ver) <br>
https://dl.acm.org/doi/abs/10.1007/s11280-022-01016-3 (TransO) <br>
https://github.com/nitishajain/ReasonKGE (ReasonKGE Negative Sampling procedure) <br>
#### This implementation is done as a part of MSc graduate diploma for Applied Informatics study programme in the Financial University under the Russian Federation Government
#### It is written with OOP pattern and allows to run training process for the different combinations of:
* JOIE Cross-View Grouping model (with / without Hierarchy-Aware part) <br>
* JOIE Cross-View Transformtaion model (with / without Hierarchy-Aware part) <br>
* TransO model <br>
#### Base models include:
* TransE <br> (JOIE / TransO)
* DistMult <br> (JOIE)
* HolE <br> (JOIE)
#### Each model can be run with the one of the following Negative Sampling Techniques: 
* Uniform Random <br>
* Bernoulli Random <br>
* ReasonKGE (uses ontology reasoning based on Description Logic) <br>
