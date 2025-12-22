
- v0.1.0
    - IMDB is added
    - DBLP is added
    - ACM is added  
- v0.1.1
    - minor bug fixed 
    - SynHIN is added

- v0.1.2
    - Change graph.ndata['labels'] -> graph.ndata['label']
    - labels of single-label classification from one-hot label to numeric labels
        - E.g. [[0, 0, 1], [0, 1, 0], ...] -> [2, 1, ...]