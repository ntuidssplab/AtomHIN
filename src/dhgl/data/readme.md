


## Quick Start

There are three heterogeneous datasets available, which are

- [ACM](#acm)
- [DBLP](#dblp)
- [IMDB](#imdb)

They can be easily imported using

```python
from dgl.data import (
    HeteroACMDataset,
    HeteroDBLPDataset,
    HeteroIMDBDataset,
)
```

All of the datasets are the subclass of [`dgl.data.DGLDataset`](https://docs.dgl.ai/en/1.1.x/generated/dgl.data.DGLDataset.html#dgl.data.DGLDataset).

```python
>>> issubclass(HeteroACMDataset, DGLDataset)
True
```

The datasets cannot be automatically downloaded at the moment.
As a result, you will need to download them manually and specify the path to the downloaded directory.

```python
>>> raw_path = '/path/to/raw/data/dir'
>>> dataset = HeteroACMDataset(raw_path=raw_path)
```

All of the currently-avaliable datasets contain only one graph which is a [`dgl.DGLGraph`](https://docs.dgl.ai/en/1.1.x/api/python/dgl.DGLGraph.html) and more precisely those graphs follow `dhgl.schema.BaseGraphSchema`.


```python
# There is only one graph in ACM dataset
>>> len(dataset)
1

>>> hg = dataset[0]
>>> isinstance(hg, dgl.DGLGraph)
True
```

`dhgl.hgget` provides various handy apis to access the hetero-graph data.

```python
>>> from dhgl import hgget
>>> hgget.info(hg)
Dimensions of features (#samples, #feature_dims):
        author:         (5988, 1902)
        paper:          (3025, 1902)
        subject:        (52, 1902)
        term:           (1902, 1902)
src_ntype    etype       dst_ntype      #edges
-----------  ----------  -----------  --------
author       writing     paper            9991
paper        cited       paper           10431
paper        citing      paper           10431
paper        contains    term           257722
paper        is-about    subject          3025
paper        written-by  author           9991
subject      has         paper            3025
term         is-in       paper          257722
Task: 3-class single-label classification
        Target "paper":
        Splits (train, valid, test): (724, 183, 2118)
```



## API Reference

### ACM 

- Label masks

```python
>>> hg.ndata['train_mask']
>>> hg.ndata['val_mask']
>>> hg.ndata['test_mask']
```

In the case of ACM, the target node type is `paper`. Thus the above code is equivalent to 

```python
>>> hg.nodes['paper'].data['train_mask']
>>> hg.nodes['paper'].data['val_mask']
>>> hg.nodes['paper'].data['test_mask']
```

- `label`

```python
>>> hg.ndata['label']
```

or, equivalently

```python
>>> hg.nodes['paper'].data['label']
```


- `feat`

```python
>>> hg.ndata['feat']
```


---

### DBLP

- Label masks

```python
>>> hg.ndata['train_mask']
>>> hg.ndata['val_mask']
>>> hg.ndata['test_mask']
```

In the case of DBLP, the target node type is `author`. Thus the above code is equivalent to 

```python
>>> hg.nodes['author'].data['train_mask']
>>> hg.nodes['author'].data['val_mask']
>>> hg.nodes['author'].data['test_mask']
```

- `label`

```python
>>> hg.ndata['label']
```

or, equivalently

```python
>>> hg.nodes['author'].data['label']
```


- `feat`

```python
>>> hg.ndata['feat']
```

---

### IMDB

- Label masks

```python
>>> hg.ndata['train_mask']
>>> hg.ndata['val_mask']
>>> hg.ndata['test_mask']
```

In the case of IMDB, the target node type is `movie`. Thus the above code is equivalent to 

```python
>>> hg.nodes['movie'].data['train_mask']
>>> hg.nodes['movie'].data['val_mask']
>>> hg.nodes['movie'].data['test_mask']
```

- `label`

```python
>>> hg.ndata['label']
```

or, equivalently

```python
>>> hg.nodes['movie'].data['label']
```


- `feat`

```python
>>> hg.ndata['feat']
```