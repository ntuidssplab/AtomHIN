# Atomic HINs

This repository contains the official implementation of “Atomic HINs: Entity–Attribute Duality for Heterogeneous Graph Modeling” (ICLR 2026).

- [Install and Setup](#install-and-setup)
- [Quick Start: Load Released Datasets](#quick-start-load-released-datasets)
  - [Released Atomic Datasets](#released-atomic-datasets)
  - [Schema Refinement (Atomic Datasets)](#schema-refinement-atomic-datasets)
  - [Vanilla Datasets](#vanilla-datasets)
- [Training, Evaluation, and Schema Search](#training-evaluation-and-schema-search)


## Install and Setup

### 1. Clone the repository

```sh
git clone <repo>
cd AtomHIN
```

### 2. Install PyTorch & DGL

Install PyTorch and DGL with the CUDA version that matches your device.
Helper scripts such as `cuda118.sh` or `cuda121.sh` may be useful.

### 3. Install the package

* **Library only** (minimal install):

```sh
pip install -e .
```

* **With training scripts** (for node-level and link-level tasks):

```sh
pip install -e .[scripts]
```

* **With precomputation support** (e.g., OGBN-MAG):

```sh
pip install -e .[precom]
```

* **With Ray Tune support** (for schema optimization):

```sh
pip install -e .[ray]
```

⚠️ **Note:** Do **not** install with `requirements.txt` — it is for debugging only.

---

## Quick Start: Load Released Datasets

We provide a collection of **released heterogeneous graph datasets** with multiple
schema profiles. Each dataset can be loaded with a single API call:

```python
import dhgl

hg = dhgl.get_dataset(name, profile)
```

### Released Atomic Datasets

The table below summarizes all released **atomic datasets** and their supported
profiles, including detailed node- and edge-type statistics.


| name | profile | node info | edge info |
|---|---|---|---|
| `atomic-imdb` | `atomic` | <details><summary>Click to expand (nodes)</summary><table><thead><tr><th>ntype</th><th>#samples</th><th>feat_dim</th></tr></thead><tbody><tr><td>actor</td><td>6124</td><td>0</td></tr><tr><td>color</td><td>3</td><td>0</td></tr><tr><td>content_rating</td><td>16</td><td>0</td></tr><tr><td>country</td><td>65</td><td>0</td></tr><tr><td>director</td><td>2393</td><td>0</td></tr><tr><td>keyword</td><td>7971</td><td>0</td></tr><tr><td>language</td><td>48</td><td>0</td></tr><tr><td>movie</td><td>4932</td><td>0</td></tr><tr><td>numerical</td><td>16</td><td>0</td></tr><tr><td>word</td><td>3341</td><td>0</td></tr></tbody></table></details> | <details><summary>Click to expand (edges)</summary><table><thead><tr><th>srctype</th><th>etype</th><th>dsttype</th><th>#edges</th></tr></thead><tbody><tr><td>actor</td><td>acts</td><td>movie</td><td>14779</td></tr><tr><td>color</td><td>is-type-of</td><td>movie</td><td>4932</td></tr><tr><td>content_rating</td><td>is-rating-for</td><td>movie</td><td>4932</td></tr><tr><td>country</td><td>is-country-of</td><td>movie</td><td>4932</td></tr><tr><td>director</td><td>directed</td><td>movie</td><td>4932</td></tr><tr><td>keyword</td><td>is-in</td><td>movie</td><td>23610</td></tr><tr><td>language</td><td>is-language-of</td><td>movie</td><td>4932</td></tr><tr><td>movie</td><td>contains</td><td>keyword</td><td>23610</td></tr><tr><td>movie</td><td>contains-word</td><td>word</td><td>31335</td></tr><tr><td>movie</td><td>directed-by</td><td>director</td><td>4932</td></tr><tr><td>movie</td><td>has-color</td><td>color</td><td>4932</td></tr><tr><td>movie</td><td>has-numerical</td><td>numerical</td><td>78912</td></tr><tr><td>movie</td><td>has-rating</td><td>content_rating</td><td>4932</td></tr><tr><td>movie</td><td>is-from-country</td><td>country</td><td>4932</td></tr><tr><td>movie</td><td>is-in-language</td><td>language</td><td>4932</td></tr><tr><td>movie</td><td>stars</td><td>actor</td><td>14779</td></tr><tr><td>numerical</td><td>is-numerical-of</td><td>movie</td><td>78912</td></tr><tr><td>word</td><td>is-word-of</td><td>movie</td><td>31335</td></tr></tbody></table></details> |
| `atomic-imdb` | `srgcn` | <details><summary>Click to expand (nodes)</summary><table><thead><tr><th>ntype</th><th>#samples</th><th>feat_dim</th></tr></thead><tbody><tr><td>actor</td><td>6124</td><td>5879</td></tr><tr><td>content_rating</td><td>16</td><td>5879</td></tr><tr><td>director</td><td>2393</td><td>5879</td></tr><tr><td>keyword</td><td>7971</td><td>5879</td></tr><tr><td>movie</td><td>4932</td><td>5879</td></tr><tr><td>numerical</td><td>16</td><td>5879</td></tr><tr><td>word</td><td>3341</td><td>5879</td></tr></tbody></table></details> | <details><summary>Click to expand (edges)</summary><table><thead><tr><th>srctype</th><th>etype</th><th>dsttype</th><th>#edges</th></tr></thead><tbody><tr><td>actor</td><td>acts</td><td>movie</td><td>14779</td></tr><tr><td>content_rating</td><td>is-rating-for</td><td>movie</td><td>4932</td></tr><tr><td>director</td><td>directed</td><td>movie</td><td>4932</td></tr><tr><td>keyword</td><td>is-in</td><td>movie</td><td>23610</td></tr><tr><td>movie</td><td>contains</td><td>keyword</td><td>23610</td></tr><tr><td>movie</td><td>contains-word</td><td>word</td><td>31335</td></tr><tr><td>movie</td><td>directed-by</td><td>director</td><td>4932</td></tr><tr><td>movie</td><td>has-numerical</td><td>numerical</td><td>78912</td></tr><tr><td>movie</td><td>has-rating</td><td>content_rating</td><td>4932</td></tr><tr><td>movie</td><td>stars</td><td>actor</td><td>14779</td></tr><tr><td>numerical</td><td>is-numerical-of</td><td>movie</td><td>78912</td></tr><tr><td>word</td><td>is-word-of</td><td>movie</td><td>31335</td></tr></tbody></table></details> |
| `freebase` | `atomic` | <details><summary>Click to expand (nodes)</summary><table><thead><tr><th>ntype</th><th>#samples</th><th>feat_dim</th></tr></thead><tbody><tr><td>book</td><td>40402</td><td>0</td></tr><tr><td>business</td><td>7153</td><td>0</td></tr><tr><td>film</td><td>19427</td><td>0</td></tr><tr><td>location</td><td>9368</td><td>0</td></tr><tr><td>music</td><td>82351</td><td>0</td></tr><tr><td>organization</td><td>2731</td><td>0</td></tr><tr><td>people</td><td>17641</td><td>0</td></tr><tr><td>sports</td><td>1025</td><td>0</td></tr></tbody></table></details> | <details><summary>Click to expand (edges)</summary><table><thead><tr><th>srctype</th><th>etype</th><th>dsttype</th><th>#edges</th></tr></thead><tbody><tr><td>book</td><td>book-about-organization</td><td>organization</td><td>21900</td></tr><tr><td>book</td><td>book-and-book</td><td>book</td><td>202674</td></tr><tr><td>book</td><td>book-and-book-inv</td><td>book</td><td>202674</td></tr><tr><td>book</td><td>book-on-location</td><td>location</td><td>26921</td></tr><tr><td>book</td><td>book-on-sports</td><td>sports</td><td>6615</td></tr><tr><td>book</td><td>book-to-film</td><td>film</td><td>38299</td></tr><tr><td>book</td><td>business-about-book-inv</td><td>business</td><td>18625</td></tr><tr><td>book</td><td>music-in-book-inv</td><td>music</td><td>31486</td></tr><tr><td>book</td><td>people-to-book-inv</td><td>people</td><td>35587</td></tr><tr><td>business</td><td>business-about-book</td><td>book</td><td>18625</td></tr><tr><td>business</td><td>business-about-film</td><td>film</td><td>8397</td></tr><tr><td>business</td><td>business-about-music</td><td>music</td><td>24764</td></tr><tr><td>business</td><td>business-about-sports</td><td>sports</td><td>610</td></tr><tr><td>business</td><td>business-and-business</td><td>business</td><td>4448</td></tr><tr><td>business</td><td>business-and-business-inv</td><td>business</td><td>4448</td></tr><tr><td>business</td><td>business-on-location</td><td>location</td><td>6647</td></tr><tr><td>business</td><td>organization-for-business-inv</td><td>organization</td><td>1073</td></tr><tr><td>business</td><td>people-in-business-inv</td><td>people</td><td>5378</td></tr><tr><td>film</td><td>book-to-film-inv</td><td>book</td><td>38299</td></tr><tr><td>film</td><td>business-about-film-inv</td><td>business</td><td>8397</td></tr><tr><td>film</td><td>film-and-film</td><td>film</td><td>87838</td></tr><tr><td>film</td><td>film-and-film-inv</td><td>film</td><td>87838</td></tr><tr><td>film</td><td>location-in-film-inv</td><td>location</td><td>21299</td></tr><tr><td>film</td><td>music-in-film-inv</td><td>music</td><td>11291</td></tr><tr><td>film</td><td>organization-in-film-inv</td><td>organization</td><td>13128</td></tr><tr><td>film</td><td>people-to-film-inv</td><td>people</td><td>17604</td></tr><tr><td>film</td><td>sports-in-film-inv</td><td>sports</td><td>6763</td></tr><tr><td>location</td><td>book-on-location-inv</td><td>book</td><td>26921</td></tr><tr><td>location</td><td>business-on-location-inv</td><td>business</td><td>6647</td></tr><tr><td>location</td><td>location-and-location</td><td>location</td><td>47817</td></tr><tr><td>location</td><td>location-and-location-inv</td><td>location</td><td>47817</td></tr><tr><td>location</td><td>location-in-film</td><td>film</td><td>21299</td></tr><tr><td>location</td><td>music-on-location-inv</td><td>music</td><td>42915</td></tr><tr><td>location</td><td>organization-on-location-inv</td><td>organization</td><td>2696</td></tr><tr><td>location</td><td>people-on-location-inv</td><td>people</td><td>15134</td></tr><tr><td>location</td><td>sports-on-location-inv</td><td>sports</td><td>656</td></tr><tr><td>music</td><td>business-about-music-inv</td><td>business</td><td>24764</td></tr><tr><td>music</td><td>music-and-music</td><td>music</td><td>283670</td></tr><tr><td>music</td><td>music-and-music-inv</td><td>music</td><td>283670</td></tr><tr><td>music</td><td>music-for-sports</td><td>sports</td><td>8975</td></tr><tr><td>music</td><td>music-in-book</td><td>book</td><td>31486</td></tr><tr><td>music</td><td>music-in-film</td><td>film</td><td>11291</td></tr><tr><td>music</td><td>music-on-location</td><td>location</td><td>42915</td></tr><tr><td>music</td><td>organization-to-music-inv</td><td>organization</td><td>10702</td></tr><tr><td>music</td><td>people-to-music-inv</td><td>people</td><td>10948</td></tr><tr><td>organization</td><td>book-about-organization-inv</td><td>book</td><td>21900</td></tr><tr><td>organization</td><td>organization-and-organization</td><td>organization</td><td>1101</td></tr><tr><td>organization</td><td>organization-and-organization-inv</td><td>organization</td><td>1101</td></tr><tr><td>organization</td><td>organization-for-business</td><td>business</td><td>1073</td></tr><tr><td>organization</td><td>organization-in-film</td><td>film</td><td>13128</td></tr><tr><td>organization</td><td>organization-on-location</td><td>location</td><td>2696</td></tr><tr><td>organization</td><td>organization-to-music</td><td>music</td><td>10702</td></tr><tr><td>organization</td><td>organization-to-sports</td><td>sports</td><td>559</td></tr><tr><td>organization</td><td>people-in-organization-inv</td><td>people</td><td>2215</td></tr><tr><td>people</td><td>people-and-people</td><td>people</td><td>22813</td></tr><tr><td>people</td><td>people-and-people-inv</td><td>people</td><td>22813</td></tr><tr><td>people</td><td>people-in-business</td><td>business</td><td>5378</td></tr><tr><td>people</td><td>people-in-organization</td><td>organization</td><td>2215</td></tr><tr><td>people</td><td>people-on-location</td><td>location</td><td>15134</td></tr><tr><td>people</td><td>people-to-book</td><td>book</td><td>35587</td></tr><tr><td>people</td><td>people-to-film</td><td>film</td><td>17604</td></tr><tr><td>people</td><td>people-to-music</td><td>music</td><td>10948</td></tr><tr><td>people</td><td>people-to-sports</td><td>sports</td><td>14850</td></tr><tr><td>sports</td><td>book-on-sports-inv</td><td>book</td><td>6615</td></tr><tr><td>sports</td><td>business-about-sports-inv</td><td>business</td><td>610</td></tr><tr><td>sports</td><td>music-for-sports-inv</td><td>music</td><td>8975</td></tr><tr><td>sports</td><td>organization-to-sports-inv</td><td>organization</td><td>559</td></tr><tr><td>sports</td><td>people-to-sports-inv</td><td>people</td><td>14850</td></tr><tr><td>sports</td><td>sports-and-sports</td><td>sports</td><td>1290</td></tr><tr><td>sports</td><td>sports-and-sports-inv</td><td>sports</td><td>1290</td></tr><tr><td>sports</td><td>sports-in-film</td><td>film</td><td>6763</td></tr><tr><td>sports</td><td>sports-on-location</td><td>location</td><td>656</td></tr></tbody></table></details> |
| `freebase` | `srgcn` | <details><summary>Click to expand (nodes)</summary><table><thead><tr><th>ntype</th><th>#samples</th><th>feat_dim</th></tr></thead><tbody><tr><td>book</td><td>40402</td><td>133662</td></tr><tr><td>business</td><td>7153</td><td>133662</td></tr><tr><td>film</td><td>19427</td><td>133662</td></tr><tr><td>location</td><td>9368</td><td>133662</td></tr><tr><td>music</td><td>82351</td><td>133662</td></tr><tr><td>organization</td><td>2731</td><td>133662</td></tr><tr><td>people</td><td>17641</td><td>133662</td></tr><tr><td>sports</td><td>1025</td><td>133662</td></tr></tbody></table></details> | <details><summary>Click to expand (edges)</summary><table><thead><tr><th>srctype</th><th>etype</th><th>dsttype</th><th>#edges</th></tr></thead><tbody><tr><td>book</td><td>book-about-organization</td><td>organization</td><td>21900</td></tr><tr><td>book</td><td>book-and-book</td><td>book</td><td>202674</td></tr><tr><td>book</td><td>book-and-book-inv</td><td>book</td><td>202674</td></tr><tr><td>book</td><td>book-on-location</td><td>location</td><td>26921</td></tr><tr><td>book</td><td>book-on-sports</td><td>sports</td><td>6615</td></tr><tr><td>book</td><td>business-about-book-inv</td><td>business</td><td>18625</td></tr><tr><td>book</td><td>people-to-book-inv</td><td>people</td><td>35587</td></tr><tr><td>business</td><td>business-about-book</td><td>book</td><td>18625</td></tr><tr><td>business</td><td>business-about-film</td><td>film</td><td>8397</td></tr><tr><td>business</td><td>business-about-music</td><td>music</td><td>24764</td></tr><tr><td>business</td><td>business-about-sports</td><td>sports</td><td>610</td></tr><tr><td>business</td><td>business-and-business</td><td>business</td><td>4448</td></tr><tr><td>business</td><td>business-and-business-inv</td><td>business</td><td>4448</td></tr><tr><td>film</td><td>business-about-film-inv</td><td>business</td><td>8397</td></tr><tr><td>film</td><td>film-and-film</td><td>film</td><td>87838</td></tr><tr><td>film</td><td>film-and-film-inv</td><td>film</td><td>87838</td></tr><tr><td>film</td><td>location-in-film-inv</td><td>location</td><td>21299</td></tr><tr><td>film</td><td>people-to-film-inv</td><td>people</td><td>17604</td></tr><tr><td>film</td><td>sports-in-film-inv</td><td>sports</td><td>6763</td></tr><tr><td>location</td><td>book-on-location-inv</td><td>book</td><td>26921</td></tr><tr><td>location</td><td>location-and-location</td><td>location</td><td>47817</td></tr><tr><td>location</td><td>location-and-location-inv</td><td>location</td><td>47817</td></tr><tr><td>location</td><td>location-in-film</td><td>film</td><td>21299</td></tr><tr><td>location</td><td>music-on-location-inv</td><td>music</td><td>42915</td></tr><tr><td>location</td><td>organization-on-location-inv</td><td>organization</td><td>2696</td></tr><tr><td>location</td><td>people-on-location-inv</td><td>people</td><td>15134</td></tr><tr><td>location</td><td>sports-on-location-inv</td><td>sports</td><td>656</td></tr><tr><td>music</td><td>business-about-music-inv</td><td>business</td><td>24764</td></tr><tr><td>music</td><td>music-and-music</td><td>music</td><td>283670</td></tr><tr><td>music</td><td>music-and-music-inv</td><td>music</td><td>283670</td></tr><tr><td>music</td><td>music-for-sports</td><td>sports</td><td>8975</td></tr><tr><td>music</td><td>music-on-location</td><td>location</td><td>42915</td></tr><tr><td>music</td><td>organization-to-music-inv</td><td>organization</td><td>10702</td></tr><tr><td>organization</td><td>book-about-organization-inv</td><td>book</td><td>21900</td></tr><tr><td>organization</td><td>organization-on-location</td><td>location</td><td>2696</td></tr><tr><td>organization</td><td>organization-to-music</td><td>music</td><td>10702</td></tr><tr><td>people</td><td>people-and-people</td><td>people</td><td>22813</td></tr><tr><td>people</td><td>people-and-people-inv</td><td>people</td><td>22813</td></tr><tr><td>people</td><td>people-on-location</td><td>location</td><td>15134</td></tr><tr><td>people</td><td>people-to-book</td><td>book</td><td>35587</td></tr><tr><td>people</td><td>people-to-film</td><td>film</td><td>17604</td></tr><tr><td>sports</td><td>book-on-sports-inv</td><td>book</td><td>6615</td></tr><tr><td>sports</td><td>business-about-sports-inv</td><td>business</td><td>610</td></tr><tr><td>sports</td><td>music-for-sports-inv</td><td>music</td><td>8975</td></tr><tr><td>sports</td><td>sports-and-sports</td><td>sports</td><td>1290</td></tr><tr><td>sports</td><td>sports-and-sports-inv</td><td>sports</td><td>1290</td></tr><tr><td>sports</td><td>sports-in-film</td><td>film</td><td>6763</td></tr><tr><td>sports</td><td>sports-on-location</td><td>location</td><td>656</td></tr></tbody></table></details> |
| `atomic-dblp` | `atomic` | <details><summary>Click to expand (nodes)</summary><table><thead><tr><th>ntype</th><th>#samples</th><th>feat_dim</th></tr></thead><tbody><tr><td>author</td><td>4057</td><td>0</td></tr><tr><td>authorfeat</td><td>334</td><td>0</td></tr><tr><td>conference</td><td>20</td><td>0</td></tr><tr><td>numerical</td><td>50</td><td>0</td></tr><tr><td>paper</td><td>14328</td><td>0</td></tr><tr><td>paperfeat</td><td>4231</td><td>0</td></tr><tr><td>term</td><td>7723</td><td>0</td></tr></tbody></table></details> | <details><summary>Click to expand (edges)</summary><table><thead><tr><th>srctype</th><th>etype</th><th>dsttype</th><th>#edges</th></tr></thead><tbody><tr><td>author</td><td>has-authorfeat</td><td>authorfeat</td><td>48810</td></tr><tr><td>author</td><td>writing</td><td>paper</td><td>19645</td></tr><tr><td>authorfeat</td><td>is-authorfeat-of</td><td>author</td><td>48810</td></tr><tr><td>conference</td><td>has</td><td>paper</td><td>14328</td></tr><tr><td>numerical</td><td>is-numerical-of</td><td>term</td><td>386150</td></tr><tr><td>paper</td><td>contains</td><td>term</td><td>85810</td></tr><tr><td>paper</td><td>has-paperfeat</td><td>paperfeat</td><td>95030</td></tr><tr><td>paper</td><td>pubs-in</td><td>conference</td><td>14328</td></tr><tr><td>paper</td><td>written-by</td><td>author</td><td>19645</td></tr><tr><td>paperfeat</td><td>is-paperfeat-of</td><td>paper</td><td>95030</td></tr><tr><td>term</td><td>has-numerical</td><td>numerical</td><td>386150</td></tr><tr><td>term</td><td>is-in</td><td>paper</td><td>85810</td></tr></tbody></table></details> |
| `atomic-dblp` | `srgcn` | <details><summary>Click to expand (nodes)</summary><table><thead><tr><th>ntype</th><th>#samples</th><th>feat_dim</th></tr></thead><tbody><tr><td>author</td><td>4057</td><td>4251</td></tr><tr><td>conference</td><td>20</td><td>4251</td></tr><tr><td>paper</td><td>14328</td><td>4251</td></tr><tr><td>paperfeat</td><td>4231</td><td>4251</td></tr><tr><td>term</td><td>7723</td><td>4251</td></tr></tbody></table></details> | <details><summary>Click to expand (edges)</summary><table><thead><tr><th>srctype</th><th>etype</th><th>dsttype</th><th>#edges</th></tr></thead><tbody><tr><td>author</td><td>writing</td><td>paper</td><td>19645</td></tr><tr><td>conference</td><td>has</td><td>paper</td><td>14328</td></tr><tr><td>paper</td><td>contains</td><td>term</td><td>85810</td></tr><tr><td>paper</td><td>has-paperfeat</td><td>paperfeat</td><td>95030</td></tr><tr><td>paper</td><td>pubs-in</td><td>conference</td><td>14328</td></tr><tr><td>paper</td><td>written-by</td><td>author</td><td>19645</td></tr><tr><td>paperfeat</td><td>is-paperfeat-of</td><td>paper</td><td>95030</td></tr><tr><td>term</td><td>is-in</td><td>paper</td><td>85810</td></tr></tbody></table></details> |
| `acm` | `atomic` | <details><summary>Click to expand (nodes)</summary><table><thead><tr><th>ntype</th><th>#samples</th><th>feat_dim</th></tr></thead><tbody><tr><td>author</td><td>5959</td><td>0</td></tr><tr><td>paper</td><td>3025</td><td>0</td></tr><tr><td>subject</td><td>56</td><td>0</td></tr><tr><td>term</td><td>1902</td><td>0</td></tr></tbody></table></details> | <details><summary>Click to expand (edges)</summary><table><thead><tr><th>srctype</th><th>etype</th><th>dsttype</th><th>#edges</th></tr></thead><tbody><tr><td>author</td><td>writing</td><td>paper</td><td>9949</td></tr><tr><td>paper</td><td>cited</td><td>paper</td><td>10628</td></tr><tr><td>paper</td><td>citing</td><td>paper</td><td>10628</td></tr><tr><td>paper</td><td>contains</td><td>term</td><td>255619</td></tr><tr><td>paper</td><td>is-about</td><td>subject</td><td>3025</td></tr><tr><td>paper</td><td>written-by</td><td>author</td><td>9949</td></tr><tr><td>subject</td><td>has</td><td>paper</td><td>3025</td></tr><tr><td>term</td><td>is-in</td><td>paper</td><td>255619</td></tr></tbody></table></details> |
| `acm` | `srgcn` | <details><summary>Click to expand (nodes)</summary><table><thead><tr><th>ntype</th><th>#samples</th><th>feat_dim</th></tr></thead><tbody><tr><td>author</td><td>5959</td><td>1902</td></tr><tr><td>paper</td><td>3025</td><td>1902</td></tr><tr><td>subject</td><td>56</td><td>1902</td></tr><tr><td>term</td><td>1902</td><td>1902</td></tr></tbody></table></details> | <details><summary>Click to expand (edges)</summary><table><thead><tr><th>srctype</th><th>etype</th><th>dsttype</th><th>#edges</th></tr></thead><tbody><tr><td>author</td><td>writing</td><td>paper</td><td>9949</td></tr><tr><td>paper</td><td>cited</td><td>paper</td><td>10628</td></tr><tr><td>paper</td><td>citing</td><td>paper</td><td>10628</td></tr><tr><td>paper</td><td>contains</td><td>term</td><td>255619</td></tr><tr><td>paper</td><td>is-about</td><td>subject</td><td>3025</td></tr><tr><td>paper</td><td>written-by</td><td>author</td><td>9949</td></tr><tr><td>subject</td><td>has</td><td>paper</td><td>3025</td></tr><tr><td>term</td><td>is-in</td><td>paper</td><td>255619</td></tr></tbody></table></details> |
| `atomic-mag` | `atomic` | <details><summary>Click to expand (nodes)</summary><table><thead><tr><th>ntype</th><th>#samples</th><th>feat_dim</th></tr></thead><tbody><tr><td>author</td><td>1134649</td><td>0</td></tr><tr><td>field_of_study</td><td>59965</td><td>0</td></tr><tr><td>institution</td><td>8740</td><td>0</td></tr><tr><td>numerical</td><td>128</td><td>0</td></tr><tr><td>paper</td><td>736389</td><td>0</td></tr><tr><td>year</td><td>8</td><td>0</td></tr></tbody></table></details> | <details><summary>Click to expand (edges)</summary><table><thead><tr><th>srctype</th><th>etype</th><th>dsttype</th><th>#edges</th></tr></thead><tbody><tr><td>author</td><td>affiliated_with</td><td>institution</td><td>1043998</td></tr><tr><td>author</td><td>writes</td><td>paper</td><td>7145660</td></tr><tr><td>field_of_study</td><td>contains</td><td>paper</td><td>7505078</td></tr><tr><td>institution</td><td>affiliates</td><td>author</td><td>1043998</td></tr><tr><td>numerical</td><td>is-numerical-of</td><td>paper</td><td>94257583</td></tr><tr><td>paper</td><td>cites</td><td>paper</td><td>10792672</td></tr><tr><td>paper</td><td>has-numerical</td><td>numerical</td><td>94257583</td></tr><tr><td>paper</td><td>has_topic</td><td>field_of_study</td><td>7505078</td></tr><tr><td>paper</td><td>published-in-year</td><td>year</td><td>629571</td></tr><tr><td>paper</td><td>written_by</td><td>author</td><td>7145660</td></tr><tr><td>year</td><td>year-of-publication</td><td>paper</td><td>629571</td></tr></tbody></table></details> |
| `atomic-mag` | `srgcn` | <details><summary>Click to expand (nodes)</summary><table><thead><tr><th>ntype</th><th>#samples</th><th>feat_dim</th></tr></thead><tbody><tr><td>author</td><td>1134649</td><td>392</td></tr><tr><td>field_of_study</td><td>59965</td><td>392</td></tr><tr><td>institution</td><td>8740</td><td>392</td></tr><tr><td>numerical</td><td>128</td><td>392</td></tr><tr><td>paper</td><td>736389</td><td>392</td></tr></tbody></table></details> | <details><summary>Click to expand (edges)</summary><table><thead><tr><th>srctype</th><th>etype</th><th>dsttype</th><th>#edges</th></tr></thead><tbody><tr><td>author</td><td>affiliated_with</td><td>institution</td><td>1043998</td></tr><tr><td>author</td><td>writes</td><td>paper</td><td>7145660</td></tr><tr><td>field_of_study</td><td>contains</td><td>paper</td><td>7505078</td></tr><tr><td>institution</td><td>affiliates</td><td>author</td><td>1043998</td></tr><tr><td>numerical</td><td>is-numerical-of</td><td>paper</td><td>94257583</td></tr><tr><td>paper</td><td>cites</td><td>paper</td><td>10792672</td></tr><tr><td>paper</td><td>has-numerical</td><td>numerical</td><td>94257583</td></tr><tr><td>paper</td><td>has_topic</td><td>field_of_study</td><td>7505078</td></tr><tr><td>paper</td><td>written_by</td><td>author</td><td>7145660</td></tr></tbody></table></details> |
| `atomic-amazon` | `atomic` | <details><summary>Click to expand (nodes)</summary><table><thead><tr><th>ntype</th><th>#samples</th><th>feat_dim</th></tr></thead><tbody><tr><td>brand</td><td>3</td><td>0</td></tr><tr><td>category</td><td>342</td><td>0</td></tr><tr><td>price</td><td>2</td><td>0</td></tr><tr><td>product</td><td>10099</td><td>0</td></tr><tr><td>sales_rank</td><td>810</td><td>0</td></tr></tbody></table></details> | <details><summary>Click to expand (edges)</summary><table><thead><tr><th>srctype</th><th>etype</th><th>dsttype</th><th>#edges</th></tr></thead><tbody><tr><td>brand</td><td>brand-product</td><td>product</td><td>10099</td></tr><tr><td>category</td><td>category-product</td><td>product</td><td>10099</td></tr><tr><td>price</td><td>price-product</td><td>product</td><td>20198</td></tr><tr><td>product</td><td>co-purchase</td><td>product</td><td>58517</td></tr><tr><td>product</td><td>co-purchase-inv</td><td>product</td><td>58517</td></tr><tr><td>product</td><td>co-view</td><td>product</td><td>62841</td></tr><tr><td>product</td><td>co-view-inv</td><td>product</td><td>62841</td></tr><tr><td>product</td><td>product-brand</td><td>brand</td><td>10099</td></tr><tr><td>product</td><td>product-category</td><td>category</td><td>10099</td></tr><tr><td>product</td><td>product-price</td><td>price</td><td>20198</td></tr><tr><td>product</td><td>product-sales_rank</td><td>sales_rank</td><td>10099</td></tr><tr><td>sales_rank</td><td>sales_rank-product</td><td>product</td><td>10099</td></tr></tbody></table></details> |
| `atomic-amazon` | `srgcn` | <details><summary>Click to expand (nodes)</summary><table><thead><tr><th>ntype</th><th>#samples</th><th>feat_dim</th></tr></thead><tbody><tr><td>brand</td><td>3</td><td>10099</td></tr><tr><td>category</td><td>342</td><td>10099</td></tr><tr><td>price</td><td>2</td><td>10099</td></tr><tr><td>product</td><td>10099</td><td>10099</td></tr><tr><td>sales_rank</td><td>810</td><td>10099</td></tr></tbody></table></details> | <details><summary>Click to expand (edges)</summary><table><thead><tr><th>srctype</th><th>etype</th><th>dsttype</th><th>#edges</th></tr></thead><tbody><tr><td>price</td><td>price-product</td><td>product</td><td>20198</td></tr><tr><td>product</td><td>co-purchase</td><td>product</td><td>58517</td></tr><tr><td>product</td><td>co-purchase-inv</td><td>product</td><td>58517</td></tr><tr><td>product</td><td>co-view</td><td>product</td><td>62841</td></tr><tr><td>product</td><td>co-view-inv</td><td>product</td><td>62841</td></tr><tr><td>product</td><td>product-price</td><td>price</td><td>20198</td></tr><tr><td>product</td><td>product-sales_rank</td><td>sales_rank</td><td>10099</td></tr><tr><td>sales_rank</td><td>sales_rank-product</td><td>product</td><td>10099</td></tr></tbody></table></details> |
| `lastfm` | `atomic` | <details><summary>Click to expand (nodes)</summary><table><thead><tr><th>ntype</th><th>#samples</th><th>feat_dim</th></tr></thead><tbody><tr><td>artist</td><td>17632</td><td>0</td></tr><tr><td>tag</td><td>1088</td><td>0</td></tr><tr><td>user</td><td>1892</td><td>0</td></tr></tbody></table></details> | <details><summary>Click to expand (edges)</summary><table><thead><tr><th>srctype</th><th>etype</th><th>dsttype</th><th>#edges</th></tr></thead><tbody><tr><td>artist</td><td>artist-tag</td><td>tag</td><td>23253</td></tr><tr><td>artist</td><td>artist-user</td><td>user</td><td>66964</td></tr><tr><td>tag</td><td>tag-artist</td><td>artist</td><td>23253</td></tr><tr><td>user</td><td>user-artist</td><td>artist</td><td>66964</td></tr><tr><td>user</td><td>user-user</td><td>user</td><td>25434</td></tr></tbody></table></details> |
| `lastfm` | `srgcn` | <details><summary>Click to expand (nodes)</summary><table><thead><tr><th>ntype</th><th>#samples</th><th>feat_dim</th></tr></thead><tbody><tr><td>artist</td><td>17632</td><td>1088</td></tr><tr><td>tag</td><td>1088</td><td>1088</td></tr><tr><td>user</td><td>1892</td><td>1088</td></tr></tbody></table></details> | <details><summary>Click to expand (edges)</summary><table><thead><tr><th>srctype</th><th>etype</th><th>dsttype</th><th>#edges</th></tr></thead><tbody><tr><td>artist</td><td>artist-tag</td><td>tag</td><td>23253</td></tr><tr><td>tag</td><td>tag-artist</td><td>artist</td><td>23253</td></tr></tbody></table></details> |
| `atomic-pubmed` | `atomic` | <details><summary>Click to expand (nodes)</summary><table><thead><tr><th>ntype</th><th>#samples</th><th>feat_dim</th></tr></thead><tbody><tr><td>chemical</td><td>26522</td><td>0</td></tr><tr><td>chemical_feat</td><td>200</td><td>0</td></tr><tr><td>disease</td><td>20163</td><td>0</td></tr><tr><td>disease_feat</td><td>200</td><td>0</td></tr><tr><td>gene</td><td>2863</td><td>0</td></tr><tr><td>gene_feat</td><td>200</td><td>0</td></tr><tr><td>species</td><td>13561</td><td>0</td></tr><tr><td>species_feat</td><td>200</td><td>0</td></tr></tbody></table></details> | <details><summary>Click to expand (edges)</summary><table><thead><tr><th>srctype</th><th>etype</th><th>dsttype</th><th>#edges</th></tr></thead><tbody><tr><td>chemical</td><td>chemical-chemical</td><td>chemical</td><td>62187</td></tr><tr><td>chemical</td><td>chemical-chemical-inv</td><td>chemical</td><td>62187</td></tr><tr><td>chemical</td><td>chemical-disease</td><td>disease</td><td>51323</td></tr><tr><td>chemical</td><td>chemical-gene</td><td>gene</td><td>6297</td></tr><tr><td>chemical</td><td>chemical-has-feat</td><td>chemical_feat</td><td>5304390</td></tr><tr><td>chemical</td><td>chemical-species</td><td>species</td><td>31277</td></tr><tr><td>chemical_feat</td><td>feat-of-chemical</td><td>chemical</td><td>5304390</td></tr><tr><td>disease</td><td>disease-chemical</td><td>chemical</td><td>51323</td></tr><tr><td>disease</td><td>disease-disease</td><td>disease</td><td>30698</td></tr><tr><td>disease</td><td>disease-disease-inv</td><td>disease</td><td>30698</td></tr><tr><td>disease</td><td>disease-gene</td><td>gene</td><td>5245</td></tr><tr><td>disease</td><td>disease-has-feat</td><td>disease_feat</td><td>4032587</td></tr><tr><td>disease</td><td>disease-species</td><td>species</td><td>25962</td></tr><tr><td>disease_feat</td><td>feat-of-disease</td><td>disease</td><td>4032587</td></tr><tr><td>gene</td><td>gene-chemical</td><td>chemical</td><td>6297</td></tr><tr><td>gene</td><td>gene-disease</td><td>disease</td><td>5245</td></tr><tr><td>gene</td><td>gene-gene</td><td>gene</td><td>798</td></tr><tr><td>gene</td><td>gene-gene-inv</td><td>gene</td><td>798</td></tr><tr><td>gene</td><td>gene-has-feat</td><td>gene_feat</td><td>572599</td></tr><tr><td>gene</td><td>gene-species</td><td>species</td><td>3155</td></tr><tr><td>gene_feat</td><td>feat-of-gene</td><td>gene</td><td>572599</td></tr><tr><td>species</td><td>species-chemical</td><td>chemical</td><td>31277</td></tr><tr><td>species</td><td>species-disease</td><td>disease</td><td>25962</td></tr><tr><td>species</td><td>species-gene</td><td>gene</td><td>3155</td></tr><tr><td>species</td><td>species-has-feat</td><td>species_feat</td><td>2712190</td></tr><tr><td>species</td><td>species-species</td><td>species</td><td>16105</td></tr><tr><td>species</td><td>species-species-inv</td><td>species</td><td>16105</td></tr><tr><td>species_feat</td><td>feat-of-species</td><td>species</td><td>2712190</td></tr></tbody></table></details> |
| `atomic-pubmed` | `srgcn` | <details><summary>Click to expand (nodes)</summary><table><thead><tr><th>ntype</th><th>#samples</th><th>feat_dim</th></tr></thead><tbody><tr><td>chemical</td><td>26522</td><td>63309</td></tr><tr><td>chemical_feat</td><td>200</td><td>63309</td></tr><tr><td>disease</td><td>20163</td><td>63309</td></tr><tr><td>disease_feat</td><td>200</td><td>63309</td></tr><tr><td>gene</td><td>2863</td><td>63309</td></tr><tr><td>gene_feat</td><td>200</td><td>63309</td></tr><tr><td>species</td><td>13561</td><td>63309</td></tr><tr><td>species_feat</td><td>200</td><td>63309</td></tr></tbody></table></details> | <details><summary>Click to expand (edges)</summary><table><thead><tr><th>srctype</th><th>etype</th><th>dsttype</th><th>#edges</th></tr></thead><tbody><tr><td>chemical</td><td>chemical-chemical</td><td>chemical</td><td>62187</td></tr><tr><td>chemical</td><td>chemical-chemical-inv</td><td>chemical</td><td>62187</td></tr><tr><td>chemical</td><td>chemical-gene</td><td>gene</td><td>6297</td></tr><tr><td>chemical</td><td>chemical-has-feat</td><td>chemical_feat</td><td>5304390</td></tr><tr><td>chemical_feat</td><td>feat-of-chemical</td><td>chemical</td><td>5304390</td></tr><tr><td>disease</td><td>disease-gene</td><td>gene</td><td>5245</td></tr><tr><td>disease</td><td>disease-has-feat</td><td>disease_feat</td><td>4032587</td></tr><tr><td>disease_feat</td><td>feat-of-disease</td><td>disease</td><td>4032587</td></tr><tr><td>gene</td><td>gene-chemical</td><td>chemical</td><td>6297</td></tr><tr><td>gene</td><td>gene-disease</td><td>disease</td><td>5245</td></tr><tr><td>gene</td><td>gene-species</td><td>species</td><td>3155</td></tr><tr><td>species</td><td>species-gene</td><td>gene</td><td>3155</td></tr></tbody></table></details> |

Each entry in the table can be loaded directly via:

```python
hg = dhgl.get_dataset('atomic-imdb', profile='atomic')
hg = dhgl.get_dataset('atomic-imdb', profile='srgcn')
```

- NOTE: for OGBN-MAG, use have to install package from [ogb](https://ogb.stanford.edu)

---

## Schema Refinement (Atomic Datasets)

Atomic datasets support **schema refinement**, allowing users to select or unselect
node types and edge types at load time.

Refinement is specified via boolean keyword arguments:

* `True`: select (keep) the type
* `False`: unselect (drop) the type

```python
hg = dhgl.get_dataset(
    'atomic-imdb',
    word=True,
    **{
        'director': True,
        'is-in': False,
    }
)
```

This example keeps the node types `word` and `director`, while removing the edge
type `is-in` (and its inverse).
Schema refinement applies on top of the selected profile and simplifies the graph
structure for task-specific modeling.

Predefined refined schemas (e.g., searched under sRGCN) can be loaded via:

```python
hg = dhgl.get_dataset('atomic-imdb', profile='srgcn')
```

Here we illustrate simple schema refinement using the `dhgl.get_dataset` API.
For more sophisticated usage, please refer to the advanced tutorial: [demo_schema_refienment.ipynb](./scripts/demo_schema_refinement.ipynb).

---

## Vanilla Datasets

Vanilla datasets with fixed schemas are also provided:

```python
hg = dhgl.get_dataset('imdb')
```

Schema refinement is **not supported** for vanilla datasets.

> *Note:* Vanilla datasets are summarized in the table below.

| name | profile | node info | edge info |
|---|---|---|---|
| `imdb` | `vanilla` | <details><summary>Click to expand (nodes)</summary><table><thead><tr><th>ntype</th><th>#samples</th><th>feat_dim</th></tr></thead><tbody><tr><td>actor</td><td>6124</td><td>3341</td></tr><tr><td>director</td><td>2393</td><td>3341</td></tr><tr><td>keyword</td><td>7971</td><td>7971</td></tr><tr><td>movie</td><td>4932</td><td>3489</td></tr></tbody></table></details> | <details><summary>Click to expand (edges)</summary><table><thead><tr><th>srctype</th><th>etype</th><th>dsttype</th><th>#edges</th></tr></thead><tbody><tr><td>actor</td><td>acts</td><td>movie</td><td>14779</td></tr><tr><td>director</td><td>directed</td><td>movie</td><td>4932</td></tr><tr><td>keyword</td><td>is-in</td><td>movie</td><td>23610</td></tr><tr><td>movie</td><td>contains</td><td>keyword</td><td>23610</td></tr><tr><td>movie</td><td>directed-by</td><td>director</td><td>4932</td></tr><tr><td>movie</td><td>stars</td><td>actor</td><td>14779</td></tr></tbody></table></details> |
| `freebase` | `vanilla` | <details><summary>Click to expand (nodes)</summary><table><thead><tr><th>ntype</th><th>#samples</th><th>feat_dim</th></tr></thead><tbody><tr><td>book</td><td>40402</td><td>40402</td></tr><tr><td>business</td><td>7153</td><td>7153</td></tr><tr><td>film</td><td>19427</td><td>19427</td></tr><tr><td>location</td><td>9368</td><td>9368</td></tr><tr><td>music</td><td>82351</td><td>82351</td></tr><tr><td>organization</td><td>2731</td><td>2731</td></tr><tr><td>people</td><td>17641</td><td>17641</td></tr><tr><td>sports</td><td>1025</td><td>1025</td></tr></tbody></table></details> | <details><summary>Click to expand (edges)</summary><table><thead><tr><th>srctype</th><th>etype</th><th>dsttype</th><th>#edges</th></tr></thead><tbody><tr><td>book</td><td>book-about-organization</td><td>organization</td><td>21900</td></tr><tr><td>book</td><td>book-and-book</td><td>book</td><td>202674</td></tr><tr><td>book</td><td>book-and-book-inv</td><td>book</td><td>202674</td></tr><tr><td>book</td><td>book-on-location</td><td>location</td><td>26921</td></tr><tr><td>book</td><td>book-on-sports</td><td>sports</td><td>6615</td></tr><tr><td>book</td><td>book-to-film</td><td>film</td><td>38299</td></tr><tr><td>book</td><td>business-about-book-inv</td><td>business</td><td>18625</td></tr><tr><td>book</td><td>music-in-book-inv</td><td>music</td><td>31486</td></tr><tr><td>book</td><td>people-to-book-inv</td><td>people</td><td>35587</td></tr><tr><td>business</td><td>business-about-book</td><td>book</td><td>18625</td></tr><tr><td>business</td><td>business-about-film</td><td>film</td><td>8397</td></tr><tr><td>business</td><td>business-about-music</td><td>music</td><td>24764</td></tr><tr><td>business</td><td>business-about-sports</td><td>sports</td><td>610</td></tr><tr><td>business</td><td>business-and-business</td><td>business</td><td>4448</td></tr><tr><td>business</td><td>business-and-business-inv</td><td>business</td><td>4448</td></tr><tr><td>business</td><td>business-on-location</td><td>location</td><td>6647</td></tr><tr><td>business</td><td>organization-for-business-inv</td><td>organization</td><td>1073</td></tr><tr><td>business</td><td>people-in-business-inv</td><td>people</td><td>5378</td></tr><tr><td>film</td><td>book-to-film-inv</td><td>book</td><td>38299</td></tr><tr><td>film</td><td>business-about-film-inv</td><td>business</td><td>8397</td></tr><tr><td>film</td><td>film-and-film</td><td>film</td><td>87838</td></tr><tr><td>film</td><td>film-and-film-inv</td><td>film</td><td>87838</td></tr><tr><td>film</td><td>location-in-film-inv</td><td>location</td><td>21299</td></tr><tr><td>film</td><td>music-in-film-inv</td><td>music</td><td>11291</td></tr><tr><td>film</td><td>organization-in-film-inv</td><td>organization</td><td>13128</td></tr><tr><td>film</td><td>people-to-film-inv</td><td>people</td><td>17604</td></tr><tr><td>film</td><td>sports-in-film-inv</td><td>sports</td><td>6763</td></tr><tr><td>location</td><td>book-on-location-inv</td><td>book</td><td>26921</td></tr><tr><td>location</td><td>business-on-location-inv</td><td>business</td><td>6647</td></tr><tr><td>location</td><td>location-and-location</td><td>location</td><td>47817</td></tr><tr><td>location</td><td>location-and-location-inv</td><td>location</td><td>47817</td></tr><tr><td>location</td><td>location-in-film</td><td>film</td><td>21299</td></tr><tr><td>location</td><td>music-on-location-inv</td><td>music</td><td>42915</td></tr><tr><td>location</td><td>organization-on-location-inv</td><td>organization</td><td>2696</td></tr><tr><td>location</td><td>people-on-location-inv</td><td>people</td><td>15134</td></tr><tr><td>location</td><td>sports-on-location-inv</td><td>sports</td><td>656</td></tr><tr><td>music</td><td>business-about-music-inv</td><td>business</td><td>24764</td></tr><tr><td>music</td><td>music-and-music</td><td>music</td><td>283670</td></tr><tr><td>music</td><td>music-and-music-inv</td><td>music</td><td>283670</td></tr><tr><td>music</td><td>music-for-sports</td><td>sports</td><td>8975</td></tr><tr><td>music</td><td>music-in-book</td><td>book</td><td>31486</td></tr><tr><td>music</td><td>music-in-film</td><td>film</td><td>11291</td></tr><tr><td>music</td><td>music-on-location</td><td>location</td><td>42915</td></tr><tr><td>music</td><td>organization-to-music-inv</td><td>organization</td><td>10702</td></tr><tr><td>music</td><td>people-to-music-inv</td><td>people</td><td>10948</td></tr><tr><td>organization</td><td>book-about-organization-inv</td><td>book</td><td>21900</td></tr><tr><td>organization</td><td>organization-and-organization</td><td>organization</td><td>1101</td></tr><tr><td>organization</td><td>organization-and-organization-inv</td><td>organization</td><td>1101</td></tr><tr><td>organization</td><td>organization-for-business</td><td>business</td><td>1073</td></tr><tr><td>organization</td><td>organization-in-film</td><td>film</td><td>13128</td></tr><tr><td>organization</td><td>organization-on-location</td><td>location</td><td>2696</td></tr><tr><td>organization</td><td>organization-to-music</td><td>music</td><td>10702</td></tr><tr><td>organization</td><td>organization-to-sports</td><td>sports</td><td>559</td></tr><tr><td>organization</td><td>people-in-organization-inv</td><td>people</td><td>2215</td></tr><tr><td>people</td><td>people-and-people</td><td>people</td><td>22813</td></tr><tr><td>people</td><td>people-and-people-inv</td><td>people</td><td>22813</td></tr><tr><td>people</td><td>people-in-business</td><td>business</td><td>5378</td></tr><tr><td>people</td><td>people-in-organization</td><td>organization</td><td>2215</td></tr><tr><td>people</td><td>people-on-location</td><td>location</td><td>15134</td></tr><tr><td>people</td><td>people-to-book</td><td>book</td><td>35587</td></tr><tr><td>people</td><td>people-to-film</td><td>film</td><td>17604</td></tr><tr><td>people</td><td>people-to-music</td><td>music</td><td>10948</td></tr><tr><td>people</td><td>people-to-sports</td><td>sports</td><td>14850</td></tr><tr><td>sports</td><td>book-on-sports-inv</td><td>book</td><td>6615</td></tr><tr><td>sports</td><td>business-about-sports-inv</td><td>business</td><td>610</td></tr><tr><td>sports</td><td>music-for-sports-inv</td><td>music</td><td>8975</td></tr><tr><td>sports</td><td>organization-to-sports-inv</td><td>organization</td><td>559</td></tr><tr><td>sports</td><td>people-to-sports-inv</td><td>people</td><td>14850</td></tr><tr><td>sports</td><td>sports-and-sports</td><td>sports</td><td>1290</td></tr><tr><td>sports</td><td>sports-and-sports-inv</td><td>sports</td><td>1290</td></tr><tr><td>sports</td><td>sports-in-film</td><td>film</td><td>6763</td></tr><tr><td>sports</td><td>sports-on-location</td><td>location</td><td>656</td></tr></tbody></table></details> |
| `dblp` | `vanilla` | <details><summary>Click to expand (nodes)</summary><table><thead><tr><th>ntype</th><th>#samples</th><th>feat_dim</th></tr></thead><tbody><tr><td>author</td><td>4057</td><td>334</td></tr><tr><td>conference</td><td>20</td><td>20</td></tr><tr><td>paper</td><td>14328</td><td>4231</td></tr><tr><td>term</td><td>7723</td><td>50</td></tr></tbody></table></details> | <details><summary>Click to expand (edges)</summary><table><thead><tr><th>srctype</th><th>etype</th><th>dsttype</th><th>#edges</th></tr></thead><tbody><tr><td>author</td><td>writing</td><td>paper</td><td>19645</td></tr><tr><td>conference</td><td>has</td><td>paper</td><td>14328</td></tr><tr><td>paper</td><td>contains</td><td>term</td><td>85810</td></tr><tr><td>paper</td><td>pubs-in</td><td>conference</td><td>14328</td></tr><tr><td>paper</td><td>written-by</td><td>author</td><td>19645</td></tr><tr><td>term</td><td>is-in</td><td>paper</td><td>85810</td></tr></tbody></table></details> |
| `acm` | `vanilla` | <details><summary>Click to expand (nodes)</summary><table><thead><tr><th>ntype</th><th>#samples</th><th>feat_dim</th></tr></thead><tbody><tr><td>author</td><td>5959</td><td>1902</td></tr><tr><td>paper</td><td>3025</td><td>1902</td></tr><tr><td>subject</td><td>56</td><td>1902</td></tr><tr><td>term</td><td>1902</td><td>1902</td></tr></tbody></table></details> | <details><summary>Click to expand (edges)</summary><table><thead><tr><th>srctype</th><th>etype</th><th>dsttype</th><th>#edges</th></tr></thead><tbody><tr><td>author</td><td>writing</td><td>paper</td><td>9949</td></tr><tr><td>paper</td><td>cited</td><td>paper</td><td>10628</td></tr><tr><td>paper</td><td>citing</td><td>paper</td><td>10628</td></tr><tr><td>paper</td><td>contains</td><td>term</td><td>255619</td></tr><tr><td>paper</td><td>is-about</td><td>subject</td><td>3025</td></tr><tr><td>paper</td><td>written-by</td><td>author</td><td>9949</td></tr><tr><td>subject</td><td>has</td><td>paper</td><td>3025</td></tr><tr><td>term</td><td>is-in</td><td>paper</td><td>255619</td></tr></tbody></table></details> |
| `amazon` | `vanilla` | <details><summary>Click to expand (nodes)</summary><table><thead><tr><th>ntype</th><th>#samples</th><th>feat_dim</th></tr></thead><tbody><tr><td>dummy</td><td>1</td><td>1</td></tr><tr><td>product</td><td>10099</td><td>1156</td></tr></tbody></table></details> | <details><summary>Click to expand (edges)</summary><table><thead><tr><th>srctype</th><th>etype</th><th>dsttype</th><th>#edges</th></tr></thead><tbody><tr><td>product</td><td>co-purchase</td><td>product</td><td>58517</td></tr><tr><td>product</td><td>co-purchase-inv</td><td>product</td><td>58517</td></tr><tr><td>product</td><td>co-view</td><td>product</td><td>62841</td></tr><tr><td>product</td><td>co-view-inv</td><td>product</td><td>62841</td></tr></tbody></table></details> |
| `lastfm` | `vanilla` | <details><summary>Click to expand (nodes)</summary><table><thead><tr><th>ntype</th><th>#samples</th><th>feat_dim</th></tr></thead><tbody><tr><td>artist</td><td>17632</td><td>17632</td></tr><tr><td>tag</td><td>1088</td><td>1088</td></tr><tr><td>user</td><td>1892</td><td>1892</td></tr></tbody></table></details> | <details><summary>Click to expand (edges)</summary><table><thead><tr><th>srctype</th><th>etype</th><th>dsttype</th><th>#edges</th></tr></thead><tbody><tr><td>artist</td><td>artist-tag</td><td>tag</td><td>23253</td></tr><tr><td>artist</td><td>artist-user</td><td>user</td><td>66964</td></tr><tr><td>tag</td><td>tag-artist</td><td>artist</td><td>23253</td></tr><tr><td>user</td><td>user-artist</td><td>artist</td><td>66964</td></tr><tr><td>user</td><td>user-user</td><td>user</td><td>25434</td></tr></tbody></table></details> |
| `pubmed` | `vanilla` | <details><summary>Click to expand (nodes)</summary><table><thead><tr><th>ntype</th><th>#samples</th><th>feat_dim</th></tr></thead><tbody><tr><td>chemical</td><td>26522</td><td>200</td></tr><tr><td>disease</td><td>20163</td><td>200</td></tr><tr><td>gene</td><td>2863</td><td>200</td></tr><tr><td>species</td><td>13561</td><td>200</td></tr></tbody></table></details> | <details><summary>Click to expand (edges)</summary><table><thead><tr><th>srctype</th><th>etype</th><th>dsttype</th><th>#edges</th></tr></thead><tbody><tr><td>chemical</td><td>chemical-chemical</td><td>chemical</td><td>62187</td></tr><tr><td>chemical</td><td>chemical-chemical-inv</td><td>chemical</td><td>62187</td></tr><tr><td>chemical</td><td>chemical-disease</td><td>disease</td><td>51323</td></tr><tr><td>chemical</td><td>chemical-gene</td><td>gene</td><td>6297</td></tr><tr><td>chemical</td><td>chemical-species</td><td>species</td><td>31277</td></tr><tr><td>disease</td><td>disease-chemical</td><td>chemical</td><td>51323</td></tr><tr><td>disease</td><td>disease-disease</td><td>disease</td><td>30698</td></tr><tr><td>disease</td><td>disease-disease-inv</td><td>disease</td><td>30698</td></tr><tr><td>disease</td><td>disease-gene</td><td>gene</td><td>5245</td></tr><tr><td>disease</td><td>disease-species</td><td>species</td><td>25962</td></tr><tr><td>gene</td><td>gene-chemical</td><td>chemical</td><td>6297</td></tr><tr><td>gene</td><td>gene-disease</td><td>disease</td><td>5245</td></tr><tr><td>gene</td><td>gene-gene</td><td>gene</td><td>798</td></tr><tr><td>gene</td><td>gene-gene-inv</td><td>gene</td><td>798</td></tr><tr><td>gene</td><td>gene-species</td><td>species</td><td>3155</td></tr><tr><td>species</td><td>species-chemical</td><td>chemical</td><td>31277</td></tr><tr><td>species</td><td>species-disease</td><td>disease</td><td>25962</td></tr><tr><td>species</td><td>species-gene</td><td>gene</td><td>3155</td></tr><tr><td>species</td><td>species-species</td><td>species</td><td>16105</td></tr><tr><td>species</td><td>species-species-inv</td><td>species</td><td>16105</td></tr></tbody></table></details> |

---

## Training, Evaluation, and Schema Search

All experiments are configured via `.env` files, which fully specify the dataset,
schema profile, model, and training settings.

### Setup

Start by copying the provided default environment files:

```sh
cp -r envs.final envs
```

Each `.env` file uses the variable `WORK` to specify the cache directory.
Set it before running any experiments:

```sh
export WORK=~/.cache/
```

---

## Node Classification

* **Entry point:** [`./scripts/train/__main__.py`](./scripts/train/__main__.py)

**Example:**

```sh
train envs/dblp/atomic-dblp.sRGCN.env
```

This command trains and evaluates the specified model on a node classification
dataset using the configuration defined in the `.env` file.

---

## Link Prediction

* **Entry point:** [`./scripts/train/linkpred/__main__.py`](./scripts/train/linkpred/__main__.py)

**Example:**

```sh
linkpred envs/amazon/atomic-amazon.sRGCN.env
```

The task type (link prediction) is inferred automatically from the dataset.

---

## Precomputation (OGBN-MAG)

For **OGBN-MAG**, feature propagation and precomputation must be performed
**manually** before training.

### Feature Propagation

```sh
propfeat envs/ogbn-mag/atomic-mag.sRGCN.env -K3 &&
propfeat envs/ogbn-mag/atomic-mag.sRGCN.env -K4 --lpa --lpa-batch-size=128
```

### Training Entry Point

* **Entry point:** [`./scripts/precom/__main__.py`](./scripts/precom/__main__.py)

```sh
precom envs/ogbn-mag/atomic-mag.sRGCN.env
```

---

## Benchmarking (Multiple Runs)

For multi-run evaluation, replace the task command (`train`, `linkpred`, or `precom`)
with `benchmark`. The task type is inferred automatically.

**Example (5 runs):**

```sh
benchmark envs/acm/acm.sRGCN.env -r5
```

---

## Schema Optimization with Ray Tune

We provide **Ray Tune** integration for schema optimization using genetic algorithms (GA).
The search space is defined directly in Python configuration files.

**Example (1024 trials):**

```sh
raytune envs/imdb/GA/tune.GA.py -n 1024
```

* For analyzing Tune results, see:
  [https://docs.ray.io/en/latest/tune/examples/tune_analyze_results.html](https://docs.ray.io/en/latest/tune/examples/tune_analyze_results.html)
* A result-parsing template is provided at:
  [`scripts/get_ray_results.py`](./scripts/get_ray_results.py)

## Citation
If you find AtomHIN useful in your research, please consider citing our paper:

```
@inproceedings{lin2026atomhin,
  title     = {Atomic HINs: Entity-Attribute Duality for Heterogeneous Graph Modeling},
  author    = {Shao-En Lin and Ming-Yi Hong and Miao-Chen Chiang and Chih-Yu Wang and Che Lin},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```
