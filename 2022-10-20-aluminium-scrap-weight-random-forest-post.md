---
toc: true
layout: post
description: Not so insightful discovery with Random Forest.
categories: [fastai]
title: How Random Forest Can Empower A Small Business 
---

![]({{ site.baseurl }}/images/forrest-gamp.png "Forrest Gump in a Random Forest")
## Preamble
While the entire world is totally captured by Stable Diffusion, I'm experimenting **randomly into the forest of Random Forest**. Here my 2 cents after about 60+ hours of fighting against Random Forest. Actually [Forrest](https://en.wikipedia.org/wiki/Forrest_Gump) is winning the game.

### Why predicting Boxes weight?
It's hard to fight entropy in my house. 
It's exponentially hard to fight entropy in a plant, in an Aluminium plant precisely.

The factory would gain lots of benefits when scraps are segregated, weighted and labeled properly[^1].
Better the process and higher the impact on the revenue of the company (true story).

The weighting process is simple: take the box, put it on an industrial scale, get the weight and repeat.

The weighting process, although heavily based on an inductive flow, it's not enough. The operators have still room of errors.
How can I solve this problem, without spending lot's of money? How can I monitoring the situation and eventually notify others departments? 
Wrong answers only: Aluminium scarps data and Random Forest. 

I concluded that Boxes due of limited types of them, can be solved or partially solved with a classification model.

Now let's see which one better performs.

**put a funny image here**

## First Round
**Model:** Random Forest Classifier.
**Dataset:** Extended. CSV file, 82k rows and 33 columns.
**Verbose:** I've joined most interesting tables, adding intentionally duplicated or closely correlated columns. May increase the noise but may drive to a better prediction as well.

### Preprocessing
Since I already worked with this data, I found a subtle feature, which is a calculated field where would create lots of trouble in production environment.
``net_weight`` is accused of Data Leakage so I firstly dropped it.

Obviously, Data Leakage is an issue faced quite the end of experimentation but, IMHO, earlier you find and better it is. It's mean you understand enough the dataset.

Via ``Categorify``, ``FillMissing``, ``cont_cat_split`` and ``RandomSplitter`` functions, the data is ready to be fitted.

```python
from fastai.tabular.all import Categorify, FillMissing, cont_cat_split, RandomSplitter
dep = "tare_weight"
df = df.drop("net_weight", axis=1)
procs = [Categorify, FillMissing]
```

```python
cont,cat = cont_cat_split(df, 1, dep_var=dep)
```

```python
splits = RandomSplitter(valid_pct=0.25, seed=42)(df)
```

I think ``cont_cat_split`` is a nice function to spend few seconds with. Going to [its source code](https://github.com/fastai/fastai/blob/master/fastai/tabular/core.py#L84), I can see how genuinely the continuous and categorical variables are managed:
```python
def cont_cat_split(df, max_card=20, dep_var=None):
	"Helper function that returns column names of cont and cat variables from given `df`."
	cont_names, cat_names = [], []
	for label in df:
		if label in L(dep_var): continue
		if ((pd.api.types.is_integer_dtype(df[label].dtype) and
			df[label].unique().shape[0] > max_card) or
			pd.api.types.is_float_dtype(df[label].dtype)):
			cont_names.append(label)
		else: cat_names.append(label)
	return cont_names, cat_names
```
For every column in dataframe, if it has more then 20 elements or it's a float, appends to continuous, otherwise categorical. I've no experience with FastAI API, but I suppose the library is full of such elegant and simple way to manage complex data and task.

Once pre-processing step is completed, the dataframe is putted inside a ``TabularPandas``.
``TabularPandas`` is an object. It's a simple wrapper with transforms.
Transforms are functions which organize the data in an optimal format.
```python
from fastai.tabular.all import TabularPandas 
to = TabularPandas(
    df, procs, cat, cont, 
    y_names=dep, splits=splits)
```
```python
to.train.xs.iloc[:3]
```
![]({{ site.baseurl }}/images/Pasted image 20221024142446.png)

>Machine learning models are only as good as the data that is used to train them.

Better data format, better generalization.

Now, save and train.
```python
from fastai.tabular.all import save_pickle
save_pickle('to.pkl',to)
```

### Fitting

```python
from fastai.tabular.all import load_pickle
load_pickle('to.pkl', to)
```
Jeremy has developed a function which wraps ``RandomForestClassifier``. It turns useful later to play with attributes of function:
```python
from sklearn.ensemble import RandomForestClassifier

def rf(xs, y, n_estimators=100,
       max_features=0.5, min_samples_leaf=5, **kwargs):
    return RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf, oob_score=True).fit(xs, y)
```
```python
m = rf(xs, y)
```
A good error metrics to understand what's going on is a simple ``mean_absolute_error``:
```python 
from sklearn.metrics import mean_absolute_error
mean_absolute_error(m.predict(xs), y), mean_absolute_error(m.predict(valid_xs), valid_y)
```
![]({{ site.baseurl }}/images/Pasted image 20221024142733.png)![[]]

What's ``mean_absolute_error``? Going to the [source code of scikit-learn](https://github.com/scikit-learn/scikit-learn/blob/36958fb24/sklearn/metrics/_regression.py#L141), I found line which calculate MAE:
```python
np.average(np.abs(y_pred - y_true), weights=sample_weight, axis=0)
```
It means:
1. calculate the delta between ``(y_pred - y_true)``
2. take the absolute value ``np.abs`` of the whole rows ``axis=0``
3. finally calculate the average with ``np.average``

Nothing to say, simple enough.

### Out Of Bag Error

Next to ``mean_absolute_error``s I have to place ``m.oob_score_`` which returns the accuracy of predictions on the residual rows not used during training.
Obviously higher score, I should expect a better generalization on validation set.

```python
m.oob_score_
```
![[Pasted image 20221025160903.png]]

There's so much resources where explain acutely and precisely what the hell OOB is. I'm not the right person to do that:
> My intuition for this is that, since every tree was trained with a different randomly selected subset of rows, out-of-bag error is a little like imagining that every tree therefore also has its own validation set. That validation set is simply the rows that were not selected for that tree's training.

### Intermediate Result

``47.06`` and ``73.49`` are just numbers. But what does it mean?
I have achieved, via a simple ``RandomForestClassifier`` with ``100`` trees (n_estimators), an average of:
- ``47.06`` KG of error on training set
- ``73.49`` KG of error on validation set

And an accuracy of ``0.709`` on the residual data not included in the fitting step.

For this reason, there are multiple objectives to try to achieve: a good trade off should be met by the following chain:
``small_enough_error > stability > maintainability``

The next steps I'm going to walk will aim to improve the above chain.

## Second Round

``RandomForest`` is composed by multiples ``DecisionTrees``. 
``DecisionTrees`` are highly interpretable, so it's time to investigate the data:
1. analyzing the most important columns, AKA ``feature_importances_``
2. analyzing the prediction behavior for each row, AKA ``treeinterpreter``
3. finding redundant columns, AKA ``cluster_columns``
4. analyzing prediction confidence of the model, AKA ``std`` of each tree
5. analyzing the relationship between independent variables and dependent variable, AKA ``partial_dependece``
6. finding out of domain data, AKA extrapolation problem
7. analyzing where most wrong prediction happens, AKA ``confusion_matrix``

### Feature Importances

```python
def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)
``` 
```python
fi = rf_feat_importance(m, xs)
fi[:5]
```
![[Pasted image 20221024145132.png]]

According to the above table:
- the box weight prediction is mainly influenced by ``weight`` itself[^4]. Sounds reasonable;
- ``id_machine``, in other words the machine which generates scraps, is the second most indicator of box weight prediction. Sounds reasonable as well;
- ``id_machine_article_description`` is the combination between ``id_machine``, ``article`` and ``description_machine``, where ``article`` is the thickness range of scarps (Ex.: from 0.5mm to 0.25mm);
- percentage of ``id`` and ``timestamp`` is too similar. Maybe, periodically, I can expect a specific type of scraps? 
- ``code_machine`` is the short name of machine;
- ``last_name``, the operator, contributes to the box weight prediction as well. Maybe some operators are more diligent then others?
- ``description_machine`` is extended name of machine;

Everything sounds reasonable so it seems I've discovered nothing so useful.

Let's visualize ``feature_importances_`` columns.
```python
def plot_fi(fi):
    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)

plot_fi(fi[:30]);
```
![[Pasted image 20221024161156.png]]

Now let's remove from training and validation sets features which tend to ``0`` .
```python
fi = fi[fi["imp"] < 0.002]

filtered_xs = xs.drop(fi["cols"], axis=1)
filtered_valid_xs = valid_xs.drop(fi["cols"], axis=1)
```
Then fitting again the model and check the error rate (``mean_absolute_error`` and ``oob_score_``).
```python
m = rf(filtered_xs, y)
```
```python
mean_absolute_error(m.predict(filtered_xs), y), mean_absolute_error(m.predict(filtered_valid_xs), valid_y)
```
![[Pasted image 20221024161755.png]]

```python
m.oob_score_
```
![[Pasted image 20221025161630.png]]

Has been achieved few improvements:
- ``mean_absolute_error`` on training set is smaller: from ``47.06`` to ``46.78``;
- ``mean_absolute_error`` on validation set is smaller: from ``73.49`` to ``73.47``;
- ``oob_score_`` stable: from ``0.709`` to ``0.708``
- features reduced: from 33 to 25.

Now, let's hunt redundant features.

### Data points based on their Similarities
```python
# https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas
def corr_filter(x: pd.DataFrame, bound: float):
    corr = x.corr()
    x_filtered = corr[((corr >= bound) | (corr <= -bound)) & (corr !=1.000)]
    x_flattened = x_flattened.unstack().sort_values().drop_duplicates()
    return x_flattened

corr_filter(filtered_xs, .8)
```
![[Pasted image 20221025101554.png]]
Giving a threshold of ``0.8``, function will return set of elements highly correlated with a score from ``0.8`` to ``0.9999``.

For a better understanding, worth to visualize them.
```python
# https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas
import matplotlib.pyplot as plt
import seaborn as sn

xs_corr = filtered_xs.corr()
compressed_xs = xs_corr[((xs_corr >= .5) | (xs_corr <= -.5)) & (xs_corr !=1.000)]
plt.figure(figsize=(30,10))
sn.heatmap(compressed_xs, annot=True, cmap="Reds")
plt.show()
```
![[Pasted image 20221024163716.png]]

An alternative to ``heatmap`` is the helper function ``cluster_columns`` which implement a ``dendrogram`` chart.
[link to dendogram chart. understand and create a separated post for each important function]
```python
# https://github.com/fastai/fastbook/blob/master/09_tabular.ipynb
from scipy.cluster import hierarchy as hc

def cluster_columns(df, figsize=(10,6), font_size=12):
    corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
    corr_condensed = hc.distance.squareform(1-corr)
    z = hc.linkage(corr, method='average')
    fig = plt.figure(figsize=figsize)
    hc.dendrogram(z, labels=df.columns, orientation='left', leaf_font_size=font_size)
    plt.show()
```

Now, iteratively remove every closely correlated feature and calculate ``oob_score_``. This task is performed by ``get_oob`` function:
```python
def get_oob(df):
    m = RandomForestClassifier(n_estimators=40, min_samples_leaf=15,
        max_samples=50000, max_features=0.5, n_jobs=-1, oob_score=True)
    m.fit(df, y)
    return m.oob_score_
```
```python
get_oob(filtered_xs)
```
![[Pasted image 20221025162507.png]]
```python
to_drop = ["id", "timestamp", "slim_alloy", "id_alloy", "pairing_alloy",
           "international_alloy", "id_user", "address",
           "location_name", "article_min_tickness", "article_max_tickness_na"]
```
```python
{c:get_oob(filtered_xs.drop(c, axis=1)) for c in to_drop}
```
![[Pasted image 20221025162546.png]]

Going to remove only features with higher score.

```python
to_drop = ["timestamp", "id_alloy", "id_user", "address", "article_min_tickness"]
filtered_xs = filtered_xs.drop(to_drop, axis=1)
filtered_valid_xs = filtered_valid_xs.drop(to_drop, axis=1)
```
```python
m = rf(filtered_xs, y)
```
```python
mean_absolute_error(m.predict(filtered_xs), y), 
mean_absolute_error(m.predict(filtered_valid_xs), valid_y)
```
![[Pasted image 20221025104016.png]]
```python
m.oob_score_
```
![[Pasted image 20221025164058.png]]

### Intermediate Result

Not much worse than the model with all the fields. I've reduced some more columns (from ``25`` to ``20``) and kept stable ``oob_score_``.
Removing redundant features help to prevent **overfitting**.

## Third Round

As showed by Jeremy, Random Forest can sin of Extrapolation problem (:open_mouth:).
![[Pasted image 20221025172423.png]]

It means, in this case, predictions are too low with new data.

> Remember, a random forest just averages the predictions of a number of trees. And a tree simply predicts the average value of the rows in a leaf. Therefore, a tree and a random forest can never predict values outside of the range of the training data. This is particularly problematic for data where there is a trend over time, such as inflation, and you wish to make predictions for a future time. Your predictions will be systematically too low.

For this reason I've to make sure validation set does not contain **out-of-domain data**.

### Out-of-Domain Data

How to understand if the data is distributed quite properly on training set and validation set?
```python
df_dom = pd.concat([filtered_xs, filtered_valid_xs])
is_valid = np.array([0]*len(filtered_xs) + [1]*len(filtered_valid_xs))

m = rf(df_dom, is_valid)
rf_feat_importance(m, df_dom)[:15]
```
![[Pasted image 20221026103311.png]]

Now, for each feature which vary a lot from training set and validation set, try to drop and check ``mean_absolute_error``. Finally, select those that keep improving the model.

```python
print('orig', mean_absolute_error(m.predict(filtered_valid_xs), valid_y))

for c in ('id','weight', 'international_alloy', 'slim_alloy',
          'pairing_alloy', 'id_machine_article_description', 'location_name', "last_name"):
    m = rf(filtered_xs.drop(c,axis=1), y)
    print(c, mean_absolute_error(m.predict(filtered_valid_xs.drop(c,axis=1)), valid_y))
```
![[Pasted image 20221026104401.png]]

Let's drop only ``slim_alloy``.
```python
to_drop = ['slim_alloy']

xs_final = filtered_xs.drop(to_drop, axis=1)
valid_xs = filtered_valid_xs.drop(to_drop, axis=1)

m = rf(xs_final, y)
mean_absolute_error(m.predict(valid_xs), valid_y)
```
![[Pasted image 20221026104546.png]]

Keep checking out of bag error:
```python
m.oob_score_
```
![[Pasted image 20221026104841.png]]

### Intermediate Result

Good news, working on **out-of-domain data** has improved both ``mean_absolute_error`` either ``oob_score_``:
- from ``74.09`` KG to ``73.98`` KG, validation set;
- from ``0.7070`` to ``0.7072``, ``oob_score_``.

What I have achieved so far are only small improvements. Looking at a simple chart which plots the delta between real value and prediction, I can see there's still lot of room to improve.
![[Pasted image 20221026110706.png]]

Some datapoints are consistently predicted wrong (dots at about ``-900/-1000`` and about ``900/1000``). Other visual tools like [Confusion matrix](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html?highlight=confusion+matrix) , **prediction confidence**, [treeinterpreter](http://blog.datadive.net/random-forest-interpretation-with-scikit-learn/) can help to analyze this behavior.  

## Final Round
Before to any hyper-mega-super tuning, I can try my last attempt removing older data.  Why?
The application which manage the weighting/labeling process of scraps has been release about 2 years ago. Wouldn't surprise me if I found some strange datapoints, especially during first period of usage where operators were not comfortable yet with the system.

Re-processing whole steps removing older 12k datapoints, seems to have better baseline model.
![[Pasted image 20221026121947.png]]

There's still miss-classification at around ``-900/-1000`` and ``900/1000``, but it's evident has been reached an improvement.

### Result
- from  ``73.98`` KG to ``72.17`` KG, validation set;
- from ``0.7072`` to ``0.7141``, ``oob_score_``.

I think as baseline model is really good: fast to fit, easily interpretable and quite stable.
All this with with few KBs of data, a laptop and a mediocre baseline model. 

## What's Next?

Once created a baseline model a simplified dataset, now it's time to make a decision: 
- creating a NN model
- or working on Radom Forest tuning
- or switching to XGBoost model 

Remember to apply as much as possible [Pareto principle](https://en.wikipedia.org/wiki/Pareto_principle):

> ...roughly 80% of consequences come from 20% of causes...

It mean to try to leverage and get as good result as soon as possible while keeping to the minimum the effort.

So next steps:
- I will implement a Neural Network model;
- then I'll combine NN with Random Forest;
- the ensembles will work in parallel with a Computer Vision model which will try to classify the same problem (a box of scraps). 

All this staff is aimed to develop an alert system where departments are notified every time the prediction of models are too different from what's happening during the weighting process. 

I've in mind already the application name: **Box ClassifAI**.

**Keep lower scraps errors and push higher revenue. That's it.**

## Open Points
- What happens if I play with categorical and continuous variables? Can them affect the prediction?
- Plotting ``dendogram`` and removing most correlated columns. Does it change the prediction? Are columns the same?
- Why is confidence of prediction totally wrong when the deviation reach value ``100``? Why is prediction not so bad with greater value? 
- Partial dependency plots for multi-class-classifiers?
- Could improve Random Forest model with additional information like weather data?

If you have any suggestions, recommendations, or corrections please reach out to me.

---

[^1]: We have developed few tools to speed up and manage the weighting and labelling process of the Aluminium scarps
[^2]:
[^3]: 
[^4]:
[^5]: 
