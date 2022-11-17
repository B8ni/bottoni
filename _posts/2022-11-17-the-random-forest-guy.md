---
toc: true
layout: post
description: Aluminium Scrap Box Classification.
categories: [project, random, forest, multi, class, classification]
title: The Random Forest Guy
---

## Scrap Box Dataset

Days passed from my [first Random Forest practical experiment](https://b8ni.github.io/bottoni/fastai/2022/10/26/aluminium-scraps-box-weight-random-forest-post.html), where I was attempting to predict the weight of an Aluminium Scarp Box.

Spending days going deeper on Random Forest, here you can find a revisioned and hope improved version of the [previous one post](https://b8ni.github.io/bottoni/fastai/2022/10/26/aluminium-scraps-box-weight-random-forest-post.html).

[Short learning cycle](https://youtu.be/yrtAoBr3iuQ?t=144) suggested me, gradually, what's matter the most. 

Figure out the metrics *properly*. 

Same tip and trick came from [Thakur book](https://github.com/abhishekkrthakur/approachingalmost/blob/master/AAAMLP.pdf) where he underlines, before any kind of splitting: understand the data and implement the right metric.

[Target drives metric](), therefore undestanding deeply the target will return the right metric.

### The Problem

Initially the problem to solve included `681` classes. Now I've kept only the `11` most common. 

[Previously](https://b8ni.github.io/bottoni/fastai/2022/10/26/aluminium-scraps-box-weight-random-forest-post.html) I was using the wrong metric, today I switched to [AUC ROC]() metric where it's mainly used on multi class classification problem.

So, but what's the target? A multi class classification problem with imbalanced data. It took me a while but worth it.

Wait, imbalanced what? I don't know yet. Let's dig into unbalanced data another day.

## Explore the Dataset

```python
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

df = pd.read_csv("scraps/scrap_202210181239.csv")
```

```python
df.shape
```


```python
df["tare_weight"].nunique()
```


```python
df["tare_weight"].value_counts().head(11)
```


```python
top_classes = df["tare_weight"].value_counts().head(11)
```


```python
(df.shape[0]-top_classes.sum())/df.shape[0] *100
```

In my case I want to reduce the target spectrum. From `681` classes to `11` classes. This target reduction impacts the dataset by `4.47%` of size. `670` classes are the result of **inappropriate software usage.** I'm pretty confident the current inserts are happening mostly right.


```python
top_classes["top_classes"] = top_classes.index
```


```python
df = df[df['tare_weight'].isin(top_classes["top_classes"])]
```


```python
df.shape
```


```python
82388 - 78708
```

Removed `3680` rows which meet the `670` surplus classes: a bit cut for a big up.

Let's see features and target correlation with `pairplot` method.


```python
import seaborn as sns
# df_2 = df_2[df_2["weight"] <= 3500]
sns.pairplot(df[:50], hue="tare_weight")
```

I don't see any strong linear correlation (except fews which are duplicated features). It suggests Random Forest, thanks to its ability to work [uninformative features](https://hal.archives-ouvertes.fr/hal-03723551v2/document), would take advantage of the dataset form.

## Data Preprocessing


```python
from fastai.tabular.all import Categorify, FillMissing, cont_cat_split, RandomSplitter

dep = "tare_weight"

df = df.drop("net_weight", axis=1)

procs = [Categorify, FillMissing]
```


```python
df = df.rename(columns={"max_tickness.1": "article_max_tickness",
                        "min_tickness.1": "article_min_tickness",
                        "max_tickness": "alloy_max_tickness",
                        "min_tickness": "alloy_min_tickness",
                        "name": "location_name"})
```


```python
cont,cat = cont_cat_split(df, 1, dep_var=dep)
```


```python
splits = RandomSplitter(valid_pct=0.25, seed=42)(df)
```


```python
from fastai.tabular.all import TabularPandas 
to = TabularPandas(
    df, procs, cat, cont, 
    y_names=dep, splits=splits)
```


```python
to.train.xs.iloc[:3]
```


```python
len(to.train),len(to.valid)    
```


```python
from fastai.tabular.all import save_pickle
save_pickle('to.pkl',to)
```


```python
from fastai.tabular.all import load_pickle
to = load_pickle('to.pkl')
```


```python
xs,y = to.train.xs,to.train.y
valid_xs,valid_y = to.valid.xs,to.valid.y
```


```python
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
```


```python
def ovr_rf(xs, y, n_estimators=40,
       max_features=0.5, min_samples_leaf=5, **kwargs):
    return OneVsRestClassifier(RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf, oob_score=True)).fit(xs, y)
```

Here I've simply re-adapted a [Jeremy](https://course.fast.ai/Lessons/lesson6.html) [function](https://github.com/fastai/fastbook/blob/master/09_tabular.ipynb) to work with [One-Versus-Rest pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html). I'm improving my buzzy worlds man!


```python
m  = ovr_rf(xs, y)
```


```python
pred_prob = m.predict_proba(valid_xs)
```


```python
pred_prob
```

Actually I'm not using the classic `predict()` method. `pred_prob` is an array - generated by `predict_proba()` method - which contains classes probabilities. See also [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html?highlight=onevsrest+predict_proba#sklearn.multiclass.OneVsRestClassifier.predict_proba) [source code](https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b6/sklearn/multiclass.py#L450).

### ROC Curve

Now it's time to analyze our performance with a different metric: AUC ROC.

First [encoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html?highlight=labelencoder#sklearn.preprocessing.LabelEncoder) all classes then [binirize](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.label_binarize.html?highlight=label_binarize#sklearn.preprocessing.label_binarize) and finally plot them.


```python
#Lets encode target labels (y) with values between 0 and n_classes-1.
#We will use the LabelEncoder to do this. 
from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
label_encoder.fit(valid_y)
transfomerd_valid_y=label_encoder.transform(valid_y)
classes=label_encoder.classes_
```


```python
from sklearn.preprocessing import label_binarize
#binarize the y_values
plt.figure(figsize = (15, 10))

y_test_binarized=label_binarize(valid_y,classes=np.unique(valid_y))

# roc curve for classes
fpr = {}
tpr = {}
thresh ={}
roc_auc = dict()

n_class = classes.shape[0]

for i in range(n_class):    
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test_binarized[:,i], pred_prob[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
    # plotting    
    plt.plot(fpr[i], tpr[i], linestyle='--', 
             label='%s vs Rest (AUC=%0.2f)'%(classes[i],roc_auc[i]))
    

plt.plot([0,1],[0,1],'b--')
plt.xlim([0,1])
plt.ylim([0,1.05])
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='lower right')
plt.show()
```


```python
avg_roc_auc = pd.Series(roc_auc)
avg_roc_auc.mean()
```

An average of `94%` of being right is really good. Only `750` box is mainly miss-classified, with a `83%`.

## Feature Selection

Feature selection starts from ``feature_importances_``. I've adapted [Jeremy](https://course.fast.ai/Lessons/lesson6.html) [method](https://github.com/fastai/fastbook/blob/master/09_tabular.ipynb) to work with multi class model.

### Feature Importances


```python
def rf_feat_importance(m, df, i):
    return pd.DataFrame({'cols':df.columns, 'imp':m.estimators_[i].feature_importances_}
                       ).sort_values('imp', ascending=False)
```

Every class have its own feature importances so I have to compress everything in array and remove the last one. I've implemented a simple `concat` and `mean`.


```python
df_all = pd.DataFrame()
for i in range(df["tare_weight"].nunique()):
    df_all = pd.concat([df_all, rf_feat_importance(m, xs, i)])
```


```python
cols = df_all["cols"].sort_index().unique()
```


```python
df_all = df_all.groupby(df_all.index).mean()
```


```python
df_all["cols"] = cols
df_all = df_all.sort_values('imp', ascending=False)
```

Finally plotting averaged feature importances of the whole classes.


```python
def plot_fi(fi):
    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)

plot_fi(df_all[:30]);
```

Let's remove less significant ones.


```python
df_all[df_all["imp"] >= 0.002]
```


```python
fi = df_all[df_all["imp"] < 0.002]

filtered_xs = xs.drop(fi["cols"], axis=1)
filtered_valid_xs = valid_xs.drop(fi["cols"], axis=1)
```


```python
filtered_xs.shape, filtered_valid_xs.shape
```


```python
m = ovr_rf(filtered_xs, y)
```


```python
pred_prob = m.predict_proba(filtered_valid_xs)
```


```python
def roc_plot(classes):
    plt.figure(figsize = (15, 10))

    y_test_binarized=label_binarize(valid_y,classes=np.unique(valid_y))

    # roc curve for classes
    fpr = {}
    tpr = {}
    thresh ={}
    roc_auc = dict()

    n_class = classes.shape[0]

    for i in range(n_class):    
        fpr[i], tpr[i], thresh[i] = roc_curve(y_test_binarized[:,i], pred_prob[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        # plotting    
        plt.plot(fpr[i], tpr[i], linestyle='--', 
                 label='%s vs Rest (AUC=%0.2f)'%(classes[i],roc_auc[i]))


    plt.plot([0,1],[0,1],'b--')
    plt.xlim([0,1])
    plt.ylim([0,1.05])
    plt.title('Multiclass ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='lower right')
    plt.show()
```


```python
roc_plot(classes)
```


```python
def avg_roc_auc(pred_prob):
    return pd.Series(roc_auc_classes(pred_prob)).mean()  
```


```python
avg_roc_auc(pred_prob)
```

With just removing the less important ones, the model has improved by few decimals.


```python
from fastai.tabular.all import save_pickle
save_pickle('filtered_xs.pkl',filtered_xs)
save_pickle('filtered_valid_xs.pkl',filtered_valid_xs)
filtered_xs = load_pickle('filtered_xs.pkl')
filtered_valid_xs = load_pickle('filtered_valid_xs.pkl')
```

### Features Correlation


```python
import matplotlib.pyplot as plt
import seaborn as sn

xs_corr = filtered_xs.corr()
compressed_xs = xs_corr[((xs_corr >= .5) | (xs_corr <= -.5)) & (xs_corr !=1.000)]
plt.figure(figsize=(30,10))
sn.heatmap(compressed_xs, annot=True, cmap="Reds")
plt.show()
```


```python
def corrFilter(x: pd.DataFrame, bound: float):
    xCorr = x.corr()
    xFiltered = xCorr[((xCorr >= bound) | (xCorr <= -bound)) & (xCorr !=1.000)]
    xFlattened = xFiltered.unstack().sort_values().drop_duplicates()
    return xFlattened

corrFilter(filtered_xs, .65)
```


```python
def oob_estimators(filtered_xs):
    m = ovr_rf(filtered_xs, y)
    return [m.estimators_[i].oob_score_ for i in range (df["tare_weight"].nunique())]
```

Since I'm working with a dataset with `11` classes, it's essential to evaluate for each class relative Out-of-Bag score. So the goal is to remove closely correlated features which keep stagnant or improve the OOB score.


```python
oob_estimators(filtered_xs)
```


```python
to_drop = ["id", "timestamp", "slim_alloy", "id_alloy", "pairing_alloy",
           "international_alloy", "id_user", "address",
           "location_name", "article_min_tickness", "article_max_tickness_na"]
```


```python
{c:oob_estimators(filtered_xs.drop(c, axis=1)) for c in to_drop}
```

The features belongs to `to_drop` list with an average of `OOB` score higher, will be dropped.


```python
to_drop = ["id", "pairing_alloy", "id_alloy",
           "article_max_tickness_na", "location_name", "article_max_tickness_na"]
```


```python
filtered_xs = filtered_xs.drop(to_drop, axis=1)
filtered_valid_xs = filtered_valid_xs.drop(to_drop, axis=1)
```


```python
filtered_valid_xs.shape, filtered_xs.shape
```


```python
m = ovr_rf(filtered_xs, y)
```


```python
pred_prob = m.predict_proba(filtered_valid_xs)
```


```python
roc_plot(pred_prob)
```


```python
avg_roc_auc(pred_prob)
```

Obtaining `94.5%` `AUC ROC` score while keeping `OOB` score higher is a good achievement. Breakpoint saved.


```python
save_pickle('filtered_xs.pkl',filtered_xs)
save_pickle('filtered_valid_xs.pkl',filtered_valid_xs)
filtered_xs = load_pickle('filtered_xs.pkl')
filtered_valid_xs = load_pickle('filtered_valid_xs.pkl')
```

## Baseline Result

Now it's time to fix [Out of Domain Data]() to minimize overfitting.

### Out of Domain Data


```python
def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)
```


```python
df_dom = pd.concat([filtered_xs, filtered_valid_xs])
is_valid = np.array([0]*len(filtered_xs) + [1]*len(filtered_valid_xs))

m = rf(df_dom, is_valid)
rf_feat_importance(m, df_dom)[:15]
```


```python
for c in ('timestamp', 'weight', 'slim_alloy', 
          'international_alloy', 'id_machine_article_description',
          'id_idp_user', 'last_name', 'id_machine', 'slim_number',
          'first_name', 'code_machine', 'description_machine'):
    m = ovr_rf(filtered_xs.drop(c,axis=1), y)
    pred_prob = m.predict_proba(filtered_valid_xs.drop(c,axis=1))
    print(c, avg_roc_auc(pred_prob))
```


```python
to_drop = ['international_alloy', 'last_name', 'id_machine', 'slim_number', 'description_machine']
```


```python
xs_final = filtered_xs.drop(to_drop, axis=1)
valid_xs = filtered_valid_xs.drop(to_drop, axis=1)
```


```python
xs_final.shape, valid_xs.shape
```


```python
m = ovr_rf(filtered_xs, y)
pred_prob = m.predict_proba(filtered_valid_xs)
avg_roc_auc(pred_prob), oob_estimators_avg(m)
```

Everything ended with less features (`15`) and **higher score** both `AUC ROC` and `OOB`.


```python
save_pickle('final_xs.pkl',xs_final)
save_pickle('final_valid_xs.pkl',valid_xs)
```

### Hyperparameter Tuning

Before the game end I'll try some hypertuning.


```python
xs_final = load_pickle('final_xs.pkl')
valid_xs = load_pickle('final_valid_xs.pkl')
```


```python
m.get_params()
```


```python
from sklearn.model_selection import RandomizedSearchCV# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 200, num = 4)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 50, num = 5)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]# Create the random grid
random_grid = {'estimator__n_estimators': n_estimators,
               'estimator__max_features': max_features,
               'estimator__max_depth': max_depth,
               'estimator__min_samples_split': min_samples_split,
               'estimator__min_samples_leaf': min_samples_leaf,
               'estimator__bootstrap': bootstrap}
```


```python
random_grid
```


```python
from sklearn.model_selection import ShuffleSplit
sp = ShuffleSplit(n_splits=2, test_size=.25, random_state=42)
```


```python
rf.get_params().keys()
```


```python
from sklearn.ensemble import RandomForestClassifier 
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = OneVsRestClassifier(RandomForestClassifier(oob_score=True))
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = sp, verbose=2, random_state=42, n_jobs = 3)# Fit the random search model
rf_random.fit(xs_final, y)
```


```python
rf_random.best_params_
```


```python
from sklearn.metrics import accuracy_score
best_model = rf_random.best_estimator_
pred_prob = best_model.predict_proba(valid_xs)
avg_roc_auc(pred_prob), oob_estimators_avg(best_model)
```

Now narrowing the range and trying to gain lil decimals.


```python
from sklearn.model_selection import RandomizedSearchCV# Number of trees in random forest
n_estimators = [50, 100, 150]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [30, 50, 100]
# Minimum number of samples required to split a node
min_samples_split = [5, 10, 20]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True]# Create the random grid
random_grid = {'estimator__n_estimators': n_estimators,
               'estimator__max_features': max_features,
               'estimator__max_depth': max_depth,
               'estimator__min_samples_split': min_samples_split,
               'estimator__min_samples_leaf': min_samples_leaf,
               'estimator__bootstrap': bootstrap}
```


```python
random_grid
```


```python
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = OneVsRestClassifier(RandomForestClassifier(oob_score=True))
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = sp, verbose=2, random_state=42, n_jobs = 3)# Fit the random search model
rf_random.fit(xs_final, y)
```


```python
rf_random.best_estimator_
```


```python
narrowed_model = rf_random.best_estimator_
pred_prob = narrowed_model.predict_proba(valid_xs)
avg_roc_auc(pred_prob), oob_estimators_avg(narrowed_model)
```

From `94%` to almost `94.7%` is the final score. OOB stable on `95.3%` range.

## Conclusion

Miss-classifying the tare weight (Aluminium scarp box) is expensive causing **damage to the company (less revenue) and environment (re-melting Aluminium)**. 

**Scoring a `94.7%` of predicting right is a great baseline. Sure less scraps will be wasted.**

## Further Work

1. Develop service which host the model.
2. How reacts the model if I remove duplicated rows? Do it.
3. I know the dataset is imbalanced. Implement it. 
4. Compare the result with [deep learning tabular model](https://arxiv.org/pdf/2207.08815.pdf).
5. Compare the result with XGBoost model.
6. Using same method to classify Aluminium alloys.
