# Machine Learning snippets

I'm learning ML and those are code snippets that are useful for me.
**Contributions are welcomed!**

# Basics

## Imports
```
%matplotlib inline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import re, json
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
```

## Mapping rows

```python
def transform_row(row):
    row["new_column"] = np.log(row["old_column"])
    row["another_column"] = np.mean(row["old_column"])
    return row

df = df.apply(transform_row, axis=1)
```

## Mapping column

```python
df["fresh_column"] = df["old_column"].map(lambda x: np.log(x) if x > 0 else 0)
```

## Merging by column

```python
new_df = pd.merge(left=df_1, right=df_2, left_on=df_1_column, right_on=df_2_column)
```

## Merging by appending

```python
new_df = pd.concat([df_1, df_2])
```

## Dropping colums

```python
new_df = df.drop(columns=["name"])

# or

df.drop(columns=["name"], inplace=True)
```

## Checking for missing values

```python
df.isnull().sum()
```

## Checking values

```python
df["column"].describe() # scalars
df["column"].value_counts() # categorical
```

# Drawing

## Distribution

```python
def draw_dist(data, x, hue):
    fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(15, 8))
    ax = sns.distplot(
        data[data[hue]==1][x].dropna(),
        bins=20,
        label = hue,
        ax = axes,
        kde = False
    )
    ax.legend()
    ax = sns.distplot(
        data[data[hue]==0][x].dropna(), 
        bins=20, 
        label = 'not ' + hue, 
        ax = axes, 
        kde = False
    )
    ax.legend()

draw_dist(df, "amount", "fraud") # Usage
```

## Swarm

Works best if `x` has less than 4 values.

```python
def draw_swarm(data, x, y, hue):
    fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(15, 8))
    sns.swarmplot(x=x, y=y, data=data, hue=hue, ax=axes)

draw_swarm(df, "age", "amount", "fraud") # Usage
```

## Count

Works best if `x` has few values.

```python
def draw_count(data, x, hue):
    fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(15, 8))
    sns.countplot(x = x, hue = hue, data = data, ax=axes)

draw_count(df, "age", "fraud")
```

## Correlation

Hierarchy shows what features are redundant.

```python
def draw_corr(df):
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(15, 8))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

def draw_corr_with_hierarchy(df)
    sns.clustermap(df.corr())
```

## Feature importance

```python
def get_importance(data, model):
    fi = {
        'Features': data.columns.to_list(), 
        'Importance': model.feature_importances_
    }
    return pd.DataFrame(fi, index=None).sort_values('Importance', ascending=False)

def draw_importance(data, model):
    importance = get_importance(data, model)
    sns.barplot(y=importance.Features, x=importance.Importance)

# Usage
m = clf.fit(X, y)
draw_importance(X, m)
```

# Feature modeling

## Fixing missing values

```python
df.loc[df["column"].isnull(), ["column"]] = "value"
```

## Dropping not important features

```python
def drop_not_important(df, m, alpha = 0.05)
    importance = get_importance(m)
    to_keep = importance[importance.Importance > alpha].Features
    return df[to_keep]
```

## Dropping outliers

Useful when there are small number of outliers. 
Don't do it, if you have unbalanced data.
You might delete data that is already scarse.


```python
def get_outliers(data, column):
    from scipy import stats
    z = np.abs(stats.zscore(data[column]))
    return np.where(z > 3)[0].tolist()

def drop_outliers(data, column):
    outliers = get_outliers(data, column)
    
    if len(outliers) < 0:
        return data
    else:
        return data.drop(outliers).reset_index(drop=True)
```

## Upsampling

Useful when categories are unbalanced.

```python
def upsample(df, column):
    from sklearn.utils import resample
    cons = df[df[column]==0]
    pros = df[df[column]==1]
    pros_upsampled = resample(fraud, replace=True, n_samples=len(cons))

    return pd.concat([cons, pros_upsampled])
```
