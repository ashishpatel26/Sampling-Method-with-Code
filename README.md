
# Sampling Methods

* **Sampling** is a *process used in statistical analysis in which a predetermined number of observations are taken from a larger population.*

---

## Simple Random Sampling
* **Simple random sampling** is the *basic sampling technique where we select a group of subjects (a sample) for study from a larger group (a population).* Each individual is chosen entirely by chance and each member of the population has an equal chance of being included in the sample. Every possible sample of a given size has the same chance of selection. 

![Simple random sampling of a sample “n” of 3 from a population “N” of 12. Image: Dan Kernler |Wikimedia Commons](https://www.statisticshowto.datasciencecentral.com/wp-content/uploads/2014/12/Simple_random_sampling-300x231.png)  
*Simple random sampling of a sample “n” of 3 from a population “N” of 12. Image: Dan Kernler |Wikimedia Commons*

* Technically, a simple random sample is a set of n objects in a population of N objects where all possible samples are equally likely to happen. Here’s a basic example of how to get a simple random sample: put 100 numbered bingo balls into a bowl (this is the population N). Select 10 balls from the bowl without looking (this is your sample n). Note that it’s important not to look as you could (unknowingly) bias the sample. While the “lottery bowl” method can work fine for smaller populations, in reality you’ll be dealing with much larger populations.

![](https://research-methodology.net/wp-content/uploads/2015/04/Simple-random-sampling2.png)


```python
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randn(5000, 4), columns=list('ABCD'))
```


```python
sample_df = df.sample(100)
```


```python
sample_df.shape
```




    (100, 4)



## Stratified Sampling

---

* **Stratified random sampling** is a method of sampling that *involves the division of a population into smaller sub-groups known* as **strata** In stratified random sampling or stratification, the strata are formed based on members' shared attributes or characteristics such as income or educational attainment.

* **Stratified random sampling** is also called *proportional random sampling or quota random sampling.*

![](https://image.slidesharecdn.com/sampling-stratifiedvscluster-170115160432/95/sampling-stratified-vs-cluster-2-638.jpg?cb=1484496290)  

##### Assume that we need to estimate the average number of votes for each candidate in an election. Assume that the country has 3 towns:
* Town A has 1 million factory workers,
* Town B has 2 million workers, and
* Town C has 3 million retirees.
* We can choose to get a random sample of size 60 over the entire population but there is some chance that the random sample turns out to be not well balanced across these towns and hence is biased causing a significant error in estimation.
* Instead, if we choose to take a random sample of 10, 20 and 30 from Town A, B and C respectively then we can produce a smaller error in estimation for the same total size of the sample.

### Method


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y, ## we need one categorical variable for that
                                                    test_size=0.25)
```

## Reservoir Sampling

---
![](https://kapilddatascience.files.wordpress.com/2015/06/reservoir.jpg)  

* **Reservoir sampling** is a *family of randomized algorithms for randomly choosing k samples from a list of n items, where n is either a very large or unknown number.* Typically n is large enough that the list doesn’t fit into main memory. For example, a list of search queries in Google and Facebook.

![](https://image.slidesharecdn.com/t10part1-141208215154-conversion-gate02/95/sampling-for-big-data-1-21-638.jpg?cb=1418075560)  



```python
import random
def generator(max):
    number = 1
    while number < max:
        number += 1
        yield number
# Create as stream generator
stream = generator(10000)
# Doing Reservoir Sampling from the stream
k=5
reservoir = []
for i, element in enumerate(stream):
    if i+1<= k:
        reservoir.append(element)
    else:
        probability = k/(i+1)
        if random.random() < probability:
            # Select item in stream and remove one of the k items already selected
             reservoir[random.choice(range(0,k))] = element
print(reservoir)
```

    [6859, 7151, 2308, 4500, 4533]


It can be mathematically proved that in the sample each element has the same probability of getting selected from the stream.

## Random Undersampling and Oversampling

---

![](https://miro.medium.com/max/700/0*u6pKLqdCDsG_5kXa.png)  

* A widely adopted technique for dealing with highly imbalanced datasets is called resampling. It consists of *removing samples from the majority class* (**under-sampling**) and/or *adding more examples from the minority class* (**over-sampling**).


```python
from sklearn.datasets import make_classification
X, y = make_classification(
    n_classes=2, class_sep=1.5, weights=[0.9, 0.1],
    n_informative=3, n_redundant=1, flip_y=0,
    n_features=20, n_clusters_per_class=1,
    n_samples=100, random_state=10
)
X = pd.DataFrame(X)
X['target'] = y
```

We can now do random oversampling and undersampling using:


```python
num_0 = len(X[X['target']==0])
num_1 = len(X[X['target']==1])
print(num_0,num_1)
# random undersample
undersampled_data = pd.concat([ X[X['target']==0].sample(num_1) , X[X['target']==1] ])
print(len(undersampled_data))
# random oversample
oversampled_data = pd.concat([ X[X['target']==0] , X[X['target']==1].sample(num_0, replace=True) ])
print(len(oversampled_data))
```

    90 10
    20
    180


## Undersampling and Oversampling using imbalanced-learn

* imbalanced-learn(imblearn) is a Python Package to tackle the curse of imbalanced datasets.
It provides a variety of methods to undersample and oversample.

#### A. Undersampling using Tomek Links:  
One of such methods it provides is called Tomek Links. Tomek links are pairs of examples of opposite classes in close vicinity.
In this algorithm, we end up removing the majority element from the Tomek link which provides a better decision boundary for a classifier.

![](https://miro.medium.com/max/700/0*huy_9J15wzYJ2o5S)


```python
from imblearn.under_sampling import TomekLinks

tl = TomekLinks(return_indices=True, ratio='majority')
X_tl, y_tl, id_tl = tl.fit_sample(X, y)
```

    Using TensorFlow backend.


#### B. Oversampling using SMOTE:

In SMOTE (Synthetic Minority Oversampling Technique) we synthesize elements for the minority class, in the vicinity of already existing elements.

![](https://miro.medium.com/max/700/0*UrGYcz_Ab-HTo4-B.png)  


```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(ratio='minority')
X_sm, y_sm = smote.fit_sample(X, y)
```

#### There are a variety of other methods in the imblearn package for both undersampling(Cluster Centroids, NearMiss, etc.) and oversampling(ADASYN and bSMOTE) that you can check out.

* For more about [**imblearn**](https://imbalanced-learn.readthedocs.io/en/stable/index.html)  

