# K-Means Clustering Project: University Classification

## Project Overview
In this project, we use **K-Means Clustering** to group universities into **Private** and **Public** categories.  

Even though the dataset includes labels for whether a university is private or public, K-Means is an **unsupervised algorithm**, so we do **not use these labels during clustering**. The labels are only used at the end to evaluate how well the clustering performed.

The dataset has **777 observations** on the following 18 variables:

- **Private**: Yes/No, indicates if the university is private  
- **Apps**: Number of applications received  
- **Accept**: Number of applications accepted  
- **Enroll**: Number of new students enrolled  
- **Top10perc**: Percent of new students from top 10% of H.S. class  
- **Top25perc**: Percent of new students from top 25% of H.S. class  
- **F.Undergrad**: Number of fulltime undergraduates  
- **P.Undergrad**: Number of parttime undergraduates  
- **Outstate**: Out-of-state tuition  
- **Room.Board**: Room and board costs  
- **Books**: Estimated book costs  
- **Personal**: Estimated personal spending  
- **PhD**: Percent of faculty with Ph.D.  
- **Terminal**: Percent of faculty with terminal degree  
- **S.F.Ratio**: Student/faculty ratio  
- **perc.alumni**: Percent of alumni who donate  
- **Expend**: Instructional expenditure per student  
- **Grad.Rate**: Graduation rate  

---

## Libraries Used
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
```

## Data Exploration

Load the dataset and inspect it:
```python
data = pd.read_csv('College_Data', index_col=0)
data.head()
data.info()
data.describe()
```

## Exploratory Data Analysis (EDA)

Visualize relationships in the data:

- Grad.Rate vs Room.Board colored by Private
- F.Undergrad vs Outstate colored by Private
- Histograms for Out-of-state tuition and Grad.Rate
```python
sns.set_style('whitegrid')
sns.scatterplot(x='Room.Board', y='Grad.Rate', data=data, hue='Private', palette='coolwarm')
sns.scatterplot(x='Outstate', y='F.Undergrad', data=data, hue='Private', palette='coolwarm')

# Outstate histogram
sns.set_style('darkgrid')
data[data['Private']=='Yes']['Outstate'].plot(kind='hist', alpha=0.2, color='blue', bins=20)
data[data['Private']=='No']['Outstate'].plot(kind='hist', alpha=0.2, color='red', bins=20)
plt.xlabel('Outstate')

# Grad.Rate histogram
plt.figure(figsize=(10,6))
data[data['Private']=='Yes']['Grad.Rate'].plot(kind='hist', alpha=0.2, color='blue', bins=20)
data[data['Private']=='No']['Grad.Rate'].plot(kind='hist', alpha=0.2, color='red', bins=20)
plt.xlabel('Grad.Rate')

# Fix graduation rates above 100%
data.loc['Cazenovia College', 'Grad.Rate'] = 100
```

## K-Means Clustering

Create and fit the K-Means model:
```python
km = KMeans(n_clusters=2)
km.fit(data.drop('Private', axis=1))
```

Cluster Centers:
```python
km.cluster_centers_
```

## Evaluation

Although we normally don’t have labels in unsupervised learning, we can use them here to evaluate performance:
```python
def convert(cluster):
    return 1 if cluster == 'Yes' else 0

data['Cluster'] = data['Private'].apply(convert)
```

Compare clusters to actual labels:
```python
print(confusion_matrix(data['Cluster'], km.labels_))
print(classification_report(data['Cluster'], km.labels_))
```

Example output:
```python
[[ 74 138]
 [ 34 531]]
Accuracy: 78%
Precision & Recall: Reasonably good
```

This shows that K-Means effectively clusters universities into Private and Public groups using only the features.

## Conclusion

- K-Means clustering can discover natural groupings in unlabeled data.
- In this dataset, K-Means successfully separated universities into two clusters that mostly correspond to Private and Public schools.
- This project demonstrates the power of unsupervised learning for data segmentation and exploration.
