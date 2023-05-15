import pandas as pd
import csv
import matplotlib.pyplot as plp
import seaborn as sns
from sklearn.cluster import KMeans

# data = pd.read_csv('project131output(2).csv')
# mass = data['Mass']
# gravity = data['Gravity(in m/s2)']
# radius = data['Radius']

# mass.sort_values()
# radius.sort_values()
# gravity.sort_values()

# plp.figure(figsize=(15,7))
# plp.scatter(x=mass, y=radius)
# plp.xlabel('Mass Of Planet')
# plp.ylabel('Radius Of Planet')
# plp.show()

# plp.figure(figsize=(15,7))
# plp.scatter(x=mass, y=gravity)
# plp.xlabel('Mass Of Planet')
# plp.ylabel('Gravity Of Planet')
# plp.show()

data = pd.read_csv('project131output(2).csv')
data2 = data.dropna(subset=['Mass', 'Radius'])

mass = data2['Mass']
radius = data2['Radius']

x = []
for i, v in enumerate(mass):
    temp2 = data2.get([radius[i],v])
    x.append(temp2)
  
wcss = []
for i in range(1,11):
  kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
  kmeans.fit(x)
  wcss.append(kmeans.inertia_)

plp.figure(figsize=(10,5))
sns.lineplot(x=range(1,11), y=wcss, marker='o', color='red')
plp.xlabel('No, Of Clusters')
plp.ylabel('WCSS')
plp.title('Elbow Method')
plp.show()

