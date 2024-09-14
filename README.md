# IVF<sup>2</sup>
## Introduction
This framework provides a flexible and efficient solution for approximate nearest neighbor search (ANNS) on high-dimensional data under attribute constraint. We cluster the data, create an inverted file, and further create an inverted file for each cluster based on the attribute values. This implementation is based on the algorithm described in our paper titled　IVF<sup>2</sup>: A Flexible and Efficient Algorithm for Approximate Nearest Neighbor Search under Attribute Constraint.

## How to use
### 1. Making index and dataset folsers
Create a folder for storing the index and a folder for storing the dataset.
```sh
mkdir index dataset
```

### 2. Preparing the dataset
Create baseset.txt, queryset.txt, and groundtruthset.txt inside the dataset folder as follows

#### The dataset.txt and queryset.txt
The baseset.txt queryset.txt should be stored in the following format:
- **First line**: The number of data points.
- **Second line**: The number of attribute types
- **Third line and onwards**: each line represents one data point. Each line consists of:
   - Attribute values (one or more, depending on the number of attributes).
   - Vector data (a list of numerical values representing the data point in high-dimensional space).

#### The groundtruthset.txt
The groundtruthset.txt should be stored Each line corresponds to a line in queryset.txt, and each line lists the data IDs in order of proximity to the query.

### 3. Compile and run
#### Building Index
```sh
cd src
g++ -O3 -o build build.cpp io.cpp utils.cpp -lfaiss -mavx512f -march=native
./build ../dataset/baseset.txt index d nc th kg
```
- **d**　is the dimensionality of the data vectors.
- **nc** controls the number of clusters in clustering.
- **th**　controls the threshold for the minimum number of data points matching a specific attribute combination in each cluster required to create a graph.
- **kg**　controls the degree of each node when creating the graph.

#### Searching Index
```sh
cd src
g++ -O3 -o search search.cpp io.cpp utils.cpp -lfaiss -mavx512f -march=native
./search ../dataset/baseset.txt ../dataset/queryset.txt index ../dataset/groundtruthset.txt d nc ks np
```
- **d**　is the dimensionality of the vector data.
- **nc** is the same number as nc during the build.
- **ks**　controls the number of searching result.
- **nc**　controls the number of searching clusters, which affects the recall.
