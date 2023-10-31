# K-Means Clustering of FIFA Players ⚽🥅

<div align="left">

[![Python](https://img.shields.io/badge/Python-3670A0?style=flat-square&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-%23013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%2300768B?style=flat-square&logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-%2318BDBB?style=flat-square&logo=seaborn&logoColor=white)](https://seaborn.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

<a href="https://colab.research.google.com/drive/1TtSZazir1GKaPp7jG_bVY4R6qb5ldP9E">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Code">
</a>

</div>


## Overview

This repository contains Python code for performing K-Means clustering on FIFA 22 player data using Google Colab and Scikit-Learn. The goal is to cluster FIFA players based on their overall performance, potential, market value, wage, and age.

## Prerequisites

Before running the code, make sure you have the following prerequisites installed:

- **Google Colab**: This code is designed to run on Google Colab, so you'll need a Google account.

- **Python Libraries**: You'll need the following Python libraries, which can be installed using pip:
   - `google.colab` for authentication.
   - `pandas` for data manipulation.
   - `numpy` for numerical operations.
   - `matplotlib` for data visualization.
   - `sklearn.preprocessing` for data preprocessing.
   - `sklearn.decomposition` for PCA (Principal Component Analysis).
   - `seaborn` for creating line plots.
   
- **Google Cloud Project**: You should have a Google Cloud project with BigQuery enabled. Replace `project_id` in the code with your project's ID.

## Code Overview

### 1. Data Loading

The code starts by authenticating your Google account and then loading FIFA 22 player data from Google BigQuery. The selected columns include the player's short name, overall rating, potential, market value, wage, and age.

### 2. K-Means Clustering Implementation

The K-Means clustering is implemented from scratch using Python and NumPy. The key functions include:

- `random_centroid(data, k)`: Generates random initial centroids for the clusters.
- `get_labels(data, centroid)`: Assigns each data point to the nearest cluster based on centroid distance.
- `plot_clusters(data, labels, centroids, iteration)`: Transforms the data to 2D using PCA and plots the clusters. This is done for each iteration of the K-Means algorithm.

![cluster](https://github.com/sagarv2522/fifa-cluster/blob/48afb75b213af56f39c07c0fe116381c9e225729/images/Screenshot%202023-09-01%20193511.png)

### 3. K-Means Using Scikit-Learn

Another method for K-Means clustering is provided using Scikit-Learn. It calculates and plots the inertia and silhouette score for different numbers of clusters (k) to help you determine the optimal number of clusters.

![elbow image](https://github.com/sagarv2522/fifa-cluster/blob/48afb75b213af56f39c07c0fe116381c9e225729/images/Screenshot%202023-09-01%20193606.png)
![sill image](https://github.com/sagarv2522/fifa-cluster/blob/48afb75b213af56f39c07c0fe116381c9e225729/images/Screenshot%202023-09-01%20193630.png)

### 4. Final Clustering

The code performs K-Means clustering with `k=3` using Scikit-Learn and fits the model to the player data.

## Usage

1. Make sure you have all the prerequisites installed and your Google Cloud project set up.

2. Run the code step by step in a Jupyter Notebook or Google Colab.

3. Analyze the results and visualizations to understand how FIFA players are clustered based on their attributes.

Feel free to modify the code and experiment with different values of `k` or add more features to the clustering process. Enjoy exploring the FIFA player data! ⚽📊
