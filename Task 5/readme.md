✅ Task 5 – Customer Clustering (K-Means)
🎯 Prodigy InfoTech – Machine Learning Internship

📌 Objective
Analyze customer behavior and divide them into similar groups (clusters) using K-Means clustering.

📂 Dataset
car_insurance.csv

✔ Numeric columns are automatically detected
✔ Missing values are handled
✔ Clustering is based on continuous features

📌 Steps Performed

1️⃣ Import Libraries

pandas

numpy

sklearn (StandardScaler, KMeans)

matplotlib

seaborn

2️⃣ Data Preprocessing

Select only numeric columns

Remove missing values using dropna()

Standardize features using StandardScaler

3️⃣ Elbow Method

Used to find the optimal number of clusters

WCSS = Within-Cluster Sum of Squares

Graph plotted for k = 1 to 10

4️⃣ Final Model (K = 3)

Applied K-Means clustering

Assigned each customer a cluster label

Analyzed patterns inside cluster groups

📊 Visualizations (Screenshots Needed)
📸 Upload these 3 screenshots to your GitHub folder:

Dataset Head

Elbow Method Graph

Cluster Results Table (df.head())

🧪 Final Code Used

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load your CSV file
df = pd.read_csv("Customer_Behavior.csv")   # change name if needed

# Select only numeric features
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Drop missing values
numeric_df = numeric_df.dropna()

# Scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)

# Elbow Method to find optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method to Determine Optimal K")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Final Model (K = 3)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

df["Cluster"] = clusters
df.head()
