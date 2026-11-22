import pandas as pd

df = pd.read_excel("online_retail_II.xlsx")

print("Dataset loaded successfully!")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())
print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())# Remove duplicate records
df = df.drop_duplicates()

# Check missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Remove rows where important fields are missing
df = df.dropna(subset=["Invoice", "Customer ID", "Description"])

# Display cleaned dataset
print("\nDataset after cleaning:")
print(df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate total sales
df["TotalPrice"] = df["Quantity"] * df["Price"]

# Top 10 best-selling products
top_products = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
top_products.plot(kind="bar")
plt.title("Top 10 Best-Selling Products")
plt.savefig("graphs/top_10_customers.png")
plt.savefig("graphs/top_10_products.png")
plt.xlabel("Product")
plt.ylabel("Quantity Sold")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

top_customers = df.groupby("Customer ID")["TotalPrice"].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 5))
top_customers.plot(kind="bar")
plt.title("Top 10 Customers by Spending")
plt.xlabel("Customer ID")
plt.ylabel("Total Spending")
plt.tight_layout()
plt.show()
# Convert InvoiceDate to datetime
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Extract month
df["Month"] = df["InvoiceDate"].dt.to_period("M")

monthly_sales = df.groupby("Month")["TotalPrice"].sum()

plt.figure(figsize=(14, 6))
monthly_sales.plot(kind="line", marker="o")

plt.title("Monthly Sales Trend")
plt.savefig("graphs/monthly_sales_trend.png")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
df = df[~df["Invoice"].astype(str).str.startswith("C")]
df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

# Set reference date
reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

# Calculate RFM values
rfm = df.groupby("Customer ID").agg({
    "InvoiceDate": lambda x: (reference_date - x.max()).days,
    "Invoice": "nunique",
    "TotalPrice": "sum"
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

print("\nRFM Analysis:")
print(rfm.head())

print("\nRFM Summary:")
print(rfm.describe())
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

rfm_features = rfm[["Recency", "Frequency", "Monetary"]].copy()

import numpy as np

rfm_features = np.log1p(rfm_features)

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_features)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)

rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

import os
import joblib

# Create models folder
os.makedirs("models", exist_ok=True)

# Save trained machine learning models
joblib.dump(kmeans, "models/kmeans_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Models saved successfully!")

print("\nCustomer Segmentation Results:")
print(rfm.head(10))

print("\nNumber of customers in each cluster:")
print(rfm["Cluster"].value_counts().sort_index())
plt.figure(figsize=(10, 6))

sns.scatterplot(
    x=rfm["Frequency"],
    y=rfm["Monetary"],
    hue=rfm["Cluster"],
    palette="viridis",
    s=60
)

plt.title("Customer Segmentation using K-Means")
plt.savefig("graphs/customer_segmentation.png")
plt.xlabel("Purchase Frequency")
plt.ylabel("Total Spending")
plt.legend(title="Customer Cluster")
plt.tight_layout()
plt.show()
from sklearn.metrics import silhouette_score

# Evaluate K-Means clustering
score = silhouette_score(rfm_scaled, rfm["Cluster"])

print("\nModel Evaluation:")
print("Silhouette Score:", round(score, 4))

if score > 0.7:
    print("Excellent clustering!")
elif score > 0.5:
    print("Good clustering!")
elif score > 0.25:
    print("Moderate clustering.")
else:
    print("Weak clustering. Improvement may be needed.")
    # Analyze customer clusters
cluster_analysis = rfm.groupby("Cluster").agg({
    "Recency": "mean",
    "Frequency": "mean",
    "Monetary": "mean",
    "Cluster": "count"
})

cluster_analysis.rename(
    columns={"Cluster": "Customer_Count"},
    inplace=True
)

print("\nCustomer Cluster Analysis:")
print(cluster_analysis.round(2))
print("\n========== BUSINESS RECOMMENDATIONS ==========")

high_value_cluster = cluster_analysis["Monetary"].idxmax()
frequent_cluster = cluster_analysis["Frequency"].idxmax()
at_risk_cluster = cluster_analysis["Recency"].idxmax()

print(f"""
1. High-Value Customers (Cluster {high_value_cluster})
   - Provide exclusive discounts and loyalty rewards.
   - Offer personalized product recommendations.
   - Encourage repeat purchases.

2. Frequent Customers (Cluster {frequent_cluster})
   - Introduce membership and reward programs.
   - Offer early access to new products.
   - Maintain customer engagement.

3. At-Risk Customers (Cluster {at_risk_cluster})
   - Send personalized re-engagement emails.
   - Provide limited-time comeback offers.
   - Recommend products based on past purchases.

4. Overall Business Strategy
   - Improve customer retention.
   - Personalize marketing campaigns.
   - Increase customer lifetime value.
""")
rfm.to_csv("customer_segmentation_results.csv")

cluster_analysis.to_csv("cluster_analysis.csv")

print("\n================================")
print("PROJECT COMPLETED SUCCESSFULLY!")
print("================================")

print("Customer segmentation results saved.")
print("Cluster analysis saved.")