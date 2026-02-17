import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Read the csv file
df = pd.read_csv('Adidas_US_Sales_Dataset.csv', sep=';')

# Removing spaces from currency, percentages and numerical values
def clean_currency(value):
    if isinstance(value, str):
        return float(value.replace('$', '').replace(' ', '').replace(',', ''))
    return value

def clean_percent(value):
    if isinstance(value, str):
        return float(value.replace('%', '').replace(' ', '')) / 100
    return value

def clean_units(value):
    if isinstance(value, str):
        return float(value.replace(' ', '').replace(',', ''))
    return value

# Clean the data into float
df['invoicedate'] = pd.to_datetime(df['invoicedate'], format='%d/%m/%Y')
df['priceperunit'] = df['priceperunit'].apply(clean_currency)
df['unitssold'] = df['unitssold'].apply(clean_units)
df['totalsales'] = df['totalsales'].apply(clean_currency)
df['operatingprofit'] = df['operatingprofit'].apply(clean_currency)
df['operatingmargin'] = df['operatingmargin'].apply(clean_percent)

# Then analyse the insights of the data
# Descriptive data analysis
desc_stats = df[['priceperunit', 'unitssold', 'totalsales', 'operatingprofit', 'operatingmargin']].describe()
print("Numerical Summary:\n", desc_stats)

# Correlation Analysis
correlation_matrix = df[['priceperunit', 'unitssold', 'totalsales', 'operatingprofit', 'operatingmargin']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Distribution and KDE
plt.figure(figsize=(10, 6))
sns.histplot(df['totalsales'], kde=True, bins=50, color='skyblue')
plt.title('Distribution of Total Sales')
plt.xlabel('Total Sales ($)')
plt.show()

# Retail analysis, count all the retailers
retailer_counts = df['retailer'].value_counts()
plt.figure(figsize=(10, 6))
retailer_counts.plot(kind='bar', color='teal')
plt.title('Number of Data Records by Retailer')
plt.ylabel('Count')
plt.show()

# For sales insights, descending
product_sales = df.groupby('product')['totalsales'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(10, 5))
# sns.barplot(data=product_sales, x='totalsales', y='product', palette='viridis') #This is the old version of sns
sns.barplot(data=product_sales, x='totalsales', y='product', hue='product', palette='viridis', legend=False) #By deepseek
plt.title('Total Sales by Product')
plt.xlabel('Total Sales ($)')
plt.show()

# Monthly Sales, firstly get the month and then groupby the total sales
df['month_year'] = df['invoicedate'].dt.to_period('M')
monthly_sales = df.groupby('month_year')['totalsales'].sum().reset_index()
monthly_sales['month_year'] = monthly_sales['month_year'].astype(str)
plt.figure(figsize=(12, 5))
sns.lineplot(data=monthly_sales, x='month_year', y='totalsales', marker='o', color='blue')
plt.xticks(rotation=45)
plt.title('Monthly Sales Trend from 2020 - 2021')
plt.xlabel('Monthly Sales ($)')
plt.show()

# Sales by method and region
region_method = df.groupby(['region', 'salesmethod'])['totalsales'].sum().unstack()
region_method.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Sales by Region and Method')
plt.ylabel('Total Sales ($)')
plt.show()

# Price per Unit VS Sales, use scatter plots and put density inside
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='priceperunit', y='unitssold', alpha=0.5, hue='product')
plt.title('Price per Unit vs Units Sold')
plt.show()