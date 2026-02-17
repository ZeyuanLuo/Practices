import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# read Excel files
def read_excel_data(file_path):
    """
    read an Excel with several sheets
    """
    try:
        # read all the sheet
        orders_df = pd.read_excel(file_path, sheet_name='Orders')
        returns_df = pd.read_excel(file_path, sheet_name='Returns')
        people_df = pd.read_excel(file_path, sheet_name='People')

        print("Successfully Read the Data! ")
        return orders_df, returns_df, people_df
    except Exception as e:
        print(f"Error when reading the Data: {e}")
        return None, None, None


# data quality analysis
def data_quality_analysis(orders_df, returns_df, people_df):
    """
    analyse the data quality including datashape, missing values, check for duplicate values
    """
    print("=" * 50)
    print("Data Quality Analysis")
    print("=" * 50)

    # orders table datashape
    print("\n1. Order Table Data Quality:")
    print(f"Data shape: {orders_df.shape}")
    print(f"Data type:\n{orders_df.dtypes}")

    # missing values
    missing_orders = orders_df.isnull().sum()
    print(f"\nMissing Value Statistics:\n{missing_orders[missing_orders > 0]}")

    # check for duplicate values
    duplicate_orders = orders_df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_orders}")

    # returns table datashape
    print("\n2. Data Quality of Returns Form:")
    print(f"Data Shape: {returns_df.shape}")
    missing_returns = returns_df.isnull().sum()
    print(f"Missing Value Statistics:\n{missing_returns[missing_returns > 0]}")

    # People table datashape
    print("\n3. Data Quality of People Form:")
    print(f"Data Shape: {people_df.shape}")
    missing_people = people_df.isnull().sum()
    print(f"Missing Value Statistics:\n{missing_people[missing_people > 0]}")


# Descriptive Statistical Analysis of Order Data
def descriptive_analysis(orders_df):
    """
    Descriptive Statistical Analysis of Order Data
    (Usually here you should write the Function logic... but I'm lazy LOL)
    (Not only Python Codes, but all codes should always with Commentary OR nobody will understand)
    """
    print("\n" + "=" * 50)
    print("Descriptive statistical analysis")
    print("=" * 50)

    # Descriptive statistics for numerical variables
    numeric_columns = ['Sales', 'Quantity', 'Discount', 'Profit', 'Shipping Cost']
    numeric_stats = orders_df[numeric_columns].describe()
    print("Descriptive Statistics for Numeric Variables:")
    print(numeric_stats)

    # Add analysis of coefficient of variation
    print("\nCoefficient of Variation Analysis (Standard Deviation/Mean):")
    for col in numeric_columns:
        cv = orders_df[col].std() / orders_df[col].mean()
        print(f"{col}: {cv:.3f}")

    # Analysis of Categorical Variables
    categorical_columns = ['Ship Mode', 'Segment', 'Category', 'Sub-Category', 'Order Priority', 'Region', 'Market']
    print("\nDistribution of categorical variables:")
    for col in categorical_columns:
        if col in orders_df.columns:
            value_counts = orders_df[col].value_counts()
            print(f"\n{col}Distribution:")
            print(value_counts)


# Sales Performance Analysis
def sales_performance_analysis(orders_df):
    """
    Sales Performance Analysis
    """
    print("\n" + "=" * 50)
    print("Sales Performance Analysis")
    print("=" * 50)

    # Analysis by category
    category_performance = orders_df.groupby('Category').agg({
        'Sales': ['sum', 'mean', 'std'],
        'Profit': ['sum', 'mean', 'std'],
        'Quantity': 'sum',
        'Order ID': 'count'
    }).round(2)
    print("Analysis by Product Category:")
    print(category_performance)

    # Analysis by region
    region_performance = orders_df.groupby('Region').agg({
        'Sales': ['sum', 'mean'],
        'Profit': ['sum', 'mean'],
        'Order ID': 'count'
    }).round(2)
    print("\nAnalysis by region:")
    print(region_performance)

    # Customer Segmentation Analysis
    segment_performance = orders_df.groupby('Segment').agg({
        'Sales': ['sum', 'mean'],
        'Profit': ['sum', 'mean'],
        'Order ID': 'count'
    }).round(2)
    print("\nCustomer Segmentation Analysis:")
    print(segment_performance)


# Returns Analysis
def returns_analysis(orders_df, returns_df):
    """
    退货数据的统计分析
    """
    print("\n" + "=" * 50)
    print("Returns Analysis")
    print("=" * 50)

    # Calculate the return rate
    total_orders = orders_df['Order ID'].nunique()
    returned_orders = returns_df['Order ID'].nunique()
    return_rate = returned_orders / total_orders * 100

    print(f"Total number of orders: {total_orders}")
    print(f"Total number of Returns: {returned_orders}")
    print(f"Returning Rate: {return_rate:.2f}%")

    # Merge data to analyse return order characteristics
    orders_df['Returned'] = orders_df['Order ID'].isin(returns_df['Order ID'])

    if 'Returned' in orders_df.columns:
        # Analysing the characteristics of return orders
        returned_analysis = orders_df.groupby('Returned').agg({
            'Sales': ['mean', 'std'],
            'Profit': ['mean', 'std'],
            'Discount': ['mean', 'std'],
            'Order ID': 'count'
        }).round(2)
        print("\nComparison of Returned vs Non-Returned Orders:")
        print(returned_analysis)


# Time Series Analysis
def time_series_analysis(orders_df):
    """
    Time Series Analysis
    """
    print("\n" + "=" * 50)
    print("Time Series Analysis")
    print("=" * 50)

    # Convert date format
    orders_df['Order Date'] = pd.to_datetime(orders_df['Order Date'])
    orders_df['Ship Date'] = pd.to_datetime(orders_df['Ship Date'])

    # Calculate transit time
    orders_df['Shipping Days'] = (orders_df['Ship Date'] - orders_df['Order Date']).dt.days

    # Monthly analysis of sales trends
    orders_df['Order Month'] = orders_df['Order Date'].dt.to_period('M')
    monthly_sales = orders_df.groupby('Order Month').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': 'count'
    }).round(2)

    print("Monthly Sales Trends:")
    print(monthly_sales)



    # Transportation Time Analysis
    shipping_stats = orders_df['Shipping Days'].describe()
    print(f"\nTransportation Time Statistics:")
    print(shipping_stats)


# Outlier Detection
def outlier_detection(orders_df):
    """
    Outlier Detection Using Statistic Methods
    """
    print("\n" + "=" * 50)
    print("Outlier")
    print("=" * 50)

    numeric_columns = ['Sales', 'Profit', 'Discount']

    for col in numeric_columns:
        if col in orders_df.columns:
            Q1 = orders_df[col].quantile(0.25)
            Q3 = orders_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = orders_df[(orders_df[col] < lower_bound) | (orders_df[col] > upper_bound)]
            outlier_percentage = len(outliers) / len(orders_df) * 100

            print(f"{col}Outlier: {len(outliers)} ({outlier_percentage:.2f}%)")


# Generate visual reports
def create_visualizations(orders_df, returns_df):
    """
    Create statistical visualisation charts
    """
    print("\nGenerate visualisation charts...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Sales distribution
    sales_data = orders_df['Sales'].dropna()
    # Calculate the appropriate display range
    q95 = sales_data.quantile(0.95)

    # Plot histograms and KDE, but restrict the display range.
    axes[0, 0].hist(sales_data, bins=30, alpha=0.3, color='lightblue', density=True, label='直方图')

    # Create KDE but restrict the display area
    sales_data.plot.kde(ax=axes[0, 0], color='red', linewidth=2, label='密度曲线')
    axes[0, 0].set_xlim(0, q95)  # Restricting the x-axis range

    axes[0, 0].set_title('Sales distribution')
    axes[0, 0].set_xlabel('Sales')
    axes[0, 0].set_ylabel('frequency')

    # 2. Product Category Sales Contribution
    category_sales = orders_df.groupby('Category')['Sales'].sum()
    axes[0, 1].pie(category_sales.values, labels=category_sales.index, autopct='%1.1f%%')
    axes[0, 1].set_title('Product Category Sales Contribution')

    # 3. Scatter Plot of Profit and Discount Relationship
    axes[1, 0].scatter(orders_df['Discount'], orders_df['Profit'], alpha=0.5)
    axes[1, 0].set_xlabel('Discount')
    axes[1, 0].set_ylabel('Profile')
    axes[1, 0].set_title('The Relationship Between Profit and Discounts')

    # 4. Sales performance by region
    region_sales = orders_df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
    axes[1, 1].bar(region_sales.index, region_sales.values)
    axes[1, 1].set_title('Sales figures by region')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


# Principal Analysis Function
def comprehensive_analysis(file_path):
    """
    Conduct comprehensive statistical analysis
    """
    # read data
    orders_df, returns_df, people_df = read_excel_data(file_path)

    if orders_df is None:
        print("Unable to read data. Please check the file path.")
        return

    # Perform various analyses
    data_quality_analysis(orders_df, returns_df, people_df)
    descriptive_analysis(orders_df)
    sales_performance_analysis(orders_df)
    returns_analysis(orders_df, returns_df)
    time_series_analysis(orders_df)
    outlier_detection(orders_df)

    # Generate visualisation
    create_visualizations(orders_df, returns_df)

    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)


# Using examples
if __name__ == "__main__":
    # Please replace the file path with your actual file path.
    file_path = "D:\PythonProjects\GlobalSuperStore_StudentData_AnalysisPractice\Global Superstore student UK 2025.xlsx"
    comprehensive_analysis(file_path)