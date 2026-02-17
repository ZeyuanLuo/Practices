import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签，如果在非中文环境可注释掉或改为 'Arial'
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

def load_and_inspect(file_path, name):
    print(f"--- Loading {name} ---")
    df = pd.read_csv(file_path)
    print(f"Shape: {df.shape}")
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nDescriptive Statistics:\n", df.describe())
    print("-" * 30)
    return df

# 1. 加载数据
# 请确保文件名与你的本地文件路径一致
file1 = 'ketchup_survey_with_scores.csv'
file2 = 'survey_data_gemini.csv'

df1 = load_and_inspect(file1, "Ketchup Survey Data")
df2 = load_and_inspect(file2, "Gemini Survey Data")

# 2. 数据清洗/预处理
# 移除ID列，因为对分析没有帮助
if 'RespondentID' in df1.columns:
    df1 = df1.drop(columns=['RespondentID'])
if 'Respondent_ID' in df2.columns:
    df2 = df2.drop(columns=['Respondent_ID'])

# 3. 可视化分析

# --- 第一部分：Ketchup Survey (df1) 分析 ---
print("\nVisualizing Ketchup Survey Data...")

# 3.1 数值型变量分布 (Age, Willingness, Scores)
num_cols_df1 = ['Age', 'Health-Conscious Condiment Use', 'Price-Sensitive Buying Style', 'Willingness_to_Buy_Heinz']
plt.figure(figsize=(15, 10))
for i, col in enumerate(num_cols_df1):
    plt.subplot(2, 2, i + 1)
    sns.histplot(df1[col], kde=True, bins=20)
    plt.title(f'Distribution of {col} (DF1)')
plt.tight_layout()
plt.show()

# 3.2 分类变量分布 (Gender, Income, Primary Segment)
cat_cols_df1 = ['Gender', 'Monthly Household Income', 'Primary Segment']
plt.figure(figsize=(18, 5))
for i, col in enumerate(cat_cols_df1):
    plt.subplot(1, 3, i + 1)
    sns.countplot(x=col, data=df1, palette='viridis')
    plt.title(f'Count of {col} (DF1)')
    if col == 'Primary Segment':
        plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3.3 相关性热力图
plt.figure(figsize=(10, 8))
# 只选择数值列进行相关性分析
numeric_df1 = df1.select_dtypes(include=[np.number])
sns.heatmap(numeric_df1.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix - Ketchup Survey (DF1)')
plt.show()

# 3.4 细分群体与购买意愿的关系 (Boxplot)
plt.figure(figsize=(12, 6))
sns.boxplot(x='Primary Segment', y='Willingness_to_Buy_Heinz', data=df1)
plt.title('Willingness to Buy by Primary Segment')
plt.show()


# --- 第二部分：Gemini Survey (df2) 分析 ---
print("\nVisualizing Gemini Survey Data...")

# 3.5 数值型变量分布 (Age, Q4, Q5, Q6)
# 注意：列名与df1不同
cols_to_plot_df2 = ['Q1_Age', 'Q4_Health_Focus', 'Q5_Price_Sensitivity', 'Q6_Willingness_Heinz']
plt.figure(figsize=(15, 10))
for i, col in enumerate(cols_to_plot_df2):
    plt.subplot(2, 2, i + 1)
    sns.histplot(df2[col], kde=True, bins=20, color='orange')
    plt.title(f'Distribution of {col} (DF2)')
plt.tight_layout()
plt.show()

# 3.6 相关性热力图 (Gemini)
plt.figure(figsize=(10, 8))
sns.heatmap(df2.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix - Gemini Survey (DF2)')
plt.show()

# --- 第三部分：对比两个数据集的关键指标 ---
# 比较 "Age" 和 "Willingness to Buy"
# 需要统一列名以便合并绘图
comp_df1 = df1[['Age', 'Willingness_to_Buy_Heinz']].copy()
comp_df1['Dataset'] = 'Ketchup Survey'
comp_df1.columns = ['Age', 'Willingness', 'Dataset']

comp_df2 = df2[['Q1_Age', 'Q6_Willingness_Heinz']].copy()
comp_df2['Dataset'] = 'Gemini Survey'
comp_df2.columns = ['Age', 'Willingness', 'Dataset']

combined_df = pd.concat([comp_df1, comp_df2], axis=0)

plt.figure(figsize=(14, 6))

# 对比年龄分布
plt.subplot(1, 2, 1)
sns.kdeplot(data=combined_df, x='Age', hue='Dataset', fill=True, common_norm=False)
plt.title('Age Distribution Comparison')

# 对比购买意愿分布
plt.subplot(1, 2, 2)
sns.boxplot(x='Dataset', y='Willingness', data=combined_df)
plt.title('Willingness to Buy Comparison')

plt.tight_layout()
plt.show()

print("Analysis Complete.")