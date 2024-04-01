#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing necessary Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2  # Change to f_classif for classification
from sklearn.preprocessing import LabelEncoder


# In[2]:


# Loading the data file 

data = pd.read_csv(r"C:\Users\amark\OneDrive\Desktop\HERO VIRED\Python\bank_marketing.csv")


# In[3]:


data


# In[4]:


# Analysing the basic Info About the Data 

def basic_info():
    # Total number of columns and then Number of categorical and numerical columns
    total_columns= data.shape[1]
    cat_columns=data.select_dtypes("object").shape[1]
    cont_columns=data.select_dtypes("int64","float64").shape[1]
    data_shape=data.shape
    
    # Creating list of categorical and numerical columns
    lst_cat_col=data.select_dtypes("object").columns
    lst_cont_col= data.select_dtypes("int64","float64").columns
    
    
    print("Shape of the dataset is {}".format(data.shape))
    
    print("---------------------------------------------------------------")
    
    print("Total number of columns are: {}".format(total_columns))
    
    print("Number of  categorical columns are: {}".format(cat_columns))
    
    print("Number of continous columns are: {}".format(cont_columns))
    
    print("---------------------------------------------------------------")
    
    print("List of Continous columns\n")
    print(lst_cont_col)
    
    print("---------------------------------------------------------------")
    
    print("List of Categorical columns\n")
    print(lst_cat_col)
    
    print("---------------------------------------------------------------")
    
    display(data.describe().T)
    
    # List of categorical columns
#categorical_columns = ['marital',"targeted","default","housing","loan","contact","month","poutcome","response","job","education","year"]

# Iterate through each categorical column and print unique values
#for column in categorical_columns:
    ####print()
        
    
    
        
basic_info()


# ## we need to perform below column operation before going any further
# 
# 1) Convert age(folat64) ---> int64
# 
# 2) split(jobedu) ----> "job" , "education"
# 
# 3) split(month) ----> month , year
# 
# 4) convert all the data in "duration" to "minutes"
# 
# 5) dropping customer id column 
# 

# In[5]:


def col_operation():
    # age into int64
    #data['age']=data['age'].astype("int64")
    
    # splitting the "jobedu" column
    data[["job","education"]]=data.jobedu.str.split(",",expand=True)
    # dropping the "jobedu" column
    #data.drop(["jobedu"],axis=1,inplace=True) -- this code is throwing error here so had to code in next cell
    
    # splitting the "month" column
    data[["month","year"]]=data.month.str.split(",",expand=True)
    
    # splitting the duration column into duration and unit
    data[["duration","unit"]]=data.duration.str.split(" ",expand=True)
    # converting the duration column into float type
    data["duration"]=round(data["duration"].astype("float"),2)
    # replacing all the sec values and converting them into min
    data.loc[data['unit'] == 'sec', 'duration'] = data[data['unit'] == 'sec']['duration'] / 60
    # Since we have converted all the duration into min lets drop the unit column 
    data.drop("unit",axis=1,inplace=True)
    
    #dropping customer id column 
    data.drop("customerid",axis=1,inplace=True)
    


# In[6]:


# Activate the col_operation function to make the changes 
col_operation()

# dropping jobedu column 
data.drop(["jobedu"],axis=1,inplace=True)


# In[7]:


# check for the new data

data


# ## Handling Missing values

# In[8]:


# chaeck missing values

data.isna().sum()


# ### As we can see "age" and "response" have missing values here
# 
# ### In order to impute the missing values we need to check two aspect
# 1) check the data type of the column that has missing values 
# (here "age" is an int and "response" is an object)
# 
# 2) check if the column with missing value has any outliers (specifically continous dtype)

# In[9]:


def age_boxplot():
    boxplot1 = sns.boxplot(y=data["age"])
    return boxplot1


# In[10]:


age_boxplot()


# In[11]:


# as we can see that we are having outliers in age column we will impute missing values with "median"
# And for "response" we will impute it with "mode"

def impute_missing_values():
    
    # imputing numerical columns
    age_median=data["age"].median()
    data['age']=data["age"].fillna(age_median)
    
    # Imputing categorical colums
    response_mode=stats.mode(data['response'])[0][0] 
    data['response']=data["response"].fillna(response_mode)
    
    month_mode=data['month'].mode()[0]
    data['month']=data['month'].fillna(month_mode)
    
    year_mode=data['year'].mode()[0]
    data['year']=data['year'].fillna(month_mode)
    
impute_missing_values()


# In[12]:


data.isna().sum()


# # Descriptive Statistice
# 
# a. Derive summary statistics (mean, median, standard deviation) for relevant columns.
# 
# b. Examine the distribution of the target variable, indicating responses to the term deposit campaign.

# In[13]:


# Summary stats
# a. Derive summary statistics (mean, median, standard deviation) for relevant columns.

summary_cont_col=data.describe().T

display(summary_cont_col)

summary_cat_col=data.select_dtypes("object").describe().T

display(summary_cat_col)


# In[14]:


def boxplot_with_summary_statistics(data):
    """
    Visualize boxplot with summary statistics annotations for each column in the dataset
    
    Parameters:
    - data: DataFrame containing the dataset
    """
    # Create boxplot for each column
    for column in data.columns:
        if data[column].dtype in [np.int64, np.float64]:  # Check if the column is numeric
            # Create box plot
            fig = go.Figure()
            fig.add_trace(go.Box(y=data[column], name=column))
            
            # Calculate summary statistics
            stats = data[column].describe()
            
            # Add annotations for summary statistics
            fig.add_annotation(x=0, y=stats['25%'], text=f"<b>25%:</b> {stats['25%']:.2f}", showarrow=False, font=dict(size=10, color='red', family='Arial'))
            fig.add_annotation(x=0, y=stats['50%'], text=f"<b>50%:</b> {stats['50%']:.2f}", showarrow=False, font=dict(size=10, color='red', family='Arial'))
            fig.add_annotation(x=0, y=stats['75%'], text=f"<b>75%:</b> {stats['75%']:.2f}", showarrow=False, font=dict(size=10, color='red', family='Arial'))
            fig.add_annotation(x=0, y=stats['min'], text=f"<b>min:</b> {stats['min']:.2f}", showarrow=False, font=dict(size=10, color='red', family='Arial'))
            fig.add_annotation(x=0, y=stats['max'], text=f"<b>max:</b> {stats['max']:.2f}", showarrow=False, font=dict(size=10, color='red', family='Arial'))
            
            # Update layout
            fig.update_layout(title=f"Boxplot of {column} with Summary Statistics", yaxis_title=column, xaxis=dict(showticklabels=False))
            
            # Show plot
            fig.show()
boxplot_with_summary_statistics(data)


# In[15]:


#b. Examine the distribution of the target variable, indicating responses to the term deposit campaign.

plt.figure(figsize=(8, 6))
ax = sns.countplot(data=data, x='response')
plt.title("Distribution of Responses to Term Deposit Campaign")
plt.xlabel("Response")
plt.ylabel("Count")
        
# Calculate percentage of each response
total_count = len(data)
counts = data['response'].value_counts()
percentages = [(count / total_count) * 100 for count in counts]
        
# Add count and percentage annotations to each bar
for i, p in enumerate(ax.patches):
    count = p.get_height()
    percentage = percentages[i]
    ax.annotate(f'{count} ({percentage:.2f}%)', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
        
plt.show()


# ## 3. Univariate Analysis
#  a. Examine the distribution of individual key features, such as age, balance,
# and call duration.
# 
# 
#  b. Employ visual aids like histograms, box plots, and kernel density plots to
# discern patterns and outliers.

# In[16]:



# Define key features for analysis
key_features = ['age', 'balance', 'duration','salary','day','campaign','pdays','previous']  # Add more features as needed
        
# Plot histograms for each key feature
for feature in key_features:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=data, x=feature,hue=data['response'], kde=True)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()


# ## Bivariate Analysis

# In[17]:



    def bivariate_analysis():
        """
        Method to perform bivariate analysis
        """
        # Part a: Evaluate the relationship between independent variables and the target variable
        
        # Analyze the association between each independent variable and the target variable
        categorical_features = ['job', 'education', 'marital','targeted',
                                'default','housing','loan','contact','month',
                                'poutcome']  # Add more categorical features as needed
        
        for feature in categorical_features:
            plt.figure(figsize=(10, 6))
            ax = sns.countplot(data=data, x=feature, hue='response')
            plt.title(f"Relationship between {feature} and Response to Term Deposit Campaign")
            plt.xlabel(feature)
            plt.ylabel("Count")
            plt.legend(title='Response', loc='upper right')
            plt.xticks(rotation=45)
            
            # Add count annotations to each bar
            for p in ax.patches:
                ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
            
            plt.show()
        


# In[ ]:





# In[18]:


bivariate_analysis()


# ## Categorical Variables Analysis

# In[19]:


def categorical_variables_analysis():
        """
        Method to perform categorical variables analysis
        """
        # Part a: Investigate the distribution of categorical variables
        
        # Define categorical variables for analysis
        categorical_variables = ['job', 'education', 'marital']  # Add more variables as needed
        
        for variable in categorical_variables:
            plt.figure(figsize=(10, 6))
            sns.countplot(data=data, x=variable)
            plt.title(f"Distribution of {variable}")
            plt.xlabel(variable)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.show()
        
        # Part b: Assess the impact of categorical variables on campaign's success
        
        # Visualize the impact of categorical variables on campaign's success
        for variable in categorical_variables:
            plt.figure(figsize=(10, 6))
            sns.countplot(data=data, x=variable, hue='response')
            plt.title(f"Impact of {variable} on Campaign Success")
            plt.xlabel(variable)
            plt.ylabel("Count")
            plt.legend(title='Response', loc='upper right')
            plt.xticks(rotation=45)
            
            # Add count annotations to each bar
            ax = plt.gca()
            for p in ax.patches:
                ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
            
            plt.show()


# In[20]:


categorical_variables_analysis()


# In[ ]:





# ## Creating salary and age categories

# In[21]:



# Define salary categories and corresponding boundaries
salary_bins = [0, 300000/12, 600000/12, 1200000/12, float('inf')]
salary_labels = ['EWS', 'LIG', 'MIG', 'HIG']

# Create salary category variable
data['salary_category'] = pd.cut(data['salary'], bins=salary_bins, labels=salary_labels, right=False)


# In[22]:


# Define categorical variables to visualize
categorical_variables = ['job', 'education', 'marital','month','contact','salary_category']  # Add more variables as needed
        
for variable in categorical_variables:
            if variable == 'job':
                # Calculate frequency of each category
                category_counts = data[variable].value_counts()
                
                # Select top 6 categories and aggregate the rest into 'Others'
                top_categories = category_counts[:6]
                other_count = category_counts[6:].sum()
                top_categories['Others'] = other_count
                
                # Plot pie chart
                plt.figure(figsize=(8, 6))
                plt.pie(top_categories, labels=top_categories.index, autopct='%1.1f%%', startangle=140)
                plt.title(f"Distribution of {variable}")
                plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                plt.show()
            else:
                # Calculate frequency of each category
                category_counts = data[variable].value_counts()
                
                # Plot pie chart
                plt.figure(figsize=(8, 6))
                plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
                plt.title(f"Distribution of {variable}")
                plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                plt.show()


# In[ ]:





# In[23]:


# Define categories and colors
categories = ['EWS', 'LIG', 'MIG', 'HIG']
colors = ['skyblue', 'lightgreen', 'orange', 'salmon']

# Plot pie chart for each category in 'salary_category'
for category, color in zip(categories, colors):
    category_data = data[data['salary_category'] == category]
    category_counts = category_data['response'].value_counts(normalize=True) * 100  # Calculate percentage
    plt.figure(figsize=(6, 6))
    plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', colors=['lightcoral', 'lightskyblue'])
    plt.title(f'Distribution of Response for {category} Salary Category')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.show()


# ## ENCODING THE CATEGORICAL VARIABLES

# ## Corelation Analysis

# In[24]:


# Compute correlation matrix
corr_matrix = data.corr()

# Plot correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()


# ## KPI's for this campaign
# 
# 

# In[25]:


data['month'].value_counts()


# In[26]:



# Group the data by "month" and calculate the sum of the "duration" column for each month
monthly_duration_sum = data.groupby('month')['duration'].sum()/60

# Sort the results in descending order
monthly_duration_sum = monthly_duration_sum.sort_values(ascending=False)

# Print the result
print(monthly_duration_sum)


# In[27]:


# Group the data by "month" and calculate the sum of the "duration" column for each month (convert to minutes)
monthly_duration_sum = data.groupby('month')['duration'].sum() / 60

# Sort the results in descending order
monthly_duration_sum = monthly_duration_sum.sort_values(ascending=False)

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x=monthly_duration_sum.index, y=monthly_duration_sum.values, color='skyblue')
plt.title('Total Call Duration per Month')
plt.xlabel('Month')
plt.ylabel('Total Duration (minutes)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add annotations
for i, value in enumerate(monthly_duration_sum.values):
    plt.text(i, value, f'{value:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()


# In[28]:


# Group the data by "month" and "response" and calculate the sum of the "duration" column for each month and response
monthly_duration_sum_by_response = data.groupby(['month', 'response'])['duration'].sum()

# Group the data by "month" and calculate the total sum of the "duration" column for each month
monthly_duration_total = data.groupby('month')['duration'].sum()

# Calculate the percentage of 'yes' and 'no' for each month
percentage_by_response = (monthly_duration_sum_by_response / monthly_duration_total) * 100

# Filter the 'yes' percentage and sort in descending order
yes_percentage = percentage_by_response.unstack()['yes'].sort_values(ascending=False)

# Print the result
print(yes_percentage)


# In[29]:


# Group the data by "month" and "response" and calculate the sum of the "duration" column for each month and response
monthly_duration_sum_by_response = data.groupby(['month', 'response'])['duration'].sum()

# Group the data by "month" and calculate the total sum of the "duration" column for each month
monthly_duration_total = data.groupby('month')['duration'].sum()

# Calculate the percentage of 'yes' and 'no' for each month
percentage_by_response = (monthly_duration_sum_by_response / monthly_duration_total) * 100

# Filter the 'yes' percentage and sort in descending order
yes_percentage = percentage_by_response.unstack()['yes'].sort_values(ascending=False)

# Plotting
plt.figure(figsize=(10, 6))
ax = yes_percentage.plot(kind='bar', color='lightgreen')
plt.title('Percentage of "Yes" Responses per Month')
plt.xlabel('Month')
plt.ylabel('Percentage of "Yes" Responses')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add annotations
for i, value in enumerate(yes_percentage):
    ax.text(i, value, f'{value:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()


# ## Here there is no such insight that can  be brought up

# In[30]:


sns.pairplot(data)


# In[31]:


# Initialize a dictionary to store the category with the highest count of 'yes' for each column
best_categories = {}

# Iterate through each column (excluding the target variable)
for column in data.columns:
    if column != 'response':
        # Count the occurrences of 'yes' for each category in the column
        category_counts = data[data['response'] == 'yes'][column].value_counts()
        
        # Check if there are any 'yes' responses for this column
        if not category_counts.empty:
            # Get the category with the highest count of 'yes'
            best_category = category_counts.idxmax()
            best_categories[column] = best_category

# Print the category with the highest count of 'yes' for each column
for column, category in best_categories.items():
    print(f"Best category for column '{column}': '{category}'")


# In[32]:


data


# In[33]:


# Filter the dataset for only "yes" responses
yes_data = data[data['response'] == 'yes']

# Create a histogram of the 'duration' variable for 'yes' responses
plt.figure(figsize=(10, 6))
sns.histplot(data=yes_data, x='duration', bins=20, kde=True, color='skyblue')
plt.title('Distribution of Call Duration for Yes Responses')
plt.xlabel('Duration (seconds)')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Create a box plot of the 'duration' variable for 'yes' responses
plt.figure(figsize=(10, 6))
sns.boxplot(data=yes_data, y='duration', color='lightgreen')
plt.title('Box Plot of Call Duration for Yes Responses')
plt.ylabel('Duration (seconds)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[34]:


# Filter the dataset for only "no" responses
no_data = data[data['response'] == 'no']

# Create a histogram of the 'duration' variable for 'no' responses
plt.figure(figsize=(10, 6))
sns.histplot(data=no_data, x='duration', bins=20, kde=True, color='skyblue')
plt.title('Distribution of Call Duration for No Responses')
plt.xlabel('Duration (seconds)')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Create a box plot of the 'duration' variable for 'no' responses
plt.figure(figsize=(10, 6))
sns.boxplot(data=no_data, y='duration', color='lightgreen')
plt.title('Box Plot of Call Duration for No Responses')
plt.ylabel('Duration (seconds)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[35]:


# Create separate dataframes for 'yes' and 'no' responses
yes_data = data[data['response'] == 'yes']
no_data = data[data['response'] == 'no']

# Create a histogram of the 'age' variable for 'yes' responses
plt.figure(figsize=(10, 6))
sns.histplot(data=yes_data, x='age', bins=20, kde=True, color='skyblue')
plt.title('Distribution of Age for Yes Responses')
plt.xlabel('Age')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Create a histogram of the 'age' variable for 'no' responses
plt.figure(figsize=(10, 6))
sns.histplot(data=no_data, x='age', bins=20, kde=True, color='lightgreen')
plt.title('Distribution of Age for No Responses')
plt.xlabel('Age')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[36]:


# List of continuous features
continuous_features = ['age', 'salary', 'balance', 'duration']

# Create histograms for each continuous feature
for feature in continuous_features:
    # Create separate dataframes for 'yes' and 'no' responses
    yes_data = data[data['response'] == 'yes']
    no_data = data[data['response'] == 'no']

    # Create a histogram of the feature for 'yes' responses
    plt.figure(figsize=(10, 6))
    sns.histplot(data=yes_data, x=feature, bins=20, kde=True, color='skyblue')
    plt.title(f'Distribution of {feature.capitalize()} for Yes Responses')
    plt.xlabel(feature.capitalize())
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Create a histogram of the feature for 'no' responses
    plt.figure(figsize=(10, 6))
    sns.histplot(data=no_data, x=feature, bins=20, kde=True, color='lightgreen')
    plt.title(f'Distribution of {feature.capitalize()} for No Responses')
    plt.xlabel(feature.capitalize())
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


# In[ ]:





# In[37]:


# Feature Importance


# In[38]:


# sorting Categorical data


# In[39]:


data_cat= data[["marital",'targeted','default','housing','loan','contact','month','poutcome','job','education','year','response']]


# In[40]:


# Label encoding 

for col in data_cat.columns:
    le=LabelEncoder()
    data_cat[col]=le.fit_transform(data_cat[col])
    
data_cat.head()


# In[41]:


# Applying Chi - SQUARE


# In[42]:


from sklearn.feature_selection import chi2

X= data_cat.drop(columns=['response'],axis=1)
y=data_cat["response"]


# In[43]:


chi_scores =chi2(X,y)


# In[44]:


chi_scores


# In[45]:


# Higher the Chi value higher the importance
chi_values= pd.Series(chi_scores[0],index=X.columns)
chi_values.sort_values(ascending=False,inplace=True)
chi_values.plot.bar()
plt.xlabel("Features")
plt.ylabel("Chi Value (scores)")
plt.show()


# ## Higher the Chi value higher the importance

# In[46]:


# Higher the P value lower the importance
# if p value>0.5 lower the importance or neglect the column

p_values= pd.Series(chi_scores[1],index=X.columns)
p_values.sort_values(ascending=False,inplace=True)
p_values.plot.bar()
plt.xlabel("Features")
plt.ylabel("P Value (scores)")
plt.show()


# In[47]:


# Filter the dataset for only "Yes" responses
yes_data = data[data['response'] == 'yes']
# Create a bar plot of 'response' (Yes) with respect to 'contact'
plt.figure(figsize=(8, 6))
sns.countplot(data=yes_data, x='contact', palette='Set2')
plt.title('Distribution of Response (Yes) by Contact Method')
plt.xlabel('Contact Method')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[48]:


# Filter the dataset for only "Yes" responses
yes_data = data[data['response'] == 'yes']

# List of columns to visualize
columns_to_visualize = ['contact','housing', 'job', 'loan', 'poutcome', 'education', 'month', 'targeted', 'marital', 'default']

# Create subplots for each column
num_columns = len(columns_to_visualize)
num_rows = num_columns // 2 + (num_columns % 2)

plt.figure(figsize=(15, 5*num_rows))

for i, column in enumerate(columns_to_visualize, start=1):
    plt.subplot(num_rows, 2, i)
    # Calculate value counts and sort in descending order
    value_counts_sorted = yes_data[column].value_counts().sort_values(ascending=False)
    sns.barplot(x=value_counts_sorted.index, y=value_counts_sorted.values, palette='Set2')
    plt.title(f'Distribution of Response (Yes) by {column.capitalize()}')
    plt.xlabel(column.capitalize())
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


# In[49]:


data.select_dtypes("int64","float")


# In[50]:


# OUTLIER ANALYSIS


# In[51]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



# Define continuous variables for analysis
continuous_vars = ['age', 'salary', 'balance', 'duration']

# Univariate Outlier Analysis
for var in continuous_vars:
    # Summary Statistics
    print(f"Summary Statistics for {var}:")
    print(data[var].describe())
    
    # Boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=data[var], orient='v', color='skyblue')
    plt.title(f'Boxplot of {var}')
    plt.ylabel(var)
    plt.show()
    
    # Histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(data=data, x=var, bins=20, kde=True, color='lightgreen')
    plt.title(f'Histogram of {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.show()

# Bivariate Outlier Analysis
# Scatterplot Matrix
sns.pairplot(data[continuous_vars])
plt.show()

# Correlation Matrix
corr_matrix = data[continuous_vars].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Multivariate Outlier Analysis
# Clustering (DBSCAN)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[continuous_vars])

# DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(data_scaled)

# Visualize clusters
plt.figure(figsize=(8, 6))
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=clusters, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.show()

# Principal Component Analysis (PCA)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data_scaled)

# Visualize PCA
plt.figure(figsize=(8, 6))
plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5)
plt.title('Principal Component Analysis (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Statistical Methods for Outlier Detection
# Z-Score
z_scores = np.abs((data[continuous_vars] - data[continuous_vars].mean()) / data[continuous_vars].std())
outliers_zscore = data[z_scores > 3]

# IQR Method
Q1 = data[continuous_vars].quantile(0.25)
Q3 = data[continuous_vars].quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = data[((data[continuous_vars] < (Q1 - 1.5 * IQR)) | (data[continuous_vars] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Visualize Outliers
print("Outliers detected using Z-Score method:")
print(outliers_zscore)

print("Outliers detected using IQR method:")
print(outliers_iqr)


# In[52]:


# Aggregate data to count 'yes' and 'no' responses for each month
response_counts = data.groupby(['month', 'response']).size().unstack(fill_value=0)

# Calculate proportion of 'yes' and 'no' responses for each month
response_proportion = response_counts.div(response_counts.sum(axis=1), axis=0)

# Define the order of months
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# Plot line plot with response proportions w.r.t. month
plt.figure(figsize=(10, 6))

# Plot 'yes' responses
sns.lineplot(data=response_proportion['yes'], marker='o', label='Yes')

# Plot 'no' responses
sns.lineplot(data=response_proportion['no'], marker='o', label='No')

plt.title('Response Proportion by Month')
plt.xlabel('Month')
plt.ylabel('Proportion')
plt.xticks(ticks=range(len(month_order)), labels=month_order, rotation=45)
plt.legend(title='Response')
plt.grid(True)
plt.tight_layout()
plt.show()

