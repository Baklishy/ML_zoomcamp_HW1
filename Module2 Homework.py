#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd

# file path
file_path = r'D:\University\laptops.csv'

# Read the CSV file
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
df.head()


# In[9]:


num_records = df.shape[0]

print(f'The dataset contains {num_records} records.')


# In[12]:


num_brands = df['Brand']
print(f'The dataset contains {num_brands} unique laptop brands.')


# In[15]:


missing_columns = df.isnull().sum()
missing_columns


# In[18]:


max_dell_price = df[df['Brand'] == 'Dell']['Final Price'].max()
max_dell_price


# In[21]:


initial_median_screen = df['Screen'].median()

most_frequent_screen = df['Screen'].mode()[0]

df['Screen'].fillna(most_frequent_screen, inplace = True)

final_median_screen = df['Screen'].median()

print(f"Initial median of 'ScreenResolution': {initial_median_screen}")
print(f"Most frequent value in 'ScreenResolution': {most_frequent_screen}")
print(f"Final median of 'ScreenResolution' after filling missing values: {final_median_screen}")


# In[27]:


innjoo_laptops = df[df['Brand'] == 'Innjoo']

selected_columns = innjoo_laptops[['RAM', 'Storage', 'Screen']]

X = selected_columns.to_numpy()

XTX = np.dot(X.T, X)

XTX_inv = np.linalg.inv(XTX)

y = np.array([1100, 1300, 800, 900, 1000, 1100])

w = np.dot(np.dot(XTX_inv, X.T), y)

sum_w = np.sum(w)
sum_w

