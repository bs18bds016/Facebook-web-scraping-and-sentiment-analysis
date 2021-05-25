#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Data was scrapped using a chrome extension DataScrapper from the Facebook and extracted as a csv file


# In[ ]:





# ![image.png](attachment:image.png)
# 

# In[1]:


import pandas as pd


# In[2]:


df2= pd.read_csv(r"D:\FB_PROJ\Fb_Data.csv")


# In[3]:


df2.head()


# In[4]:


print(df2.dtypes)


# In[5]:


df2.to_csv('out_full_sd.csv',encoding='utf-8-sig')


# In[6]:


#Re-Importing the data and do data analysis on it 
import pandas as pd
df = pd.read_csv('out_full_sd.csv')
print(df.dtypes)
df


# In[7]:


#Converting time 
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df['Date'] = pd.to_datetime(df['Date']).dt.date

startdate = pd.to_datetime('2021-01-01').date()

mask = (df['Date'] > startdate)
df = df.loc[mask]

print(df.dtypes)

df


# In[ ]:





# In[8]:


#Where the data Analysis Start


import pandas as pd
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib


# In[9]:


#Filtering Connor for analysis
is_connor =  df['User']=="Conor McGregor"
df_connor  = df[is_connor]
df_connor


# In[10]:


#Processing time and filter columns for analysis
df_connor = df_connor[['Date', 'Likes']].groupby('Date').mean().reset_index()



# In[11]:


df_connor = df_connor.groupby('Date').mean().reset_index()


# In[12]:


df_connor.head()


# In[13]:


import pandas as pd
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib

df_connor.plot(x = "Date", y = "Likes" ,figsize = (15, 6) )


# In[ ]:





# In[28]:


is_connor =  df['User']=="Dustin Poirier"
df_connor  = df[is_connor]
df_connor


# In[29]:


df_connor = df_connor[['Date', 'Likes']].groupby('Date').mean().reset_index()


# In[30]:


import pandas as pd
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib

df_connor.plot(x = "Date", y = "Likes" ,figsize = (15, 6) )


# In[ ]:





# In[ ]:


print(df_connor.dtypes)


# In[ ]:


#Plotting trend Alongside Observed data accross the timeline
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, ax = plt.subplots()
ax.grid(True)

year = mdates.YearLocator(month=1)
month = mdates.MonthLocator(interval=3)
year_format = mdates.DateFormatter('%Y')
month_format = mdates.DateFormatter('%m')

ax.xaxis.set_minor_locator(month)
ax.xaxis.grid(True, which = 'minor')
ax.xaxis.set_major_locator(year)
ax.xaxis.set_major_formatter(year_format)

plt.plot(df_connor.index, df_connor['Likes'], c='blue')
plt.plot(decomposition_connor.trend.index, decomposition_connor.trend, c='red')


# In[ ]:





# In[ ]:





# In[ ]:


df_connor['caption']= df['Description']


# In[ ]:


df_connor_description=df_connor['caption']


# In[ ]:


df_connor_description.head()


# In[ ]:


import re
corpus=[]
for i in (df_connor_description):
    words=re.sub('[^a-zA-Z]',' ',i);
    words=words.lower()
    print(words)
    for j in words.split():
        if j not in corpus:
            corpus.append(j)


# In[ ]:


# Python program to convert a list to string

# Function to convert
def listToString(s):

    str1 = ""

    for ele in s:
        str1 += ele

    return str1


# In[ ]:


d=listToString(df_connor_description)


# In[ ]:


description_conor


# In[ ]:


from textblob import TextBlob


# In[ ]:


edu=TextBlob(d)

x=edu.sentiment.polarity

if x<0:
    print("negative")
elif x==0:
    print("Neutral")
elif x>0 and x<=1:
    print("positive")


# In[ ]:


#Filtering Nate for analysis
is_nate =  df['User']=="Nate Diaz"
df_nate  = df[is_nate]
df_nate


# In[ ]:


df_nate['caption']= df['Description']


# In[ ]:


df_nate_description=df_nate['caption']


# In[ ]:


d_nate=listToString(df_nate_description)


# In[ ]:


edu=TextBlob(d_nate)

x=edu.sentiment.polarity

if x<0:
    print("negative")
elif x==0:
    print("Neutral")
elif x>0 and x<=1:
    print("positive")


# In[ ]:


#Filtering Dustin for analysis
is_dustin =  df['User']=="Dustin Poirier"
df_dustin  = df[is_dustin]
df_dustin


# In[ ]:


df_dustin['caption']= df['Description']


# In[ ]:


df_dustin_description=df_dustin['caption']


# In[ ]:


d_dustin=listToString(df_dustin_description)


# In[ ]:


edu=TextBlob(d_dustin)

x=edu.sentiment.polarity

if x<0:
    print("negative")
elif x==0:
    print("Neutral")
elif x>0 and x<=1:
    print("positive")


# In[ ]:


#Data was scrapped using a chrome extension DataScrapper from the UFC website and extracted as a csv file


# ![image.png](attachment:image.png)
# 

# In[ ]:





# In[46]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from plotly.offline  import download_plotlyjs,init_notebook_mode,plot, iplot
import cufflinks as cf
init_notebook_mode(connected = True)
cf.go_offline()
get_ipython().run_line_magic('matplotlib', 'inline')
from chart_studio.plotly import *
import chart_studio.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.offline as offline
# Squarify for treemaps
import squarify
# Random for well, random stuff
import random
# operator for sorting dictionaries
import operator
# For ignoring warnings
import warnings
warnings.filterwarnings('ignore')


# In[47]:


df=pd.read_csv(r"D:\FB_PROJ\data.csv")


# In[48]:


df.describe()


# In[ ]:





# In[51]:


#fig, ax = plt.subplots(1,2, figsize=(12, 20))
fig, ax = plt.subplots(1,2, figsize=(15, 5))
sns.distplot(df.B_Age, ax=ax[0])
sns.distplot(df.R_Age, ax=ax[1])


# In[ ]:





# In[ ]:





# In[50]:


fig, ax = plt.subplots(1,2, figsize=(15, 5))
above35 =['above35' if i >= 35 else 'below35' for i in df.B_Age]
df_B = pd.DataFrame({'B_Age':above35})
sns.countplot(x=df_B.B_Age, ax=ax[0])
plt.ylabel('Number of fighters')
plt.title('Age of Blue fighters',color = 'blue',fontsize=15)

above35 =['above35' if i >= 35 else 'below35' for i in df.R_Age]
df_R = pd.DataFrame({'R_Age':above35})
sns.countplot(x=df_R.R_Age, ax=ax[1])
plt.ylabel('Number of Red fighters')
plt.title('Age of Red fighters',color = 'Red',fontsize=15)


# In[55]:


cnt_srs = df['R_Location'].value_counts().head(15)

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
    ),
)

layout = go.Layout(
    title='Most Popular cities for Red fighters'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename="Ratio")


# In[56]:


cnt_srs = df['B_Location'].value_counts().head(15)

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
    ),
)

layout = go.Layout(
    title='Most Popular cities for Blue fighters'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename="Ratio")


# In[ ]:





# In[52]:


sns.lmplot(x="B__Round1_Strikes_Body Significant Strikes_Attempts", 
               y="B__Round1_Strikes_Body Significant Strikes_Landed", 
               col="winner", hue="winner", data=df, col_wrap=2, size=6)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




