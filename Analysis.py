
# coding: utf-8

# ## MIE1624 Project - Analyzing International Perspectives on Canada Through Trip Advisor’s Canadian Attraction Reviews

# ### Input of Trip Advisor Data (77 Selected Attractions) into PANDAS Dataframe

# In[1]:

import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:

# merge all csv files together

def get_merged_csv(flist, **kwargs):
    return pd.concat([pd.read_csv(f, **kwargs).assign(Attraction=os.path.splitext(os.path.basename(f))[0]) for f in flist], ignore_index=True)


# In[3]:

# path of data
# df is the base dataframe uncleaned
# length of dataframe df (unfiltered) reviews 112467

path = '/Users/KP/Desktop/MIE1624/Project/data_with_date'
fmask = os.path.join(path,'*.csv')
df = get_merged_csv(glob.glob(fmask), usecols=[1,2,3,4], header=0, encoding='latin-1', names=['Date','Location', 'ReviewText','Rating'])
print('Number of Reviews from the 77 Attractions is:',len(df))


# ### Functions for Cleaning Input Data and Creating Summary Statistics

# In[4]:

# clean df dataframe to remove single string location entries and remove canadian reviewers

def clean_df(df):
    
    df1 = df[df['Location'].apply(lambda x: len(str(x).strip().split(',')) != 1)]
    
    canadian_areas = [' Canada', ' Manitoba Canada', ' Ontario Canada', ' BC Canada', ' Quebec Canada', ' Manitoba Canada', ' Nova Scotia Canada',                   ' PEI Canada', ' Saskatchewan Canada' , ' New Brunswick Canada', ' BC', ' Quebec', ' Manitoba', ' Ontario', ' PEI', ' Nova Scotia',                  ' New Brunswick', ' Saskatchewan', ' Yukon', ' Northwest Territories', ' Yukon Canada', ' Northwest Territories Canada']
    
    df2 = df1['Location'].str.split(',').apply(lambda x: any(loc in canadian_areas for loc in x))
    
    df3 = df1.loc[~df2]
    
    return df3


# In[5]:

# input clean df and group by different columns in the dataframe such as 'Location' or 'Attraction'

def summary_stats(df, grouping):

    aggregations = {
        'Rating': {   
            'num': 'count',  
            'average': 'mean',
        }
    }
    
    df1 = df.groupby(grouping).agg(aggregations)
     
    df2 = df1.sort([('Rating', 'num')], ascending=False)
    
    return df2


# In[6]:

# clean dataframe of analysis

clean_df(df).head()


# ## Summary Statistics for the Canadian Attractions

# In[7]:

# length of clean df after filtering is 45462 reviews

print('The number of reviews from the 77 attractions after filtering the dataset is:',len(clean_df(df)))


# In[8]:

summary_stats(clean_df(df), 'Location').describe()
print ('The number of unique cities that visitors are from is:', len(summary_stats(clean_df(df), 'Location')))


# ### Locations of Reviewers of Canadian Attractions

# In[9]:

summary_stats(clean_df(df), 'Location')[0:20]


# In[10]:

summary_stats(clean_df(df), 'Location')[('Rating','num')][0:30].plot(kind='bar', legend=False, figsize=(20,10), title='Location of Reviewers')


# ### Average Ratings by Locations of Reviewers of Canadian Attractions

# In[11]:

summary_stats(clean_df(df), 'Location')[('Rating','average')][0:30].plot(kind='barh', legend=False, figsize=(20,10),                                                                         title='Average Rating Per Most Popular Locations of Reviewers', xlim=(4.2,4.8))


# In[12]:

summary_stats(clean_df(df), 'Location')[('Rating','num')][0:20].plot.pie(y=['Rating'], legend=False, figsize=(15,10), autopct='%1.1f%%', title='Location Proportion of Reviewers')


# ### Number of Reviews and Average Rating by Attraction

# In[13]:

summary_stats(clean_df(df), 'Attraction').describe()
print ('The number of attractions selected for the analysis is:',len(summary_stats(clean_df(df), 'Attraction')))


# In[14]:

summary_stats(clean_df(df), 'Attraction')


# ### Most Reviewed Attractions

# In[15]:

summary_stats(clean_df(df), 'Attraction')[('Rating','num')][0:50].plot(kind='bar', legend=False, figsize=(20,10), title='Most Reviewed Attractions')


# ### Distribution of Ratings

# In[16]:

summary_stats(clean_df(df), 'Rating').head(10)


# In[17]:

summary_stats(clean_df(df), 'Rating')[('Rating','num')].plot(kind='bar')


# ## Time Series Analysis of the Canadian Attractions

# In[18]:

from datetime import datetime, timedelta

# function was run to clean up the date field before and data is re-imported below
def date_cleanup(date):
    #strip \r\n
    date = date.replace("\r\n","")
    
    #strip "reviewed"
    date = date.replace("Reviewed ","")
    
    #change dates
    if date == "yesterday":
        date = "16 March 2017"
    if date == "2 days ago":
        date = "15 March 2017"
    elif date == "3 days ago":
        date = "14 March 2017"
    elif date == "4 days ago":
        date = "13 March 2017"
    elif date == "5 days ago":
        date = "12 March 2017"
    elif date == "6 days ago":
        date = "11 March 2017"
    elif date == "1 week ago":
        date = "10 March 2017"
    elif date == "2 weeks ago":
        date = "03 March 2017"
    elif date == "3 weeks ago":
        date = "24 February 2017"
    elif date == "4 weeks ago":
        date = "17 February 2017"
    elif date == "5 weeks ago":
        date = "10 February 2017"

    #convert to date time object
    datetimeobj = datetime.strptime(date, '%d %B %Y')
    
    return datetimeobj# function was run before to clean the data field and it is re-imported below
df1 = clean_df(df)
df1.reset_index(level = 0, drop = True, inplace = True)# function was run before to clean the data field and it is re-imported below
for i in range (len(df1)):
    df1['Date'][i] = date_cleanup(df1['Date'][i])
    if (i % 100 == 0):
        print("counted %d rows" % (i))
print('finished!!')  # df1.to_csv('df1-8.csv', sep='\t', encoding='utf-8')
# CSV file was exporting before and loaded below.
# In[19]:

# import the cleanned data, parse date
df2 = pd.read_csv('/Users/KP/Desktop/MIE1624/Project/df1-8.csv', header = 0, sep = '\t', encoding = 'utf-8', parse_dates=True, index_col='Date')
df2.columns = ['Index','Location','ReviewText','Rating','Attraction']


# In[20]:

# date parsed
df2.loc['august 2016'].head(5)


# In[21]:

# choose plot style
plt.style.use('seaborn-deep')
# seaborn-whitegrid, bmh, seaborn-paper, seaborn-deep, seaborn-talk


# In[22]:

# resample by:
# A year; Q quarter; M month; W week...


# ### Average Rating by Time Period
# There is no clear trend between the average rating and time.

# In[24]:

#average ratings by quarter
df2.loc['2012':'2017'].resample('Q').mean()[[1]].plot(figsize=(12,5), title = 'Average Ratings by Quarter (2012-present)') 

#average ratings by month
df2.loc['2012':'2017'].resample('M').mean()[[1]].plot(figsize=(12,5), title = 'Average Ratings by Month (2012-present)')


# ### Number of Ratings by Time Period
# Overall, the number of ratings increases rapidly in the past 4 years.
# 
# The peak for each year is at Q3 (July - September) 
# 
# Q1 and Q4 have fewer ratings, but there is a small peak at Q4 December (Christmas holiday)

# In[26]:

df2.loc['2012':'2016'].resample('A').count()[[0]].plot(figsize = (12,5),                                                 title = 'Number of Ratings by year (2012-2016)')

df2.loc['2012':'2016'].resample('Q').count()[[0]].plot(figsize = (12,5),                                                 title = 'Number of Ratings by quarter (2012-2016)')

df2.loc['jan 01 2012':'feb 28 2017'].resample('M').count()[[0]].plot(figsize = (12,5),                                                 title = 'Number of Ratings by month (2012-present)')

df2.loc['2016'].resample('M').count()[[0]].plot(figsize = (5,5),                                                 title = 'Number of Ratings by month (2016)')


# In[27]:

plt.rcParams["figure.figsize"] = (12,7)
plt.plot(df2.loc['2012':'2016'].resample('Q').count()[[1]].iloc[4::4], label = 'Q1')
plt.plot(df2.loc['2012':'2016'].resample('Q').count()[[1]].iloc[1::4], label = 'Q2')
plt.plot(df2.loc['2012':'2016'].resample('Q').count()[[1]].iloc[2::4], label = 'Q3')
plt.plot(df2.loc['2012':'2016'].resample('Q').count()[[1]].iloc[3::4], label = 'Q4')
plt.title('Number of Ratings by quarter (2012-2016)')
plt.xlabel('Date')
plt.legend(loc='upper left', numpoints = 1, prop={'size':18})
plt.plot(plotsize=(14,10))
plt.show()


# ### Number of Reviews per Time Period and Attractions
# Now we look at some specific attractions: CN Tower, Notre-Dame Basilica and the Air Canada Center. 
# 
# The first two attractions have more ratings during summer times than winter.
# 
# As for air canada center, it is an indoor attraction, so this trend does not exist.

# In[29]:

df2.loc[df2['Attraction'] == 'CN Tower'].loc['2016'].resample('M').count()[[0]].plot(figsize=(5,6),                                                  title = 'Number of Ratings per Month (CN Tower, 2016)')

df2.loc[df2['Attraction'] == 'CN Tower'].loc['jan 01 2014':'feb 28 2017'].resample('M').count()[[0]].plot(figsize = (15,6),                                                 title = 'Number of Ratings per Month (CN Tower, 2014-present)')

df2.loc[df2['Attraction'] == 'Notre-Dame Basilica'].loc['2016'].resample('M').count()[[0]].plot(figsize=(5,6),                                                  title = 'Number of Ratings per Month (Notre-Dame Basilica, 2016)')

df2.loc[df2['Attraction'] == 'Notre-Dame Basilica'].loc['jan 01 2014':'feb 28 2017'].resample('M').count()[[0]].plot(figsize = (15,6),                                                 title = 'Number of Ratings per Month (Notre-Dame Basilica, 2014-present)')

df2.loc[df2['Attraction'] == 'The Air Canada Centre'].loc['2016'].resample('M').count()[[0]].plot(figsize=(5,6),                                                  title = 'Number of Ratings per Month (The Air Canada Centre, 2016)')

df2.loc[df2['Attraction'] == 'The Air Canada Centre'].loc['jan 01 2015':'feb 28 2017'].resample('M').count()[[0]].plot(figsize=(15,6),                                                  title = 'Number of Ratings per Month (The Air Canada Centre, 2015-present)')


# ## Sentiment Analysis
# 
# 

# In[38]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import math
#Download required ntlk packages and lib
get_ipython().system('pip --quiet install nltk')
import nltk
nltk.download("vader_lexicon")
nltk.download("stopwords")
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from nltk import tokenize
from nltk.corpus import stopwords
sid = SentimentIntensityAnalyzer()


# In[39]:

attraction_df = pd.read_csv('attraction.csv')
attraction_list = []
for index, row in attraction_df.iterrows():
    attraction_list.append([row['Location']]+[row['ReviewText']]+[row['Rating']]+[row['Attraction']]+['positive' if row['Rating'] > 3 else 'negative'])
attraction_df = pd.DataFrame(attraction_list)
attraction_df.columns=['Review_Location','Review','Rating','Attraction','GroundTruth']


# In[40]:

stop = set(stopwords.words('english'))
stop.add('see')
stop.add('falls')
stop.add('tower')
stop.add('city')
stop.add('get')


# In[41]:

attractions = attraction_df['Review'].as_matrix()
#Count the frequency of words
from collections import Counter
import re
counter = Counter()
for attraction in attractions:
        counter.update([word.lower() for word in re.findall(r'\w+', attraction) if word.lower() not in stop and len(word) > 2])
k = 500
topk = counter.most_common(k)


# In[42]:

pdlist = []
#Assign Vader score to individual review using Vader compound score
for rownum, attraction in enumerate(attractions):
    ss = sid.polarity_scores(attraction)
    pdlist.append([attraction]+[ss['compound']])
    if (rownum % 5000 == 1):
            print("processed %d reviews" % (rownum+1))


# In[43]:

reviewDf = pd.DataFrame(pdlist)
reviewDf.columns = ['reviewCol','vader']
#Find out if a particular review has the word from topk list
freqReview = []
for i in range(len(reviewDf)):
    tempCounter = Counter([word.lower() for word in re.findall(r'\w+',reviewDf['reviewCol'][i])])
    topkinReview = [1 if tempCounter[word] > 0 else 0 for (word,wordCount) in topk]
    freqReview.append(topkinReview)
#Prepare freqReviewDf
freqReviewDf = pd.DataFrame(freqReview)
dfName = []
for c in topk:
    dfName.append(c[0])
freqReviewDf.columns = dfName


# In[44]:

finalreviewDf = reviewDf.join(freqReviewDf)
finaldf = attraction_df[['Review_Location','Attraction','Rating','GroundTruth']].join(finalreviewDf)


# ### Find most sentimental words by Mutual Information
# The Mutual Information is a measure of the similarity between two labels of the same data.

# In[45]:

gtScore = []
for i in range(len(finaldf)):
    if finaldf['Rating'][i]>3:
        gtScore.append(1)
    else:
        gtScore.append(0)


# In[46]:

#Calculate muual information score using scikit lean package
import sklearn
import sklearn.metrics as metrics
miScore = []
for word in topk:
    miScore.append([word[0]]+[metrics.mutual_info_score(gtScore,finaldf[word[0]].as_matrix())])
miScoredf = pd.DataFrame(miScore).sort_values(1,ascending=0)
miScoredf.columns = ['Word','MI Score']
miScoredf.head(20)


# ### PMI for Positive and Negative
# Similar to MI, PMI is measuring for sigle event where MI is the average of all possible event.
# The events P(x,y) = P(0,1) means the event of the review is negative but the specific word is existing in that review

# In[47]:

#Obtain finaldf again for PMI, it should be same as the finaldf before. But just tobe safe, we retrieve it again
freqReview = []
for i in range(len(reviewDf)):
    tempCounter = Counter([word.lower() for word in re.findall(r'\w+',reviewDf['reviewCol'][i])])
    topkinReview = [1 if tempCounter[word] > 0 else 0 for (word,wordCount) in topk]
    freqReview.append(topkinReview)
freqReviewDf = pd.DataFrame(freqReview)
dfName = []
for c in topk:
    dfName.append(c[0])
freqReviewDf.columns = dfName
finalreviewDf = reviewDf.join(freqReviewDf)
finaldf = attraction_df[['Attraction','Rating','GroundTruth']].join(finalreviewDf)
finaldf


# In[48]:

def pmiCal(df, x):
    pmilist=[]
    for i in ['positive','negative']:
        for j in [0,1]:
            px = sum(finaldf['GroundTruth']==i)/len(df)
            py = sum(finaldf[x]==j)/len(df)
            pxy = len(finaldf[(finaldf['GroundTruth']==i) & (finaldf[x]==j)])/len(df)
            if pxy==0:#Log 0 cannot happen
                pmi = math.log10((pxy+0.0001)/(px*py))
            else:
                pmi = math.log10(pxy/(px*py))
            pmilist.append([i]+[j]+[px]+[py]+[pxy]+[pmi])
    pmidf = pd.DataFrame(pmilist)
    pmidf.columns = ['x','y','px','py','pxy','pmi']
    return pmidf


# In[49]:

pmiCal(finaldf,'fall')


# ### Select all Ground Truth 1 and 2 reviews and produce top PMI word scores for top 500 words so we can see negative sentiment words

# In[50]:

#Calculate muual information score using scikit lean package
import sklearn
import sklearn.metrics as metrics
pmiScore = []
for rownum,word in enumerate(topk):
    pmiScore.append([word[0]]+[pmiCal(finaldf,word[0])['pmi'][3]])
    if (rownum % 25 == 1):
        print("processed %d reviews" % (rownum+1))
pmiScoredf = pd.DataFrame(pmiScore).sort_values(1,ascending=0)
pmiScoredf.columns = ['Word','PMI Score']
pmiScoredf.head(20)


# ### Select top reviewed attraction 'Niagra Falls' to extract top PMI words maybe 20 words so we can see why people like it so much

# In[51]:

niagarafalls_df = finaldf.loc[finaldf['Attraction'] == 'Niagara Falls']


# In[52]:

pmiScore = []
for rownum,word in enumerate(topk):
    pmiScore.append([word[0]]+[pmiCal(niagarafalls_df,word[0])['pmi'][1]])
    if (rownum % 25 == 1):
        print("processed %d reviews" % (rownum+1))
pmiScoredf = pd.DataFrame(pmiScore).sort_values(1,ascending=0)
pmiScoredf.columns = ['Word','PMI Score']
pmiScoredf.head(20)


# ### Analysis Over All Items

# In[53]:

#We are only intereseted in this three column for overall analysis
itemAnalysisDf = finaldf[['reviewCol','GroundTruth','vader']]


# In[54]:

from collections import Counter
import re
#To find out the most frequent word in review when the ground truth is positive
counter = Counter()
for review in itemAnalysisDf.loc[itemAnalysisDf['GroundTruth']=='positive']['reviewCol']:
        counter.update([word.lower() for word in re.findall(r'\w+', review) if word.lower() not in stop and len(word) > 2])


# In[55]:

k=10
topkPos = counter.most_common(k)
topkPos


# In[56]:

from collections import Counter
import re
counter = Counter()
#To find out the most frequent word in review when the ground truth is negative
for review in itemAnalysisDf.loc[itemAnalysisDf['GroundTruth']=='negative']['reviewCol']:
        counter.update([word.lower() for word in re.findall(r'\w+', review) if word.lower() not in stop and len(word) > 2])


# In[57]:

k=10
topkNeg = counter.most_common(k)
topkNeg


# In[58]:

from collections import Counter
import re
counter = Counter()
#To find out the most frequent word in review when the vader score is positive
for review in itemAnalysisDf.loc[itemAnalysisDf['vader']>0]['reviewCol']:
        counter.update([word.lower() for word in re.findall(r'\w+', review) if word.lower() not in stop and len(word) > 2])
k=10
topk = counter.most_common(k)
topk


# In[59]:

from collections import Counter
import re
counter = Counter()
#To find out the most frequent word in review when the vader score is negative
for review in itemAnalysisDf.loc[itemAnalysisDf['vader']<0]['reviewCol']:
        counter.update([word.lower() for word in re.findall(r'\w+', review) if word.lower() not in stop and len(word) > 2])
k=10
topk = counter.most_common(k)
topk


# ### Analysis per Item

# In[60]:

#Extract a list of attractions
attractions = finaldf['Attraction'].unique()
attractions


# In[61]:

#Rank by ground truth rating score
attractionRating = []
for attraction in attractions:
    itemDf = finaldf.loc[finaldf['Attraction']==attraction]
    attractionRating.append([attraction,itemDf['Rating'].mean()])
attractionRatingDfGt = pd.DataFrame(attractionRating)
attractionRatingDfGt.columns=['Attraction','avgRatingScore']
attractionRatingDfGt.sort_values('avgRatingScore',ascending=0).head(10)


# In[62]:

#Rank the attraction by ground truth rating score
attractionRating = []
for attraction in attractions:
    itemDf = finaldf.loc[finaldf['Attraction']==attraction]
    attractionRating.append([attraction,itemDf['vader'].mean()])
attractionRatingDfVd = pd.DataFrame(attractionRating)
attractionRatingDfVd.columns=['Attraction','avgRatingScore']
attractionRatingDfVd.sort_values('avgRatingScore',ascending=0).head(10)


# ### Plots Comparing Groundtruth and Vader

# In[63]:

import matplotlib.pyplot as plt
import numpy as np


# In[64]:

#Obtain finaldf again for PMI, it should be same as the finaldf before. But just tobe safe, we retrieve it again
freqReview = []
for i in range(len(reviewDf)):
    tempCounter = Counter([word.lower() for word in re.findall(r'\w+',reviewDf['reviewCol'][i])])
    topkinReview = [1 if tempCounter[word] > 0 else 0 for (word,wordCount) in topk]
    #topkinReview = [tempCounter[word] for (word,wordCount) in topk]
    freqReview.append(topkinReview)
freqReviewDf = pd.DataFrame(freqReview)
dfName = []
for c in topk:
    dfName.append(c[0])
freqReviewDf.columns = dfName
finalreviewDf = reviewDf.join(freqReviewDf)
finaldf = attraction_df[['Attraction','Rating','GroundTruth']].join(finalreviewDf)


# In[65]:

plt.hist(finaldf['Rating'].as_matrix())
plt.title("Ground Truth")
plt.xlabel("Value")
plt.ylabel("Frequency")
fig = plt.gcf()


# In[66]:

plt.hist(finaldf['vader'].as_matrix())
plt.title("Vadar Sentiment Analysis")
plt.xlabel("Value")
plt.ylabel("Frequency")
fig = plt.gcf()


# In[67]:

#Overlayed Histogram for GT rating and VD score
import numpy
from matplotlib import pyplot
#Just for demonstrating, I am dividing the rating score by 5
x = [finaldf['Rating'].as_matrix()/5]
y = [finaldf['vader'].as_matrix()]
bins = numpy.linspace(-1, 1, 100)
pyplot.hist(x, bins, label='x')
pyplot.hist(y, bins, label='y')
pyplot.legend(loc='upper right')
pyplot.show()


# In[68]:

tpdf = pd.DataFrame(topkPos)
tndf = pd.DataFrame(topkNeg)
tpdf.columns =['word','freq']
tndf.columns =['word','freq']
overlayhist = pd.merge(tpdf, tndf, on=['word'])
overlayhist


# In[69]:

word = overlayhist['word']
counts = overlayhist['freq_x']
# Plot stacked bar chart using matplotlib bar().
indexes = np.arange(len(word))
width = 0.5
p1 = plt.bar(indexes, counts, width, color='r')
word = overlayhist['word']
counts = overlayhist['freq_y']
indexes = np.arange(len(word))
width = 0.5
p2 = plt.bar(indexes, counts, width, color='b')
plt.xticks(indexes + width * 0.5, word)
plt.legend((p1[0], p2[0]), ('Positive', 'Negative'))


# In[ ]:




# In[ ]:




# In[ ]:



