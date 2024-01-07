from nltk.corpus import stopwords
from  nltk.tokenize import  word_tokenize
import pandas as pd
import re

# def workshop2():
df = pd.read_csv('spam.csv',encoding='ANSI')
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
def pre_process(sentence):
    stop_words = set(stopwords.words('english'))
    # remove EXPECT A-Za-z white space
    newsentence = re.sub(re.compile(r'[^A-Za-z\s+]'),'',sentence.strip().lower())
    pattern = re.compile(r'\s+')
    # replace mult_white space to one white space
    newsentence = re.sub(pattern, ' ', newsentence)
    word_tokens = word_tokenize(newsentence)
    filtered_sentence = [w for w in word_tokens if w not in stop_words]
    return filtered_sentence

conlumn = df.columns
df.drop(conlumn[4:],axis=1,inplace=True)
new_column = ['label','text','length','text2']
df.columns = new_column
for index, row in df.iterrows():
    sentence = row.loc['text']
    filtered_sentence = pre_process(sentence)
    row.loc['text'] = filtered_sentence
    df.loc[index,'length'] = len(filtered_sentence)
    row.loc['text'] = ' '.join(filtered_sentence)

print('2.After preprocess the text. The df is :')
print(df)
# 4.Use labelEncoder method to convert class target
le = LabelEncoder()
for index, row in df.iterrows():
    sentence = row.loc['text']
    sentence = sentence.split()
    encoded_label = le.fit_transform(sentence)
    row.loc['text2'] = encoded_label
pd.set_option('display.max_column',None)
print('\n3&4.Take Encoded-label into text2:')
print("\nTop 5 labelEncoder results(in text2):")
print(df['text2'].head())
print("\nBottom 5 labelEncoder results(in text2):")
print(df['text2'].tail())


# 5.Use CountVectorize to perform BOW
cv = CountVectorizer(max_features= 100)
corpus = df['text'].tolist()
corpus  = '\n'.join(corpus)
bow = cv.fit_transform([corpus])
print('\n5.Get BOW matrix:')
print(bow.toarray())
feature_names = cv.get_feature_names_out()
freq = bow.toarray().sum(axis=0)
bow_freq = {'Word': feature_names,'Frequency': freq}
df_bow = pd.DataFrame(bow_freq)
print('\n6.Get Word-frequency table with Top5 and bottom5 results:')
print(df_bow)
sorted_df_bow = df_bow.sort_values('Frequency',ascending=False)
print("\nTop 5 words by frequency:")
print(sorted_df_bow.head(5))
print("\nBottom 5 words by frequency:")
print(sorted_df_bow.tail(5))

