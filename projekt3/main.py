import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import re
import text2emotion as te
import datetime

file_path = 'C:\infa\zajecia\\4 semestr\io\projekt3\\ufc_data.csv'

df = pd.read_csv(file_path, encoding='cp1250', delimiter=';')
df.drop(['url', 'rawContent', 'id', 'user', 'replyCount', 'retweetCount', 'likeCount', 'quoteCount', 'conversationId',
         'source',	'sourceUrl', 'sourceLabel',	'links', 'media', 'retweetedTweet', 'quotedTweet', 'inReplyToTweetId', 'inReplyToUser',
         'mentionedUsers', 'coordinates', 'place', 'hashtags', 'cashtags', 'card', 'viewCount', 'vibe', 'bookmarkCount'], axis=1, inplace=True)
df = df[(df['lang'] == 'en')]

def preprocess_text(text):
    
    tokens = word_tokenize(text.lower())
    
    additional_stopwords = ['ufc', 'mma', 'danawhite', 'youtube', 'mmatwitter', 'fight', 'fighter', 'bst', '>', '<', 'ufc' '&', '#', '(', ')', '.', ',', '\'', '?', '!', '``', 'â€', '"', '--', 'n\'t', "''", '“', '\'s', '@', 'et', 'pt', ':', 're']
    stopwords_list = stopwords.words('english') + additional_stopwords
    filtered_tokens = [token for token in tokens if token not in stopwords_list]

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

def remove_url(string):
    return re.sub(r'http\S+|www\S+', '', string)

df['renderedContent'] = df['renderedContent'].apply(lambda x: remove_url(x))
df['processed'] = df['renderedContent'].apply(lambda x: preprocess_text(x))
def strip_time(date_string):
    datetime_format = "%d.%m.%Y %H:%M"
    try:
        datetime_obj = datetime.datetime.strptime(date_string, datetime_format + ":%S")
    except ValueError:
        datetime_obj = datetime.datetime.strptime(date_string, datetime_format)
    date_only = datetime_obj.strftime("%d.%m.%Y")
    return date_only
df['date'] = df['date'].apply(strip_time)

# analyzer = SentimentIntensityAnalyzer()
# df['vader'] = df['processed'].apply(lambda x: analyzer.polarity_scores(x))
# df["vadNeg"] = df['vader'].apply(lambda x: x["neg"])
# df["vadPos"] = df['vader'].apply(lambda x: x["pos"])
# df["vadCompound"] = df['vader'].apply(lambda x: x["compound"])
# df['vadSentiment'] = df['vader'].apply(lambda x: 'positive' if x['compound'] >= 0 else 'negative')

# positive_string = df[df['vadSentiment'] == 'positive']['processed'].str.cat(sep=' ')
# negative_string = df[df['vadSentiment'] == 'negative']['processed'].str.cat(sep=' ')

# wordcloudPositive = WordCloud(width=800, height=400, background_color='white').generate(positive_string)
# wordcloudNegative = WordCloud(width=800, height=400, background_color='white').generate(negative_string)

# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloudPositive, interpolation='bilinear')
# plt.axis('off')
# plt.title('Positive Vader Tweets')
# plt.show()

# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloudNegative, interpolation='bilinear')
# plt.axis('off')
# plt.title('Negative Vader Tweets')
# plt.show()

df["date"] = df.apply(lambda x: datetime.datetime.strptime(x["date"], "%d.%m.%Y").date(), axis=1)
# df_vadGrouped = df[["date", "vadNeg", "vadPos", "vadCompound"]]
# vadGroupedMeans = df_vadGrouped.groupby(["date"], as_index=False).mean()

# plt.plot(vadGroupedMeans["date"], vadGroupedMeans['vadPos'], color='green', label="Positive sentiment")
# plt.xlabel('Date')
# plt.ylabel('Mean Vader')
# plt.title('Mean Vader for Positive Sentiment')
# plt.legend()
# plt.show()

# plt.plot(vadGroupedMeans["date"], vadGroupedMeans['vadNeg'], color='red', label="Negative sentiment")
# plt.xlabel('Date')
# plt.ylabel('Mean Vader')
# plt.title('Mean Vader for Negative Sentiment')
# plt.legend()
# plt.show()

def t2e_processing():
    tweets = df['processed'].apply(lambda x: te.get_emotion(x))
    df = pd.DataFrame(tweets)
    file_path = "t2e_data.csv"
    df.to_csv(file_path, sep='\t', index=False)

file_path2 = 'C:\infa\zajecia\\4 semestr\io\projekt3\\t2e_data.csv'
data = []
with open(file_path2, 'r') as file:
    for line in file:
        line_data = eval(line.strip())
        data.append(line_data)
        
df['t2e'] = data
df["t2eHappy"] = df.apply(lambda x: x["t2e"]["Happy"], axis=1)
df["t2eAngry"] = df.apply(lambda x: x["t2e"]["Angry"], axis=1)
df["t2eSurprise"] = df.apply(lambda x: x["t2e"]["Surprise"], axis=1)
df["t2eSad"] = df.apply(lambda x: x["t2e"]["Sad"], axis=1)
df["t2eFear"] = df.apply(lambda x: x["t2e"]["Fear"], axis=1)

df['t2eSentiment'] = df['t2e'].apply(lambda x: 'surprise' if x['Surprise'] >= 0.25 else 'angry' if x['Angry'] >= 0.25 else 'other')

surprise_string = df[df['t2eSentiment'] == 'surprise']['processed'].str.cat(sep=' ')
angry_string = df[df['t2eSentiment'] == 'angry']['processed'].str.cat(sep=' ')

# wordcloudSurprise = WordCloud(width=800, height=400, background_color='white').generate(surprise_string)
# wordcloudAngry = WordCloud(width=800, height=400, background_color='white').generate(angry_string)

# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloudSurprise, interpolation='bilinear')
# plt.axis('off')
# plt.title('Surprise T2E Tweets')
# plt.show()

# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloudAngry, interpolation='bilinear')
# plt.axis('off')
# plt.title('Angry T2E Tweets')
# plt.show()

df_t2eGrouped = df[["date", "t2eHappy", "t2eSurprise", "t2eFear", "t2eAngry", "t2eSad"]]

t2eGroupedMeans = df_t2eGrouped.groupby(["date"], as_index=False).mean()

plt.plot(t2eGroupedMeans["date"], t2eGroupedMeans['t2eHappy'], color='green', label="Happy sentiment")
plt.xlabel('Date')
plt.ylabel('Mean Vader')
plt.title('Mean Vader for happy Sentiment')
plt.legend()
plt.show()

plt.plot(t2eGroupedMeans["date"], t2eGroupedMeans['t2eSurprise'], color='red', label="Surprise sentiment")
plt.xlabel('Date')
plt.ylabel('Mean Vader')
plt.title('Mean Vader for surprise Sentiment')
plt.legend()
plt.show()

plt.plot(t2eGroupedMeans["date"], t2eGroupedMeans['t2eFear'], color='red', label="Fear sentiment")
plt.xlabel('Date')
plt.ylabel('Mean Vader')
plt.title('Mean Vader for fear Sentiment')
plt.legend()
plt.show()

plt.plot(t2eGroupedMeans["date"], t2eGroupedMeans['t2eAngry'], color='red', label="Angry sentiment")
plt.xlabel('Date')
plt.ylabel('Mean Vader')
plt.title('Mean Vader for angry Sentiment')
plt.legend()
plt.show()

plt.plot(t2eGroupedMeans["date"], t2eGroupedMeans['t2eSad'], color='red', label="Sad sentiment")
plt.xlabel('Date')
plt.ylabel('Mean Vader')
plt.title('Mean Vader for sad Sentiment')
plt.legend()
plt.show()

