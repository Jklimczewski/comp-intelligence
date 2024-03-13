import snscrape.modules.twitter as sntwitter
import pandas as pd
import datetime

def scrap():
    tweets = []

    hashtag = "#UFC"
    beginning_date = datetime.date(2022, 9, 1)
    end_date = date + datetime.timedelta(days=1)
    max_tweets_per_day = 200

    while date != datetime.date(2023, 1, 30):
        query = f'#{hashtag} since:{beginning_date} until:{end_date}'
        new_tweets = []
        for i, tweet in enumerate(sntwitter.TwitterHashtagScraper(query, maxEmptyPages=5).get_items()):
            new_tweets.append(tweet)

            if len(new_tweets) >= max_tweets_per_day:
                break
        tweets.extend(new_tweets)
        beginning_date += datetime.timedelta(days=1)
        end_date += datetime.timedelta(days=1)

    print("Number of scraped tweets:", len(tweets))

    df = pd.DataFrame(tweets)
    file_path = "ufc_data.csv"

    df.to_csv(file_path, sep='\t', index=False)