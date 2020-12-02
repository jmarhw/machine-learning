import pandas as pd
import re

data = pd.read_csv('twitter_sentiment_data.csv', delimiter=',', header=None, usecols = [0,1])
data_to_process = list(data[1])

def preprocess(data):
    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern       = '@[^\s]+'
    alphaPattern      = "[^a-zA-Z0-9]"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"

    stopwords = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
                    'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
                    'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
                    'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
                    'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
                    'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
                    'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
                    'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
                    'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
                    's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
                    't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
                    'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
                    'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
                    'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
                    'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
                    "youve", 'your', 'yours', 'yourself', 'yourselves', 'http', 'htt', 'https', 'rt']

    clean_data = []
    for tweet in data:
        new_tweet = tweet.lower()
        change_URL = re.sub(urlPattern, ' URL', new_tweet)
        # Replace @USERNAME to 'USER'.
        change_username = re.sub(userPattern,'', change_URL)
        # Replace all non alphabets.
        change_signs = re.sub(alphaPattern, " ", change_username)
        # Replace 3 or more consecutive letters by 2 letter.
        long_words = re.sub(sequencePattern, seqReplacePattern, change_signs)

        tweetwords = ''
        for word in long_words.split():
            # Checking if the word is a stopword.
            if word not in stopwords:
                if len(word) > 1:
                    tweetwords += (word+' ')

        clean_data.append(tweetwords)
    return clean_data

data1 = data[0].replace({-1: 0, 0: 1, 1:2, 2:3}) #replace labels because function .to_categorical takes only integers from 0...
new_df = pd.concat([data1, pd.DataFrame(preprocess(data_to_process))], axis = 1)
new_df.to_csv("processed_data.csv", index = False, header = False)


test_data =     ["Obama administration outlines path for climate change resiliency.",
                "Icebergs are melting due to climate change, study finds",
                "Do you know how dumb you have to be to not believe in climate change?!",
                "I think it’s due to climate change. Some people don’t believe in it. I do.",
                "Chinese government faked climate change to scare and control society",
                "The Russians did it. Oh, wait, we say it's climate change now.",
                "Leonardo DiCaprio looks great in the new movie about climate change."]

processed_test_data = pd.concat([pd.DataFrame(preprocess(test_data)), pd.DataFrame(test_data)], axis = 1)
processed_test_data.to_csv("test_data.csv", index = False, header = False)