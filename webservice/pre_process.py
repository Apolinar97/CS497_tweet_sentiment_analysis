def norm_tweet(tweet):
  tweet = tweet.lower()
  tweet = re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', ' ', tweet)
  tweet = re.sub('[^a-z]+', ' ', tweet)
  return tweet

def remove_stopwords_for_analysis(tweet):
  stop_words = stopwords.words('english')
  white_list = ['no', 'not']
  words_tweet = tweet.split()
  clean_tweet = [word for word in words_tweet if(word not in stop_words or word in white_list)]
  return ' '.join(clean_tweet)


def process_tweet(tweet):
  tweet = norm_tweet(tweet)
  return remove_stopwords_for_analysis(tweet)

def validate_tweet(tweet):
  if (not tweet or not isinstance(tweet,str)):
    return False
  else:
    return True