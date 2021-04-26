from os import path
import spacy
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class ReviewTopicsSentimentAnalyzer():
    def __init__(self, books_reviews_dict, topics_keywords_dict):
        self.books_reviews_dict = books_reviews_dict
        self.topics_keywords_dict = topics_keywords_dict

        self.spacy_nlp = spacy.load('en_core_web_sm')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
    
    def get_reivew_aspects(self, review_text):
        ''' Finds the aspects addressed in a review, and their associated modifiers.

        Returns a list of (keyword, modifiers, negation) tuple for a single review,
        where the keyword is from the topic keywords list, the modifiers is a list
        of modifiers that apply to that keyword, and negation is a boolean value
        corresponding to whethere a negation word applies to the aspect sentiment.
        '''
        keyword_mod_list = []
        review = self.spacy_nlp(review_text)
        for sent in review.sents:
            keywords = []
            modifiers = []
            negation = False
            for token in sent:
                if token.lemma_.isalpha(): # and not token.is_stop:
                    # if token.pos_ == 'NOUN':
                    if token.dep_ == 'nsubj':
                        if token.lemma_ in self.topics_keywords_dict:
                            keywords.append(token.lemma_)
                    elif token.dep_ == 'neg':
                        negation = True
                    if token.dep_ == 'amod' or token.pos_ == 'ADJ':
                        adv_mods = []
                        for child in token.children:
                            if child.dep_ == 'advmod':
                                adv_mods.append(child.text)
                        if len(adv_mods) > 0:
                            modifiers.append(token.text + ' ' + ' '.join(adv_mods))
                        else:
                            modifiers.append(token.text)
            if len(keywords) > 0 and len(modifiers) > 0:
                for keyword in keywords:
                    keyword_mod_list.append((keyword, modifiers, negation))
        
        return keyword_mod_list
    

    def get_book_topic_sentiments(self, reviews_text):
        '''For a list of reviews, returns a dictionary of the avg sentiments for each topic.'''
        # summing up topic sentiments and keeping count of number of times the topic is addressed
        total_topic_sentiment_count = {}
        for review in reviews_text:
            keyword_mod_list = self.get_reivew_aspects(review)
            for keyword, modifiers, negation in keyword_mod_list:
                # find the topic the keyword falls under and make sure it is valid - it should always be
                try:
                    topic, _ = self.topics_keywords_dict[keyword]
                except KeyError:
                    print('Invalid keyword: ', keyword)
                    continue
                # find the sentiment
                sentiment = self.sentiment_analyzer.polarity_scores(' '.join(modifiers))['compound']
                if negation:
                    sentiment = -sentiment
                # print(topic, sentiment)
                if topic in total_topic_sentiment_count:
                    sent_sum, count = total_topic_sentiment_count[topic]
                    sent_sum += sentiment
                    count += 1
                    total_topic_sentiment_count[topic] = (sentiment, count)
                else:
                    total_topic_sentiment_count[topic] = (sentiment, 1)
                # print(topic, keyword, sentiment)
        
        avg_topic_sentiments = {}
        for topic in total_topic_sentiment_count:
            sent_sum, count = total_topic_sentiment_count[topic]
            # only consider topics that were mentioned at least 3 times
            # and do not consider topics that are purely neutral
            if count > 3 and sent_sum != 0.0:
                avg_topic_sentiments[topic] = sent_sum/count

        return avg_topic_sentiments                


    def get_all_books_reivews_aspects_sentiments(self):
        '''For each book, finds the average sentiment for each topic.
        
        Builds and saves a dictionary containing each book (identified by the title and
        author), and a list of the topics addressed in the reviews and the average sentiment 
        expressed for that topic.'''

        if path.exists('rec_features/review_aspect_sentiments.p'):
            with open('rec_features/review_aspect_sentiments.p', 'rb') as f:
                return pickle.load(f)
        else:
            reviews_sentiment_dict = {}
            for book, reviews in self.books_reviews_dict:
                avg_topic_sentiments = self.get_book_topic_sentiments(book['reviews_text'])
                reviews_sentiment_dict[book] = [(int(topic), sentiment) 
                                                for topic, sentiment in avg_topic_sentiments.items()]
            
            with open('rec_features/review_aspect_sentiments.p', 'wb') as f:
                pickle.dump(reviews_sentiment_dict, f)
            return reviews_sentiment_dict