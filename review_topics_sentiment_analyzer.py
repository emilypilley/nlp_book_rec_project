from os import path
import spacy

class ReviewTopicsSentimentAnalyzer():
    def __init__(self, books_reviews_dict, review_topics_keywords_dict):
        self.books_reviews_dict = books_reviews_dict
        self.reviews_topics_keywords_dict = reviews_topics_keywords_dict
    
    def get_reivew_aspects_sentiments(self, review):
        '''Returns tuples of each topics addressed in the review and its corresponding sentiment score'''
        pass
    
    def get_all_books_reivews_aspects_sentiments(self):
        '''For each book, finds the average sentiment for each topic if topic is mentioned in more than two reviews'''
        pass