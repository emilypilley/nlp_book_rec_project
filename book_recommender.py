from os import path
from sklearn.feature_extraction import DictVectorizer
from book_text_analyzer import BookTextAnalyzer
from review_topics_sentiment_analyzer import ReviewTopicsSentimentAnalyzer


class BookRecommender:
    def __init__(self, all_books_info, book_text_analyzer, review_topics_sentiment_analyzer):
        self.book_text_analyzer = book_text_analyzer
        self.review_topics_sentiment_analyzer = review_topics_sentiment_analyzer

        self.books_synopses_topics_dict = (
            self.book_text_analyzer.get_books_synopses_classifications(all_books_info))
        self.books_reivews_aspects_sentiments_dict = (
            self.review_topics_sentiment_analyzer.get_all_books_reivews_aspects_sentiments())
    

    def get_combined_synopsis_reviews_dicts(self):
        '''Returns a list of the combined synopsis and reviews feature for all the books.
        
        The list contains a dictionary for each book, in the form:
            {"review_topic": [("topic0", sentiment0), ("topic1", sentiment1), ...],
            "synopsis_topic":["topic0", "topic1", ...]}
        for use with DictVectorizer.'''
        books_features_dicts = []
        for book in set(self.books_synopses_topics_dict).union(self.books_reivews_aspects_sentiments_dict):
            book_features = {}
            review_topics_list = []
            synopsis_topics_list = []
            for review_topic_sent in self.books_reivews_aspects_sentiments_dict[book]:
                review_topics_list.append(review_topic_sent)
            for topic, relevancy in self.books_synopses_topics_dict.items():
                synopsis_topics_list.append(topic)
            
            book_features['reviews_topic'] = review_topics_list
            book_features['synopsis_topic'] = review_topics_list
            books_features_dicts.append(book_features)
        
        return books_features_dicts

    def get_books_features_for_recs(self):
        '''Uses DictVectorizer to represent variable number of features.'''
        pass
    
    def find_similar_books(self, book):
        pass

    def find_top_n_recommendations(self, book):
        pass