from os import path
import pandas as pd
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

        self.books_features_dicts = None
        self.dict_vectorizer = DictVectorizer()

    def get_combined_synopsis_reviews_features_df(self):
        '''Returns a dataframe of the combined synopsis and reviews feature for all the books.
        
        The dataframe consists of columns for each synopsis and review topic, and entries
        equal to the sentiment (for reviews topics) or relevancy (for synopses topics), 
        and all empty cells are filled in with 0.0.'''

        books_features_dicts = []
        df_indices = []
        for book in set(self.books_synopses_topics_dict).union(self.books_reivews_aspects_sentiments_dict):
            book_features = {}
            review_topics_list = []
            synopsis_topics_list = []
            for topic, sentiment in self.books_reivews_aspects_sentiments_dict[book]:
                book_features['review_topic_' + str(topic)] = sentiment
            for topic, relevancy in self.books_synopses_topics_dict[book]:
                book_features['synopsis_topic_' + str(topic)] = relevancy
            
            books_features_dicts.append(book_features)
            df_indices.append(book)
        
        self.books_features_dicts = books_features_dicts
        
        df = pd.DataFrame(books_features_dicts, index=df_indices)
        df = df.fillna(0.0)
        df.to_csv("recommendation_features.csv")

        return df


    def get_book_features_for_recs(self, book_feature_dicts_list):
        '''Uses DictVectorizer to represent variable number of features.
        
        Input is a list of book feature dictionaries, with a variable number
        of synopsis topics and review topic sentiments.'''
        features = self.dict_vectorizer.transform(book_features_dicts_list)
        print(features.toarray())
        print(dict_vectorizer.get_feature_names())
        return features
    
    def group_similar_books(self, book):
        pass

    def find_top_n_recommendations(self, book):
        pass