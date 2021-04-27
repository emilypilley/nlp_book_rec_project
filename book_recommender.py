from os import path
import pandas as pd
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

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
        self.book_features_df = self.get_combined_synopsis_reviews_features_df()


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


    def group_similar_books(self):
        '''Uses DBSCAN to cluster books based on features gathered from synopses and reviews'''
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.book_features_df)

        dbscan = DBSCAN(eps=3.0, min_samples=5).fit(scaled_features)
        labels = dbscan.labels_
        print(labels)

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        print("Silhouette Coefficient: %0.3f"
            % metrics.silhouette_score(scaled_features, labels))
        
        clustered_df = self.book_features_df
        clustered_df['Group'] = labels

        print('\nBook Groups:')
        for i in range(-1, n_clusters_):
            cluster_list = []
            for idx in clustered_df.index:
                if clustered_df['Group'][idx] == i:
                    cluster_list.append(idx)
            print(i, cluster_list, '\n')

    def find_top_n_recommendations(self, book):
        pass