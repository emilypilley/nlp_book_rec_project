from os import path
import pandas as pd
import validators
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

        for book in set(self.books_synopses_topics_dict).union(self.books_reivews_aspects_sentiments_dict):
            book_features = {}
            book_features['title-author'] = book
            review_topics_list = []
            synopsis_topics_list = []
            for topic, sentiment in self.books_reivews_aspects_sentiments_dict[book]:
                book_features['review_topic_' + str(topic)] = sentiment
            for topic, relevancy in self.books_synopses_topics_dict[book]:
                book_features['synopsis_topic_' + str(topic)] = relevancy
            
            books_features_dicts.append(book_features)
        
        self.books_features_dicts = books_features_dicts
        
        df = pd.DataFrame(books_features_dicts)
        df = df.fillna(0.0)
        df.to_csv("recommendation_features.csv")

        return df


    def group_similar_books(self):
        '''Uses DBSCAN to cluster books based on features gathered from synopses and reviews'''
        scaler = StandardScaler()
        df_features = self.book_features_df.loc[:, self.book_features_df.columns != 'title-author']
        scaled_features = scaler.fit_transform(df_features)

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
                    cluster_list.append(clustered_df['title-author'][idx])
            print(i, cluster_list, '\n')


    def get_book_df_idx_for_rec(self, book):
        '''Returns the index in the dataframe for the requested book.
        
        If user inputs a URL, they are looking to add a book to the dataset, 
        and the necessary features will be extracted from the page. If the user
        did not input a valid URL, they are trying to use an existing book in the 
        dataset, and the appropriate frame will be extracted if it existis.'''

        book_idx = None
        if validators.url(book):
            # TODO: get features and add to dataframe
            print("String is a valid URL - Need to implement still")
            # raise error if it was a URL but info couldn't be extracted
        else:
            book_name = book.replace(' ', '_').lower()
            for idx in self.book_features_df.index:
                if self.book_features_df['title-author'][idx].lower().startswith(book_name):
                    book_idx = idx
            if book_idx == None:
                raise Exception(book + ' is not in dataset, please add it first.')
        
        return book_idx


    def find_top_n_recommendations(self, book, num_books=10):
        '''Get top n book reccomendations based on cosine similarity'''
        if num_books < 1 or num_books > 50:
            raise Error('Number of books to reccomend must be between 1 and 50')

        book_idx = self.get_book_df_idx_for_rec(book)
        if book_idx is not None:
            # Calculate cosine similarity over all books
            df_features = self.book_features_df.loc[:, self.book_features_df.columns != 'title-author']
            cosine_sim = cosine_similarity(df_features)
            similar_books = list(enumerate(cosine_sim[book_idx]))
            sorted_similar_books = (
                sorted(similar_books, key=lambda x:x[1], reverse=True))
            print('all sim books: ', len(sorted_similar_books))
            
            similar_books_list = []
            count = 0
            for idx, sim in sorted_similar_books:
                # The most similar book will be itself
                if count == 0:
                    count += 1
                    continue
                book_str = self.book_features_df.at[idx, 'title-author']
                similar_books_list.append(book_str)
                count += 1
                if count > num_books:
                    break
            return similar_books_list
        else:
            raise Exception('Sorry, we could not find the requested book.')
