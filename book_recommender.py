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
    def __init__(self, book_info_obj, book_text_analyzer, review_topics_sentiment_analyzer):
        self.book_info_obj = book_info_obj
        self.all_books_info = self.book_info_obj.book_info_dicts
        self.book_text_analyzer = book_text_analyzer
        self.review_topics_sentiment_analyzer = review_topics_sentiment_analyzer

        self.books_synopses_topics_dict = (
            self.book_text_analyzer.get_books_synopses_classifications(self.all_books_info))
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


    def get_synopsis_reviews_features_dict_from_url(self, book_url, verbose=True):
        '''Returns a dictionary of the combined synopsis and reviews feature for a book.'''
        feature_dict = {}
        try:
            new_book = self.book_info_obj.get_book_info_from_url(book_url)
        except Exception:
            raise Exception('Could not get book info from ', book_url)
        
        new_book_synopsis = new_book['synopsis']
        new_book_reviews = new_book['reviews_text']

        title_author_str = new_book['title'] + '-' + new_book['author']
        feature_dict['title-author'] = title_author_str.replace(' ', '_')

        synopsis_topics = self.book_text_analyzer.get_topics_from_synopsis(new_book_synopsis)
        review_topic_sents = self.review_topics_sentiment_analyzer.get_book_topic_sentiments(new_book_reviews)
    
        if verbose:
            print('\n', title_author_str)
            print('\nSynopsis Topics:\n', synopsis_topics)
            print('\nReivew Topics + Sentiments:\n', review_topic_sents)
        
        for topic, sentiment in review_topic_sents.items():
            feature_dict['review_topic_' + str(topic)] = sentiment
        for topic, relevancy in synopsis_topics:
            feature_dict['synopsis_topic_' + str(topic)] = relevancy

        return feature_dict        


    def get_book_df_idx_for_rec(self, book):
        '''Returns df index for the book, adding a new book to df if needed.
        
        If user inputs a URL, they are looking to add a book to the dataset, 
        and the necessary features will be extracted from the page. This URL must
        be for the main page of the desired book on Goodreads.
        
        If the user did not input a valid URL, they may be trying to use an existing 
        book in the dataset, and the appropriate frame will be extracted if it existis.
        In this case, the title of the book should be the argument that is passed in,
        and (ignoring case) this title must match the beginning of that title of the
        book as it is in the dataset.'''

        book_idx = None
        if validators.url(book):
            book_feature_dict = self.get_synopsis_reviews_features_dict_from_url(book)
            book_title_author = book_feature_dict['title-author']
            df = pd.DataFrame(book_feature_dict, index=[0])
            self.book_features_df = self.book_features_df.append(df, ignore_index = True)
            self.book_features_df = self.book_features_df.fillna(0.0)
            self.book_features_df.to_csv("updated_recommendation_features.csv")
            book_idx = self.book_features_df.index[self.book_features_df['title-author'] == book_title_author][0]
        else:
            book_name = book.replace(' ', '_').lower()
            for idx in self.book_features_df.index:
                if self.book_features_df['title-author'][idx].lower().startswith(book_name):
                    book_idx = idx
            if book_idx == None:
                raise Exception(book + ' is not in dataset, please add it first.')
        
        return book_idx


    def find_top_n_recommendations(self, book, num_books=10):
        '''Get top n book reccomendations based on cosine similarity.
        
        The "book" argument passed in can either be a URL for the page of the
        relevant book on Goodreads, or the title of the book (if it already 
        exists in the dataset). If a new book is requested here, it will 
        be added to the dataframe and thus added to the pool of possible
        books to recommend in the future.'''

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
