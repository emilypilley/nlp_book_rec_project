from os import path

class BookRecommender():
    def __init__(self, all_books_info_dicts, all_books_synopses_topics, all_books_reivews_aspects_sentiments):
        self.all_books_info_dicts = all_books_info_dicts
        self.books_synopses_topics_dict = all_books_synopses_topics
        self.books_reivews_aspects_sentiments_dict = all_books_reivews_aspects_sentiments
    

    def get_books_features_for_recs(self):
        pass
    
    def find_similar_books(self, book):
        pass

    def find_top_n_recommendations(self, book):
        pass