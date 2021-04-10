import requests
import time
import os
from os import path
import re
import pickle
from bs4 import BeautifulSoup
import sys
import spacy
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics import silhouette_score
from sklearn.decomposition import LatentDirichletAllocation as LDA


class GoodreadsBookInfo():
    def __init__(self, num_pages=5, num_books=500):
        assert num_books <= 100 * num_pages
        self.URLSTART = 'https://goodreads.com'
        self.num_pages = num_pages

        self.download_top_book_pages_html(num_pages)
        self.list_of_book_urls = self.get_top_book_urls(num_pages)

        self.book_info_dict = self.get_top_n_books_info(num_books)


    def download_top_book_pages_html(self, num_pages):
        BESTBOOKS = '/list/show/1.Best_Books_Ever?page='

        for i in range(1, num_pages + 1):
            if not path.exists('top_books_data/page' + '%02d' % i + '.html'):
                # get html for the the top books on Goodreads (100 books per page)
                bookpage = str(i)
                top_books_html = requests.get(self.URLSTART + BESTBOOKS + bookpage)
                file_to_write = 'top_books_data/page' + '%02d' % i + '.html'
                with open(file_to_write, 'w+') as f:
                    f.write(top_books_html.text)
                time.sleep(2)


    def get_top_book_urls(self, num_pages):
        if path.exists('top_books_data/url_list.txt'):
            url_list_file = open('top_books_data/url_list.txt','r')
            return url_list_file.read().splitlines()

        else:
            book_url_list = []
            for i in range(1, num_pages + 1):
                pg_num = '%02d' % i
                file_to_read = 'top_books_data/page' + pg_num+ '.html'

                with open(file_to_read) as fdr:
                    data = fdr.read()
                soup = BeautifulSoup(data, 'html.parser')

                for e in soup.select('.bookTitle'):
                    book_url_list.append(self.URLSTART + e['href'])
                
            with open('top_books_data/url_list.txt', 'w') as f:
                f.write('\n'.join(book_url_list))

            return book_url_list


    def get_book_info_from_url(self, book_page_url):
        book_info = {}
        book_page_html = requests.get(book_page_url)
        book_data = BeautifulSoup(book_page_html.text, 'html.parser')

        if book_data.select_one("div[itemprop='inLanguage']").text == 'English':
            book_info['title'] = book_data.select_one("meta[property='og:title']")['content']
            book_info['isbn'] = book_data.select_one("meta[property='books:isbn']")['content']

            author_url = book_data.select_one("meta[property='books:author']")['content']
            author_page_html = requests.get(author_url)
            author_data = BeautifulSoup(author_page_html.text, 'html.parser')
            book_info['author'] = author_data.select_one("meta[property='og:title']")['content']

            rating = book_data.select_one("span[itemprop='ratingValue']").text
            book_info['rating'] = re.findall(r"\d\.\d+", rating)[0]

            book_info['total_rating_count'] = book_data.select_one("meta[itemprop='ratingCount']")['content']
            book_info['total_review_count'] = book_data.select_one("meta[itemprop='reviewCount']")['content']

            # first item is actually synopsis, rest are reviews
            reviews = book_data.find_all(id=re.compile("freeText\d+"))
            book_info['synopsis']= reviews[0].get_text()
            book_info['reviews_text'] = [review.get_text() for review in reviews[1:]]

            print("Got: ", book_data.select_one("meta[property='og:title']")['content'])

        return book_info


    def get_top_n_books_info(self, num_books):
        if path.exists('top_books_data/book_info_dicts.p'):
            with open('top_books_data/book_info_dicts.p', 'rb') as f:
                return pickle.load(f)
        else:
            book_info_dicts = []
            for i in range(num_books):
                book_page_url = self.list_of_book_urls[i]
                try:
                    book_info_dict = self.get_book_info_from_url(book_page_url)
                    if len(book_info_dict) > 0:
                        book_info_dicts.append(book_info_dict)
                except Exception:
                    try:
                        book_info_dict = self.get_book_info_from_url(book_page_url)
                        if len(book_info_dict) > 0:
                            book_info_dicts.append(book_info_dict)
                    except Exception:
                        try:
                            book_info_dict = self.get_book_info_from_url(book_page_url)
                            if len(book_info_dict) > 0:
                                book_info_dicts.append(book_info_dict)
                        except Exception:
                            print("Error with: ", book_page_url)

                time.sleep(2)
            
            with open('top_books_data/book_info_dicts.p', 'wb') as f:
                pickle.dump(book_info_dicts, f)

            return book_info_dicts
    

class BookTextAnalyzer():
    def __init__(self, synopses_list, reviews_list, seed=7):
        self.seed = seed
        self.spacy_nlp = spacy.load('en_core_web_sm')

        self.synopses_list = synopses_list
        self.reviews_list = reviews_list
        
        self.synopses_topics = self.get_synopses_topics()
        self.reviews_topics = self.get_reviews_topics()

    def get_text_features(self, text_list):
        processed_text = []

        # lemmatize, remove non-alphabetic words and stopwords
        for text in self.spacy_nlp.pipe(text_list):
            only_alpha = ' '.join(token.lemma_ for token in text if token.lemma_.isalpha() and not token.is_stop)
            processed_text.append(only_alpha)
        
        tf_idf_vec = TfidfVectorizer(max_features=5000, lowercase=True, tokenizer=self.spacy_nlp)

        text_features = tf_idf_vec.fit_transform(processed_text)
        # list of unique words found by vectorizer
        text_feature_names = tf_idf_vec.get_feature_names()

        return (text_features, text_feature_names)
    

    def get_synopses_topic_model(self, text_features, num_topics=20):
        if path.exists('topic_models/synopses_model.joblib'):
            with open('topic_models/synopses_model.joblib', 'rb') as f:
                return joblib.load('topic_models/synopses_model.joblib')
        else:
            nmf = NMF(n_components=num_topics, random_state=self.seed)
            nmf.fit(text_features)
            joblib.dump(nmf, 'topic_models/synopses_model.joblib')
            return nmf
    
    def get_reviews_topic_model(self, text_features, num_topics=20):
        if path.exists('topic_models/reviews_model.joblib'):
            with open('topic_models/reviews_model.joblib', 'rb') as f:
                return joblib.load('topic_models/reviews_model.joblib')
        else:
            nmf = NMF(n_components=num_topics, random_state=self.seed)
            nmf.fit(text_features)
            joblib.dump(nmf, 'topic_models/reviews_model.joblib')
            return nmf


    def get_clusters(self, text_list, for_synopses=False, num_topics=20, words_per_topic=10):
        text_features, text_feature_names = self.get_text_features(text_list)

        if for_synopses:
            nmf = self.get_synopses_topic_model(text_features, num_topics)
        else:
            nmf = self.get_reviews_topic_model(text_features, num_topics)

        topic_clusters = []
        for idx, topic_vec in enumerate(nmf.components_):
            print(idx, end=' ')

            topic_words = []
            for fid in topic_vec.argsort()[-1:-words_per_topic-1:-1]:
                print(text_feature_names[fid], end=' ')
                topic_words.append(text_feature_names[fid])

            print()
            topic_clusters.append({str(idx) : topic_words})
        
        return topic_clusters
    

    def get_synopses_topics(self, num_topics=20, words_per_topic=10):
        print('SYNOPSES TOPICS:')
        return self.get_clusters(self.synopses_list, for_synopses=True, 
                num_topics=num_topics, words_per_topic=words_per_topic)
    

    def get_reviews_topics(self, num_topics=20, words_per_topic=10):
        print('REVIEWS TOPICS:')
        return self.get_clusters(self.reviews_list, for_synopses=False, 
                num_topics=num_topics, words_per_topic=words_per_topic)
    

if __name__ == '__main__':
    max_rec = 0x100000
    sys.setrecursionlimit(max_rec)

    book_info_obj = GoodreadsBookInfo(num_pages=5, num_books=500)
    all_books_info = book_info_obj.book_info_dict
    print(len(all_books_info))

    all_synopses = [book['synopsis'] for book in all_books_info]
    # print('all synopses: ', len(all_synopses))
    all_reviews = [review for book in all_books_info for review in book['reviews_text']]
    # print('all reviews: ', len(all_reviews))
    # print(all_reviews[77])

    book_analyzer = BookTextAnalyzer(all_synopses, all_reviews)
    # print('SYNOPSES TOPICS:')
    # book_analyzer.get_synopses_topics()

    # print('REVIEWS TOPICS:')
    # book_analyzer.get_reviews_topics()