from os import path
import spacy
import numpy as np
import pandas as pd
import joblib
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation as LDA
   

class BookTextAnalyzer():
    def __init__(self, synopses_list, reviews_list, num_synopses_topics=20, 
                words_per_synopses_topic=10, num_reviews_topics=20, 
                words_per_reviews_topic=10, synopses_model_type='nmf', reviews_model_type='nmf',seed=7):
        self.seed = seed

        self.num_synopses_topics = num_synopses_topics
        self.words_per_synopses_topic = words_per_synopses_topic
        self.synopses_model_type = synopses_model_type

        self.num_reviews_topics = num_reviews_topics
        self.words_per_reviews_topic = words_per_reviews_topic
        self.reviews_model_type = reviews_model_type

        self.spacy_nlp = spacy.load('en_core_web_sm')

        self.synopses_list = synopses_list
        self.reviews_list = reviews_list

        self.tf_idf_syn_vec = None
        self.tf_idf_rev_vec = None
        self.synopses_model = None
        self.reviews_model = None
        
        self.synopses_topics = self.get_synopses_topics()
        self.reviews_topics = self.get_reviews_topics()


    def get_training_text_features(self, text_list, features):
        if path.exists('top_books_data/' + features + '_processed_text.p'):
            with open('top_books_data/' + features + '_processed_text.p', 'rb') as f:
                processed_text = pickle.load(f)
        else:
            processed_text = []
            # lemmatize, remove non-alphabetic words and stopwords, only keep nouns
            for text in self.spacy_nlp.pipe(text_list):
                only_alpha_nouns = ' '.join(token.lemma_ for token in text 
                                            if token.lemma_.isalpha() and not token.is_stop and token.pos_ == 'NOUN')
                processed_text.append(only_alpha_nouns)

            with open('top_books_data/' + features + '_processed_text.p', 'wb') as f:
                pickle.dump(processed_text, f)
            
        if features == 'synopses':
            self.tf_idf_syn_vec = TfidfVectorizer(max_features=5000, lowercase=True)
            self.tf_idf_syn_vec.fit(processed_text)
            text_features = self.tf_idf_syn_vec.transform(processed_text)
            text_feature_names = self.tf_idf_syn_vec.get_feature_names()

        elif features == 'reviews':
            self.tf_idf_rev_vec = TfidfVectorizer(max_features=5000, lowercase=True)
            self.tf_idf_rev_vec.fit(processed_text)
            text_features = self.tf_idf_rev_vec.transform(processed_text)
            text_feature_names = self.tf_idf_rev_vec.get_feature_names()

        return (text_features, text_feature_names)
    

    def get_synopses_topic_model(self, text_features):
        if path.exists('topic_models/synopses_' + self.synopses_model_type + '_model_' + str(self.num_synopses_topics) + '_topics.joblib'):
            with open('topic_models/synopses_' + self.synopses_model_type + '_model_' + str(self.num_synopses_topics) + '_topics.joblib', 'rb') as f:
                model = joblib.load(f)
        else:
            if self.synopses_model_type == 'nmf':
                model = NMF(n_components=self.num_synopses_topics, random_state=self.seed)
            elif self.synopses_model_type == 'lda':
                model = LDA(n_components=self.num_synopses_topics, random_state=self.seed)
            else:
                raise Exception("Invalid model name")
            model.fit(text_features)
            joblib.dump(model, 'topic_models/synopses_' + self.synopses_model_type + '_model_' + str(self.num_synopses_topics) + '_topics.joblib')
        self.synopses_model = model

        return model
    
    def get_reviews_topic_model(self, text_features):
        if path.exists('topic_models/reviews_' + self.reviews_model_type + '_model_' + str(self.num_reviews_topics) + '_topics.joblib'):
            with open('topic_models/reviews_' + self.reviews_model_type + '_model_' + str(self.num_reviews_topics) + '_topics.joblib', 'rb') as f:
                model = joblib.load(f)
        else:
            if self.reviews_model_type == 'nmf':
                model = NMF(n_components=self.num_reviews_topics, random_state=self.seed)
            elif self.reviews_model_type == 'lda':
                model = LDA(n_components=self.num_reviews_topics, random_state=self.seed)
            else:
                raise Exception("Invalid model name")
            model.fit(text_features)
            joblib.dump(model, 'topic_models/reviews_' + self.reviews_model_type + '_model_' + str(self.num_reviews_topics) + '_topics.joblib')
        self.reviews_model = model
        return model


    def get_clusters(self, text_list, for_synopses=False):
        if for_synopses:
            text_features, text_feature_names = self.get_training_text_features(text_list, 'synopses')
            model = self.get_synopses_topic_model(text_features)
            words_per_topic = self.words_per_synopses_topic
        else:
            text_features, text_feature_names = self.get_training_text_features(text_list, 'reviews')
            model = self.get_reviews_topic_model(text_features)
            words_per_topic = self.words_per_reviews_topic

        topic_clusters = {}

        for idx, topic_vec in enumerate(model.components_):
            topic_words = []
            for fid in topic_vec.argsort()[-1:-words_per_topic-1:-1]:
                topic_words.append((text_feature_names[fid], topic_vec[fid]))
            topic_clusters[str(idx)] = topic_words

        return topic_clusters
    

    def get_synopses_topics(self):
        '''Topic modeling for book synopses'''
        return self.get_clusters(self.synopses_list, for_synopses=True)
    

    def get_reviews_topics(self):
        '''Topic modeling for book reviews'''
        return self.get_clusters(self.reviews_list, for_synopses=False)
    

    def get_topic_classification_features(self, synopsis):
        only_alpha_nouns = []
        for token in self.spacy_nlp(synopsis):
            if token.lemma_.isalpha() and not token.is_stop and token.pos_ == 'NOUN':
                only_alpha_nouns .append(token.lemma_)
                    
        processed_text = ' '.join(only_alpha_nouns)
        return self.tf_idf_syn_vec.transform([processed_text])

    def get_topics_from_synopsis(self, synopsis_text):
        '''Gets the most relevant topics for a single book's synopsis'''
        synopsis_features = self.get_topic_classification_features(synopsis_text)
        output = np.squeeze(self.synopses_model.transform(synopsis_features))
        top_book_topics = output.argsort(axis=0)[::-1]
        most_relevant_topics = []
        for idx, topic in enumerate(top_book_topics):
            # always add at least top topic - for NMF model probabilities are lower
            if idx == 0:
                most_relevant_topics.append((topic, output[topic]))
            # if there is a second or third relevant topic, add it as well
            elif output[topic] >= 0.2:
                most_relevant_topics.append((topic, output[topic]))
        return most_relevant_topics

    def get_books_synopses_classifications(self, all_books_info_dicts):
        '''Builds a dictionary containg each book and the topics matched most closely by its synopses.
        
        Entries in the dictionary are keys, consiting of the title and author name, and a list of the
        most relevant topics for that book, which contains tuples of the topic number and relevancy of it.
        '''
        all_books_synopses_topics = {}
        for book in all_books_info_dicts:
            synopsis = book['synopsis']
            topics = self.get_topics_from_synopsis(synopsis)
            title_author_str = book['title'] + '-' + book['author']
            all_books_synopses_topics[title_author_str.replace(' ', '_')] = topics
        return all_books_synopses_topics
    

    def get_books_reviews_dict(self, all_books_info_dicts):
        books_reviews_dict = {}
        for book in all_books_info_dicts:           
            title_author_str = book['title'] + '-' + book['author']
            books_reviews_dict[title_author_str.replace(' ', '_')] = book['reviews_text']
        return books_reviews_dict

    
    def get_removal_indices_of_word(self, target, word_list, start_idx=0):
        target_idxs = []
        highest_importance = 0.0
        highest_importance_idx = -1
        for idx, (word, val, topic) in enumerate(word_list[start_idx:]):
            real_idx = idx + start_idx
            if word == target and val > highest_importance:
                highest_importance = val
                highest_importance_idx = real_idx
                target_idxs.append(real_idx)
            elif word == target:
                target_idxs.append(real_idx)
        if highest_importance_idx > -1:
            target_idxs.remove(highest_importance_idx)
        return target_idxs


    def get_reviews_topics_keywords(self):
        '''Removes duplicates from review topic keywords lists, keeping it on most relevant list'''
        reviews_topics_keywords_list = []
        for topic in self.reviews_topics.keys():
            reviews_topics_keywords_list.extend([(w, v, int(topic)) for (w, v) in self.reviews_topics[topic]])
        
        for idx in range(len(reviews_topics_keywords_list)):
            word, val, topic = reviews_topics_keywords_list[idx]
            repeats = self.get_removal_indices_of_word(word, reviews_topics_keywords_list, idx)
            if repeats:
                reviews_topics_keywords_list = [(w, v, t) for i, (w, v, t) in enumerate(reviews_topics_keywords_list) 
                                                if i not in repeats and w.isascii()]
            if idx == len(reviews_topics_keywords_list) - 1:
                break

        reviews_topics_keywords_reduced = {}
        for (w, v, t) in reviews_topics_keywords_list:
            reviews_topics_keywords_reduced[w] = (t, v)

        return reviews_topics_keywords_reduced
    