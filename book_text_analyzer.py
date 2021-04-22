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
        # self.spacy_similarity_nlp = spacy.load('en_core_web_lg')

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
            # lemmatize, remove non-alphabetic words and stopwords
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

        topic_clusters = []

        for idx, topic_vec in enumerate(model.components_):
            print(idx, end=' ')

            topic_words = []
            for fid in topic_vec.argsort()[-1:-words_per_topic-1:-1]:
                print(text_feature_names[fid], end=' ')
                topic_words.append(text_feature_names[fid])

            print()
            topic_clusters.append({str(idx) : topic_words})

        return topic_clusters
    

    def get_synopses_topics(self):
        '''Topic modeling for book synopses'''
        print('\nSYNOPSES TOPICS:')
        return self.get_clusters(self.synopses_list, for_synopses=True)
    

    def get_reviews_topics(self):
        '''Topic modeling for book reviews'''
        print('\nREVIEWS TOPICS:')
        return self.get_clusters(self.reviews_list, for_synopses=False)
    

    def get_topic_classification_features(self, synopsis):
        only_alpha_nouns = []
        for token in self.spacy_nlp(synopsis):
            if token.lemma_.isalpha() and not token.is_stop and token.pos_ == 'NOUN':
                only_alpha_nouns .append(token.lemma_)
                    
        processed_text = ' '.join(only_alpha_nouns)
        return self.tf_idf_syn_vec.transform([processed_text])

    def get_topics_from_synopsis(self, synopsis_text):
        synopsis_features = self.get_topic_classification_features(synopsis_text)
        output = np.squeeze(self.synopses_model.transform(synopsis_features))
        top_book_topics = output.argsort(axis=0)[::-1]
        most_relevant_topics = []
        for idx, topic in enumerate(top_book_topics):
            # always add at least top topic - for NMF model probabilities are lower
            if idx == 0:
                most_relevant_topics.append((topic, output[topic]))
                print('topic: ', topic, 'prob: ', output[topic])
            # if there is a second or third relevant topic, add it as well
            elif output[topic] >= 0.2:
                most_relevant_topics.append((topic, output[topic]))
                print('topic: ', topic, 'prob: ', output[topic])
        return most_relevant_topics

    def get_books_synopses_classifications(self, all_books_info_dicts):
        '''Builds a dictionary containg each book and the topics matched most closely by its synopses'''
        all_books_synopses_topics = {}
        print('\n\nBook Synopses Topics: ')
        for book in all_books_info_dicts[:20]:
            print(book['title'] + ' - ' + book['author'])
            synopsis = book['synopsis']
            topics = self.get_topics_from_synopsis(synopsis)
            title_author_str = book['title'] + '-' + book['author']
            all_books_synopses_topics[title_author_str.replace(' ', '_')] = topics
        return all_books_synopses_topics
    

    ##### TODO: FINISH IMPLEMENTING AND TESTING ##########
    # def get_n_most_similar_words(self, word, n=5):
    #     word = self.spacy_similarity_nlp.vocab[word]
    #     queries = [
    #         w for w in word.vocabif w.is_lower == word.is_lower and w.prob >= -15 and np.count_nonzero(w.vector)
    #     ]

    #     by_similarity = sorted(queries, key=lambda w: word.similarity(w), reverse=True)
    #     return [(w.lower_, w.similarity(word)) for w in by_similarity[:n+1] if w.lower_ != word.lower_]

    # def get_reviews_topics_keywords(self):
    #     '''Builds a dictionary for each review topic identified containing the list of keywords associated with it'''
    #     reviews_topics_keywords_dict = self.reviews_topics[topic]
    #     for topic in reviews_topics_keywords_dict:
    #         for word in self.reviews_topics[topic]:
    #             # find lemmas of 5 most similar words to this word and add as keywords
    #             keywords = self.get_n_most_similar_words(word, n=5)
    #             reviews_topics_keywords_dict[topic].extend(keywords)

    #     return reviews_topics_keywords_dict
    