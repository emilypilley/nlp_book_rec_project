from os import path
import spacy
import joblib
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation as LDA
   

class BookTextAnalyzer():
    def __init__(self, synopses_list, reviews_list, num_synopses_topics=20, 
                words_per_synopses_topic=10, num_reviews_topics=20, 
                words_per_reviews_topic=10, seed=7):
        self.seed = seed
        self.spacy_nlp = spacy.load('en_core_web_sm')

        self.synopses_list = synopses_list
        self.reviews_list = reviews_list
        
        self.synopses_topics = self.get_synopses_topics(num_synopses_topics, words_per_synopses_topic)
        self.reviews_topics = self.get_reviews_topics(num_reviews_topics, words_per_reviews_topic)

    def get_text_features(self, text_list, features):
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
            
        tf_idf_vec = TfidfVectorizer(max_features=5000, lowercase=True, tokenizer=self.spacy_nlp)

        text_features = tf_idf_vec.fit_transform(processed_text)
        # list of unique words found by vectorizer
        text_feature_names = tf_idf_vec.get_feature_names()

        return (text_features, text_feature_names)
    

    def get_synopses_topic_model(self, text_features, num_topics=20, model_name='nmf'):
        if path.exists('topic_models/synopses_' + model_name + '_model_' + str(num_topics) + '_topics.joblib'):
            with open('topic_models/synopses_' + model_name + '_model_' + str(num_topics) + '_topics.joblib', 'rb') as f:
                return joblib.load(f)
        else:
            if model_name == 'nmf':
                model = NMF(n_components=num_topics, random_state=self.seed)
            elif model_name == 'lda':
                model = LDA(n_components=num_topics, random_state=self.seed)
            else:
                raise Exception("Invalid model name")
            model.fit(text_features)
            joblib.dump(model, 'topic_models/synopses_' + model_name + '_model_' + str(num_topics) + '_topics.joblib')
            return model
    
    def get_reviews_topic_model(self, text_features, num_topics=20, model_name='nmf'):
        if path.exists('topic_models/reviews_' + model_name + '_model_' + str(num_topics) + '_topics.joblib'):
            with open('topic_models/reviews_' + model_name + '_model_' + str(num_topics) + '_topics.joblib', 'rb') as f:
                return joblib.load(f)
        else:
            if model_name == 'nmf':
                model = NMF(n_components=num_topics, random_state=self.seed)
            elif model_name == 'lda':
                model = LDA(n_components=num_topics, random_state=self.seed)
            else:
                raise Exception("Invalid model name")
            model.fit(text_features)
            joblib.dump(model, 'topic_models/reviews_' + model_name + '_model_' + str(num_topics) + '_topics.joblib')
            return model


    def get_clusters(self, text_list, for_synopses=False, num_topics=20, words_per_topic=10):
        if for_synopses:
            text_features, text_feature_names = self.get_text_features(text_list, 'synopses')
            model = self.get_synopses_topic_model(text_features, num_topics, model_name='lda')
        else:
            text_features, text_feature_names = self.get_text_features(text_list, 'reviews')
            model = self.get_reviews_topic_model(text_features, num_topics, model_name='lda')

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
    

    def get_synopses_topics(self, num_topics=20, words_per_topic=10):
        print('\nSYNOPSES TOPICS:')
        return self.get_clusters(self.synopses_list, for_synopses=True, 
                num_topics=num_topics, words_per_topic=words_per_topic)
    

    def get_reviews_topics(self, num_topics=20, words_per_topic=10):
        print('\nREVIEWS TOPICS:')
        return self.get_clusters(self.reviews_list, for_synopses=False, 
                num_topics=num_topics, words_per_topic=words_per_topic)