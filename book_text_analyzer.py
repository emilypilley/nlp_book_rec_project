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

        self.synopses_model = None
        self.reviews_model = None
        
        self.synopses_topics = self.get_synopses_topics()
        self.reviews_topics = self.get_reviews_topics()

    def get_text_features(self, text_list, features, train=True):
        if train and path.exists('top_books_data/' + features + '_processed_text.p'):
            with open('top_books_data/' + features + '_processed_text.p', 'rb') as f:
                processed_text = pickle.load(f)
                print('loaded processed text')
        else:
            processed_text = []
            # lemmatize, remove non-alphabetic words and stopwords
            for text in self.spacy_nlp.pipe(text_list):
                only_alpha_nouns = ' '.join(token.lemma_ for token in text 
                                            if token.lemma_.isalpha() and not token.is_stop and token.pos_ == 'NOUN')
                processed_text.append(only_alpha_nouns)
            
            if train:
                with open('top_books_data/' + features + '_processed_text.p', 'wb') as f:
                    pickle.dump(processed_text, f)
            
        tf_idf_vec = TfidfVectorizer(max_features=5000, lowercase=True, tokenizer=self.spacy_nlp)

        text_features = tf_idf_vec.fit_transform(processed_text)
        # list of unique words found by vectorizer
        text_feature_names = tf_idf_vec.get_feature_names()

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
            text_features, text_feature_names = self.get_text_features(text_list, 'synopses')
            model = self.get_synopses_topic_model(text_features)
            words_per_topic = self.words_per_synopses_topic
        else:
            text_features, text_feature_names = self.get_text_features(text_list, 'reviews')
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
        print('\nSYNOPSES TOPICS:')
        return self.get_clusters(self.synopses_list, for_synopses=True)
    

    def get_reviews_topics(self):
        print('\nREVIEWS TOPICS:')
        return self.get_clusters(self.reviews_list, for_synopses=False)