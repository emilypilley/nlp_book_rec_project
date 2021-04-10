from os import path
import spacy
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
   

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
        print('\nSYNOPSES TOPICS:')
        return self.get_clusters(self.synopses_list, for_synopses=True, 
                num_topics=num_topics, words_per_topic=words_per_topic)
    

    def get_reviews_topics(self, num_topics=20, words_per_topic=10):
        print('\nREVIEWS TOPICS:')
        return self.get_clusters(self.reviews_list, for_synopses=False, 
                num_topics=num_topics, words_per_topic=words_per_topic)