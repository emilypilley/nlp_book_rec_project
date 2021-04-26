from os import path
import spacy
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class ReviewTopicsSentimentAnalyzer():
    def __init__(self, books_reviews_dict, topics_keywords_dict):
        self.books_reviews_dict = books_reviews_dict
        self.topics_keywords_dict = topics_keywords_dict

        self.spacy_nlp = spacy.load('en_core_web_sm')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
    
    def get_reivew_aspects_sentiments(self, review_text):
        '''Returns tuples of each topics addressed in the review and its corresponding sentiment score
        '''
        keyword_mod_list = []
        review = self.spacy_nlp(review_text)
        for sent in review.sents:
            keywords = []
            modifiers = []
            negation = False
            for token in sent:
                if token.lemma_.isalpha(): # and not token.is_stop:
                    # if token.pos_ == 'NOUN':
                    if token.dep_ == 'nsubj':
                        if token.lemma_ in self.topics_keywords_dict:
                            keywords.append(token.text)
                    elif token.dep_ == 'neg':
                        negation = True
                    if token.dep_ == 'amod' or token.pos_ == 'ADJ':
                        adv_mods = []
                        for child in token.children:
                            if child.dep_ == 'advmod':
                                adv_mods.append(child.text)
                        if len(adv_mods) > 0:
                            modifiers.append(token.text + ' ' + ' '.join(adv_mods))
                        else:
                            modifiers.append(token.text)
            if len(keywords) > 0 and len(modifiers) > 0:
                for keyword in keywords:
                    keyword_mod_list.append((keyword, modifiers, negation))
        
        return keyword_mod_list


    def get_all_books_reivews_aspects_sentiments(self):
        '''For each book, finds the average sentiment for each topic if topic is mentioned in more than two reviews'''
        pass