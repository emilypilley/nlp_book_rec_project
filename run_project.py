import sys
from goodreads_book_info import GoodreadsBookInfo
from book_text_analyzer import BookTextAnalyzer
from review_topics_sentiment_analyzer import ReviewTopicsSentimentAnalyzer
from book_recommender import BookRecommender


def train_recommender_system(num_pages=5, num_books=500, num_synopses_topics=20, 
                            words_per_synopses_topic=10, synopses_model_type='lda', 
                            num_reviews_topics=5, words_per_reviews_topic=10, 
                            reviews_model_type='lda', verbose=True):
    '''Trains a new recommender system using Goodreads top book list data.
    
    Process starts with topic modeling/keyword identification, then performs
    aspect based sentiment analysis on review keywords, and uses these features
    to find most similar books bases on cosine similarity.
    
    If verbose=1, the topic lists are printed out during intermediate stages and
    the feature dictionaries for the first 5 books are printed.'''

    # obtain book information, synopses, and reviews from Goodreads
    # using webscraping techniques
    book_info_obj = GoodreadsBookInfo(num_pages, num_books)
    all_books_info = book_info_obj.book_info_dicts
    if verbose:
        print('\nTotal number of books: ', len(all_books_info))
    
    all_synopses = [book['synopsis'] for book in all_books_info]
    all_reviews = [review for book in all_books_info for review in book['reviews_text']]

    # Topic modeling for reviews and synopses and keyword identification for reviews
    book_analyzer = BookTextAnalyzer(all_synopses, all_reviews, 
        num_synopses_topics=num_synopses_topics, words_per_synopses_topic=words_per_synopses_topic, 
        synopses_model_type=synopses_model_type, num_reviews_topics=num_reviews_topics, 
        words_per_reviews_topic=words_per_reviews_topic, reviews_model_type=reviews_model_type)
    
    if verbose:
        print('\nSynopses Topics: ')
        for topic in book_analyzer.synopses_topics.keys():
            print(topic, [w for w, v in book_analyzer.synopses_topics[topic]])

        print('\nReviews Topics: ')
        for topic in book_analyzer.reviews_topics.keys():
            print(topic, [w for w, v in book_analyzer.reviews_topics[topic]])

    review_keywords_dict = book_analyzer.get_reviews_topics_keywords()
    if verbose:
        print('\nReduced Reviews Topic Keyword Lists: ')
        topic_keyword_dict = {}
        for word, (topic, _) in review_keywords_dict.items():
            if topic in topic_keyword_dict:
                topic_keyword_dict[topic].append(word)
            else:
                topic_keyword_dict[topic] = [word]

        for topic, word_list in topic_keyword_dict.items():
            print(topic, word_list)
        
    books_synopses_topics_dict = (
        book_analyzer.get_books_synopses_classifications(all_books_info))

    reviews_dict = book_analyzer.get_books_reviews_dict(all_books_info)

    # Perform aspect based sentiment analysis using review keywords found
    # during topic modeling in previous step
    review_sentiment_analyzer = (
        ReviewTopicsSentimentAnalyzer(books_reviews_dict=reviews_dict, 
                                    topics_keywords_dict=review_keywords_dict))
        
    # Use synopsis topic categorizations and review aspect sentiments to 
    # recommend new books based on similarity    
    book_recommender = (
        BookRecommender(all_books_info=all_books_info, book_text_analyzer=book_analyzer, 
                        review_topics_sentiment_analyzer=review_sentiment_analyzer))
            
    book_features_df = book_recommender.get_combined_synopsis_reviews_features_df()
    book_feature_dicts = book_recommender.books_features_dicts
    
    if verbose:
        print('\nTop 5 Books Feature Dictionaries')
        for book in book_feature_dicts[:5]:
            print(book)
        print()
    
    # print(book_recommender.get_books_features_for_recs())

    return book_info_obj, book_recommender
        

def recommend_books(new_book_url, book_info_obj, book_recommender, verbose=True):
    '''Returns reccomendations based on overall topic and opinions expressed in reviews.'''
    new_book = book_info_obj.get_book_info_from_url(new_book_url)
    new_book_synopsis = new_book['synopsis']
    new_book_reviews = new_book['reviews_text']

    if verbose:
        print(new_book['title'] + ' - ' + new_book['author'])
        print('\nSynopsis Topics:')
        print(book_recommender.book_text_analyzer.get_topics_from_synopsis(new_book_synopsis))
        print('\nReivew Topics + Sentiments:')
        print(book_recommender.review_topics_sentiment_analyzer.get_book_topic_sentiments(new_book_reviews))


if __name__ == '__main__':
    # Need this for pickling
    max_rec = 0x100000
    sys.setrecursionlimit(max_rec)

    # Will train a new recommender system if given new specifications,
    # or load existing data and models if they already exist
    book_info_obj, book_recommender = train_recommender_system()

    new_book_url = "https://www.goodreads.com/book/show/7723542-a-dog-s-purpose"
    recommend_books(new_book_url, book_info_obj, book_recommender)
    


    


    
    
