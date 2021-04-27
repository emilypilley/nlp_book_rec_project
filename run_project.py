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
        BookRecommender(book_info_obj=book_info_obj, book_text_analyzer=book_analyzer, 
                        review_topics_sentiment_analyzer=review_sentiment_analyzer))
            
    book_features_df = book_recommender.get_combined_synopsis_reviews_features_df()
    book_feature_dicts = book_recommender.books_features_dicts

    if verbose:
        print('\nTop 5 Books Feature Dictionaries')
        for book in book_feature_dicts[:5]:
            print(book)
        print()
    
    book_recommender.group_similar_books()

    return book_info_obj, book_recommender
        

if __name__ == '__main__':
    # Need this for pickling
    max_rec = 0x100000
    sys.setrecursionlimit(max_rec)

    # Will train a new recommender system if given new specifications,
    # or load existing data and models if they already exist
    book_info_obj, book_recommender = train_recommender_system()

    # The following is an example of getting recommendations for a book that
    # already exists in the dataset - the (first part) of the name must be an
    # exact match with the title of the book in the dataset
    new_book_name = 'Harry Potter and the Order of the Phoenix'
    rec_list_1 = book_recommender.find_top_n_recommendations(new_book_name)

    print('Books similar to ', new_book_name)
    for book in rec_list_1:
        print(book)
    
    # The following are books that are not currently in the dataset, which is ensured 
    # because they were obtained from later in the Goodread's most popular books list
    new_book_url_list = ["https://www.goodreads.com/book/show/7723542-a-dog-s-purpose",
                        "https://www.goodreads.com/book/show/3869.A_Brief_History_of_Time",
                        "https://www.goodreads.com/book/show/7631105-the-scorch-trials",
                        "https://www.goodreads.com/book/show/1371.The_Iliad",
                        "https://www.goodreads.com/book/show/3063499-the-lucky-one"
                        ]
    for book_url in new_book_url_list:
        rec_list_2 = book_recommender.find_top_n_recommendations(book_url)
        print('\nBooks similar to ', book_url)
        for book in rec_list_2:
            print(book)    


    


    
    
