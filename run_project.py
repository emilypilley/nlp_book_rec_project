import sys
from goodreads_book_info import GoodreadsBookInfo
from book_text_analyzer import BookTextAnalyzer

if __name__ == '__main__':
    max_rec = 0x100000
    sys.setrecursionlimit(max_rec)

    book_info_obj = GoodreadsBookInfo(num_pages=5, num_books=500)
    all_books_info = book_info_obj.book_info_dicts
    print('Total nubmer of books: ', len(all_books_info))

    all_synopses = [book['synopsis'] for book in all_books_info]
    all_reviews = [review for book in all_books_info for review in book['reviews_text']]
    
    book_analyzer = BookTextAnalyzer(all_synopses, all_reviews, num_synopses_topics=20, words_per_synopses_topic=10, 
                                    synopses_model_type='lda', num_reviews_topics=5, words_per_reviews_topic=10, 
                                    reviews_model_type='lda')
    
    print('\nSynopses Topics: ')
    for topic in book_analyzer.synopses_topics.keys():
        print(topic, [w for w, v in book_analyzer.synopses_topics[topic]])
        print('\n')

    new_book = book_info_obj.get_book_info_from_url('https://www.goodreads.com/book/show/7723542-a-dog-s-purpose')
    new_book_synopsis = new_book['synopsis']
    print('Topics of ', new_book['title'])
    print(book_analyzer.get_topics_from_synopsis(new_book_synopsis))

    print('\n\nReviews Topics: ')
    for topic in book_analyzer.reviews_topics.keys():
        print(topic, [w for w,v in book_analyzer.reviews_topics[topic]])

    all_books_synopses_topics = book_analyzer.get_books_synopses_classifications(all_books_info)
    # print('\n\nALL BOOKS SYNOPSES TOPICS (classified):')
    # for idx, book in enumerate(all_books_synopses_topics.items()):
    #     if idx > 20:
    #         break
    #     print(book)
    
    print('\n\nReduced Keyword Set:')
    review_keywords = book_analyzer.get_reviews_topics_keywords()
    for topic in review_keywords.keys():
        print(topic, ':', review_keywords[topic])
    
    reviews_dict = book_analyzer.get_books_reviews_dict(all_books_info)



    ################## NOT IMPLEMENTED YET ####################

    # review_sentiment_analyzer = ReviewTopicsSentimentAnalyzer(books_reviews_dict, review_topics_keywords_dict)
    
