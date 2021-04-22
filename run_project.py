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

    all_books_synopses_topics = book_analyzer.get_books_synopses_classifications(all_books_info)
    print('ALL BOOKS SYNOPSES TOPICS:')
    for idx, book in enumerate(all_books_synopses_topics.items()):
        if idx > 20:
            break
        print(book)

    ################## NOT IMPLEMENTED YET ####################


    # print('\n\nBook Synopses Topics:\n', all_books_synopses_topics)

    # review_topics_keywords_dict = book_analyzer.get_reviews_topics_keywords()
    # print('\n\nReview Topics Keywords:\n', review_topics_keywords_dict)

    # books_reviews_dict = {}
    # for book in all_books_info:
        # title_author_str = book['title'] + '-' + book['author']
    #     books_reviews_dict[title_author_str.replace(' ', '_')] = book['reviews_text']

    # review_sentiment_analyzer = ReviewTopicsSentimentAnalyzer(books_reviews_dict, review_topics_keywords_dict)
    
