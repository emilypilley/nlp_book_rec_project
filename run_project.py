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

    book_analyzer = BookTextAnalyzer(all_synopses, all_reviews, num_synopses_topics=35, words_per_synopses_topic=5, 
                                    synopses_model_type='lda', num_reviews_topics=6, words_per_reviews_topic=5, 
                                    reviews_model_type='lda')

    synopses_topics = book_analyzer.synopses_topics
    # print(synopses_topics[0])

    reviews_topics = book_analyzer.reviews_topics
    # print(reviews_topics[0])
