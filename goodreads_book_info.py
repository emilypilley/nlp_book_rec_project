import requests
import time
from os import path
import re
import pickle
from bs4 import BeautifulSoup
import langdetect


class GoodreadsBookInfo():
    def __init__(self, num_pages=5, num_books=500):
        assert num_books <= 100 * num_pages
        self.URLSTART = 'https://goodreads.com'
        self.num_pages = num_pages

        self.download_top_book_pages_html(num_pages)
        self.list_of_book_urls = self.get_top_book_urls(num_pages)

        self.book_info_dicts = self.get_top_n_books_info(num_books)


    def download_top_book_pages_html(self, num_pages):
        BESTBOOKS = '/list/show/1.Best_Books_Ever?page='

        for i in range(1, num_pages + 1):
            if not path.exists('top_books_data/page' + '%02d' % i + '.html'):
                # get html for the the top books on Goodreads (100 books per page)
                bookpage = str(i)
                top_books_html = requests.get(self.URLSTART + BESTBOOKS + bookpage)
                file_to_write = 'top_books_data/page' + '%02d' % i + '.html'
                with open(file_to_write, 'w+') as f:
                    f.write(top_books_html.text)
                time.sleep(2)


    def get_top_book_urls(self, num_pages):
        if path.exists('top_books_data/url_list.txt'):
            url_list_file = open('top_books_data/url_list.txt','r')
            return url_list_file.read().splitlines()
        else:
            book_url_list = []
            for i in range(1, num_pages + 1):
                pg_num = '%02d' % i
                file_to_read = 'top_books_data/page' + pg_num+ '.html'

                with open(file_to_read) as fdr:
                    data = fdr.read()
                soup = BeautifulSoup(data, 'html.parser')

                for e in soup.select('.bookTitle'):
                    book_url_list.append(self.URLSTART + e['href'])
                
            with open('top_books_data/url_list.txt', 'w') as f:
                f.write('\n'.join(book_url_list))

            return book_url_list


    def get_book_info_from_url(self, book_page_url):
        book_info = {}
        book_page_html = requests.get(book_page_url)
        book_data = BeautifulSoup(book_page_html.text, 'html.parser')

        if book_data.select_one("div[itemprop='inLanguage']").text == 'English':
            book_info['title'] = book_data.select_one("meta[property='og:title']")['content']
            book_info['isbn'] = book_data.select_one("meta[property='books:isbn']")['content']

            author_url = book_data.select_one("meta[property='books:author']")['content']
            author_page_html = requests.get(author_url)
            author_data = BeautifulSoup(author_page_html.text, 'html.parser')
            book_info['author'] = author_data.select_one("meta[property='og:title']")['content']

            rating = book_data.select_one("span[itemprop='ratingValue']").text
            book_info['rating'] = re.findall(r"\d\.\d+", rating)[0]

            book_info['total_rating_count'] = book_data.select_one("meta[itemprop='ratingCount']")['content']
            book_info['total_review_count'] = book_data.select_one("meta[itemprop='reviewCount']")['content']

            # first item is actually synopsis, rest are reviews
            reviews = book_data.find_all(id=re.compile("freeText\d+"))
            book_info['synopsis']= reviews[0].get_text()
            book_info['reviews_text'] = [review.get_text() for review in reviews[1:] 
                                        if langdetect.detect(review.get_text()) == 'en']

            print("Got: ", book_data.select_one("meta[property='og:title']")['content'])

        return book_info


    def get_top_n_books_info(self, num_books):
        if path.exists('top_books_data/book_info_dicts.p'):
            with open('top_books_data/book_info_dicts.p', 'rb') as f:
                return pickle.load(f)
        else:
            book_info_dicts = []
            for i in range(num_books):
                book_page_url = self.list_of_book_urls[i]
                try:
                    book_info_dict = self.get_book_info_from_url(book_page_url)
                    if len(book_info_dict) > 0:
                        book_info_dicts.append(book_info_dict)
                except Exception:
                    try:
                        book_info_dict = self.get_book_info_from_url(book_page_url)
                        if len(book_info_dict) > 0:
                            book_info_dicts.append(book_info_dict)
                    except Exception:
                        try:
                            book_info_dict = self.get_book_info_from_url(book_page_url)
                            if len(book_info_dict) > 0:
                                book_info_dicts.append(book_info_dict)
                        except Exception:
                            print("Error with: ", book_page_url)

                time.sleep(2)
            
            with open('top_books_data/book_info_dicts.p', 'wb') as f:
                pickle.dump(book_info_dicts, f)

            return book_info_dicts
    