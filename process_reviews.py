import requests
import time
import os
from os import path
import re
import pickle
from bs4 import BeautifulSoup

class GoodreadsBookInfo():
    def __init__(self, num_pages=5, num_books=500):
        assert num_books <= 100 * num_pages
        self.URLSTART = 'https://goodreads.com'
        self.num_pages = num_pages

        self.download_top_book_pages_html(num_pages)
        self.list_of_book_urls = self.get_top_book_urls(num_pages)

        # print('book urls 1-10: ', self.list_of_book_urls[:10])

        self.book_info_dict = self.get_top_n_books_info(num_books)


    def download_top_book_pages_html(self, num_pages):
        BESTBOOKS = '/list/show/1.Best_Books_Ever?page='

        for i in range(1, num_pages + 1):
            if not path.exists('top_books_data/page' + '%02d' % i + '.html'):
                # get html for the the top books on Goodreads (100 books per page)
                bookpage = str(i)
                stuff = requests.get(self.URLSTART + BESTBOOKS + bookpage)
                file_to_write = 'top_books_data/page' + '%02d' % i + '.html'
                print('File to Write', file_to_write)
                fd = open(file_to_write, 'w+')
                fd.write(stuff.text)
                fd.close()
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
                print('File to read: ', file_to_read)

                with open(file_to_read) as fdr:
                    data = fdr.read()
                soup = BeautifulSoup(data, 'html.parser')

                for e in soup.select('.bookTitle'):
                    book_url_list.append(self.URLSTART + e['href'])
                
            fd = open('top_books_data/url_list.txt', 'w')
            fd.write('\n'.join(book_url_list))
            fd.close()
        
            return book_url_list


    def get_top_n_books_info(self, num_books):
        if path.exists('top_books_data/book_info_dicts.p'):
            return pickle.load(open('top_books_data/book_info_dicts.p', 'rb'))
        else:
            book_info_dicts = []
            for i in range(num_books):
            # for i in range(3):
                book_info = {}
                book_page_url = self.list_of_book_urls[i]

                try:
                  book_page_html = requests.get(book_page_url)
                  book_data = BeautifulSoup(book_page_html.text, 'html.parser')

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
                  book_info['synopsis']= reviews[0]
                  book_info['reviews_text'] = [review.get_text() for review in reviews[1:]]

                  book_info_dicts.append(book_info)
                  print("Got: ", book_data.select_one("meta[property='og:title']")['content'])

                # except requests.exceptions.ConnectionError:
                except Exception:
                  print("Error with: ", book_page_url)

                time.sleep(5)
            
            pickle.dump(book_info_dicts, open('top_books_data/book_info_dicts.p', 'wb'))
            return book_info_dicts