# CIS4930 NLP Final Project
## Recommendation System for Books Based on Clustering Content/Topic and Aspect Based Opinion Mining
This system finds books related to one (or possibly more) selected books, based on the content and qualities of the book, rather than relying on a selection based on what other users have read. The project will employ two main methods to recommend books similar to a given title, using a graph based technique. 

The lexical content and general topic and subject matter of the books is determined by looking at the provided synopsis. A clustering technique is used to group books that are similar in content, allowing for varying degrees of similarity to be considered.

To make the recommendations more specific than only relying on similarities among the content or genre of the book, information is extracted from book reviews on Goodreads, to account for other aspects that might be similar between books, such as plot, character development, or writing style. Aspect based opinion mining techniques are used to group and extract opinions on these various aspects of books based on what readers focus on in their reviews, to get a more full idea of what draws readers to these books.

With these analyses of a given book as input, a clustering algorithm groups together books that are similar in both content and other key qualities, presenting several options for similar books, providing a range of recommendations ranked based on their similarity to the given title.


# Resources:
* [Harvard CS109A Webscraping Tutorial](https://harvard-iacs.github.io/2018-CS109A/labs/lab-2/scraping/student/)
* [Blog - NLP with Python: Topic Modeling Tutorial](https://sanjayasubedi.com.np/nlp/nlp-with-python-topic-modeling/)
* [Beyond the Stars - Jakob et al.](https://dl.acm.org/doi/pdf/10.1145/1651461.1651473?casa_token=zVVqi0EC7sUAAAAA:R2pPfxXXAp-iMLvddvSb46Lq2FCy-TRNVihyPpjFRfgyAYIGoEOsVRZ4Q56H0aG_ZlN7anzK1NGcfQ)
