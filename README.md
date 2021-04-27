# CIS4930 NLP Final Project
## Recommendation System for Books Based on Clustering Content/Topic and Aspect Based Opinion Mining
This system finds books related to one (or possibly more) selected books, based on the content and qualities of the book, rather than relying on a selection based on what other users have read. The project analyzes books from two perspectives - the content of the synopsis and the content of the reviews, using NLP techniques to gain insight from this text and recommend books that are similar to a given title. 

The content and general topic and subject matter of the books is determined by looking at its provided synopsis. Eeither LDA or NMF can be used as the clustering technique to group books that are similar in content, allowing for varying degrees of similarity to be considered.

To make the recommendations more specific than only relying on similarities among the synopses of the books, information is extracted from book reviews on Goodreads, to account for other aspects that might be similar between books, such as plot, character development, or writing style. Aspect based sentiment analysis techniques are used to group and extract opinions on these various aspects of books based on what readers focus on in their reviews, to get a better idea of what draws readers to these books.

With these analyses of a given book as input, the book recommendation system uses cosine similarity scores to find the books that are similar in both content and other attributes/qualities that were mentioned in the reviews, presenting several options for similar books, providing a range of recommendations ranked based on their similarity to various aspects of the given book.

## Instructions
Before running, the packages for the relevant imports must be installed. Outside of the standard list of external packages for assignments, this includes langdetect, spacy and vadersentiment, validators, and spaCy's 'en_core_web_sm' pipeline should be downloaded. 
* [Install spaCy - Documenation](https://spacy.io/usage)
* [Install spaCy en_core_web_sm model with conda](https://anaconda.org/conda-forge/spacy-model-en_core_web_sm)
* [Install VADER sentiment analysis with conda](https://anaconda.org/conda-forge/vadersentiment)
* [Install langdetect with conda](https://anaconda.org/conda-forge/langdetect)
* [Install validators with conda](conda install -c bioconda validators)

To run the project using existing models for the topic modeling/aspect keyword identification and reccomendations:
* Run the "run_project.py" file, inserting either the URL for the page of the book from Goodreads, or the title of one of the [top 500 books on Goodreads](https://www.goodreads.com/list/show/1.Best_Books_Ever?page=1), which is already included in the dataset. 
* The train_recommender_system() function is called with default values for the number of books to train on, model types for topic modeling, and number of topics to extract from the synopses and reviews, however if these defaults are changed to values corresponding with existing models (found under the topic_models directory), the model will not be retrained.

To train a new model for topic modeling/aspect keyword identification, and a new model for reccomendations to reflect these changes:
* Set the parameters of the train_recommender_system() function to new values that do not correspond with existing models in the topic_models directory, and run the project with your chosen books as described above. Note that changing the number of words per topic output will not retrain a new model, if that is the only difference from the default values.

## Resources:
* [A Friendly Introduction to Text Clustering](https://towardsdatascience.com/a-friendly-introduction-to-text-clustering-fa996bcefd04)
* [Harvard CS109A Webscraping Tutorial](https://harvard-iacs.github.io/2018-CS109A/labs/lab-2/scraping/student/)
* [Beyond the Stars - Jakob et al.](https://dl.acm.org/doi/pdf/10.1145/1651461.1651473?casa_token=zVVqi0EC7sUAAAAA:R2pPfxXXAp-iMLvddvSb46Lq2FCy-TRNVihyPpjFRfgyAYIGoEOsVRZ4Q56H0aG_ZlN7anzK1NGcfQ)
* [spaCy Linguistic Features Documentation](https://spacy.io/usage/linguistic-features)
* [NLP with Python: Topic Modeling Tutorial](https://sanjayasubedi.com.np/nlp/nlp-with-python-topic-modeling/)
* [Aspect Based Sentiment Analysis Example](https://towardsdatascience.com/aspect-based-sentiment-analysis-using-spacy-textblob-4c8de3e0d2b9)
* [VADER Sentiment Analysis Tutorial](https://www.geeksforgeeks.org/python-sentiment-analysis-using-vader/)
* [Movie Recommendation System Using Cosine Similarity](https://towardsdatascience.com/using-cosine-similarity-to-build-a-movie-recommendation-system-ae7f20842599)
* [Cosine Similarity and Handling Categorical Variables](https://medium.com/@rahulkuntala9/cosine-similarity-and-handling-categorical-variables-29f907951b5)

