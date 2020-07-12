from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.cluster import KMeans

import nltk
import pandas as pd

################################## helper functions ##################################

stemmer = PorterStemmer()
lemmatzer = WordNetLemmatizer()

def lemtz_tokens(tokens, lemmatzer):
    lemmatzed = []
    for item in tokens:
        lemmatzed.append(lemmatzer.lemmatize(item))
    return lemmatzed

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def nltk_tokenize_stemm(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def nltk_tokenize_lemtz(text):
    tokens = nltk.word_tokenize(text)
    lmtz = lemtz_tokens(tokens, lemmatzer) # find lemmas for tokenized words
    return lmtz

def main():
    ################################# Load your Text data #################################
    # local array of string-documents #################################

    corpus = """
    The students went to their new school yesterday.
    Azad was the best basketball player in the city.
    Next week the basketball league will start in Zakho. 
    Today the school is full of happy students for starting their new school year. 
    Best team in zakho will play with the best team from Dohuk. 
    Teachers were so excited seeing their students happy.
    Hello dear students! Cheered the school principle.
    Students started to play basketball from day one.
    """.split("\n")[1:-1]

    print("No of docs in cporpus:")
    corpus_len = len(corpus)
    print(corpus_len )

    print("\n original input corpus: ################################# \n")

    print(*corpus, sep="\n")

    corpus_len = len(corpus)

    ################################# Text Pre-processing #################################

    # Transforming the corpus into vector space using TF/IDF

    documents_tokens_dict={}
    for k, doc in enumerate(corpus):
        documents_tokens_dict[k]=doc

    print(documents_tokens_dict)

    # TfidfVectorizer: Convert a collection of raw documents to a matrix of TF-IDF features.
    # sklearn.feature_extraction.text.TfidfVectorizer

    sklearn_tfidf_matrix = TfidfVectorizer(
                                    lowercase=True,
                                    tokenizer=nltk_tokenize_lemtz,
                                    stop_words=stopwords.words('english')
                                    )

    ################################# Text Transformation #################################

    tf_idf_matrix = sklearn_tfidf_matrix.fit_transform(documents_tokens_dict.values())

    print("\n Term/Feature Names:  ################################# \n")
    print(sklearn_tfidf_matrix.get_feature_names())

    file_ids = list(documents_tokens_dict.keys())  # list of document names/ids

    print("\n File_IDSs: ", file_ids)

    print("tf_idf_matrix: ################################# \n ")

    print("number of features/term: ", len(sklearn_tfidf_matrix.get_feature_names()))  # list of term (features)

    print("number of docs: ", len(file_ids), "\n ")

    #printing tf/idf matrix
    for i in range(0, len(documents_tokens_dict)):
        print(file_ids[i],[round(w,3) for w in tf_idf_matrix.toarray()[i].tolist()])

    ################################# Text Mining #################################

    # Calculating Euclidean Distance between each document (measure of similarity)
    # Clustering the documents by k-means algorithm

    # KMeans: K-Means clustering Algorithm

    number_of_clusters = 2  # The number of clusters to form as well as the number of centroids to generate.

    km = KMeans(n_clusters=number_of_clusters)

    km.fit(tf_idf_matrix)

    print("\n km.fit :  \n ", km.fit)

    results = pd.DataFrame()
    results['text'] = corpus
    results['category'] = km.labels_

    print("Results: ################################# \n ", results)

    ################################# Evaluation #################################

    y_test_classes = [1, 0, 0, 1, 0, 1, 1, 1]
    predicted = km.labels_
    precision, recall, fscore, support = score(y_test_classes, predicted)

    print("Clustering Evaluation Results: ################################# \n ")
    print('Precision Metric: {}'.format(precision))
    print('Recall Metric: {}'.format(recall))
    print('F1score Metric: {}'.format(fscore))
    print('Support Metric: {}'.format(support))

main()
