import pandas as pd
import math

def compute_tf(text):
    """
        - For a single document, analyse how frequently a term appears in the document.
        :param text: the document
        :return: dicitonary of term to frequency
    """

    words = text.split()
    word_count = len(words)

    tf = {}

    for word in words:
        # Normalise by dividing by word count.
        tf[word] = tf.get(word, 0) + 1 / word_count

    return tf

def compute_idf(documents):
    """
        Computes the inverse document frequency - essentially determining how important a word is in the document.
        This inverse document frequency, looks at the entire corpus and looks at words like "the", "and", "because"
        which might appear many times in the corpus, and reduce the weight associated with them.

        :param documents: the entire corpus
        :return: a dictionary of the idf for the corpus.
    """

    N = len(documents)
    idf = {}

    for document in documents:
        for word in document.split():
            idf[word] = idf.get(word, 0) + 1

    for word, val in idf.items():
        idf[word] = math.log(N / float(val)) # log is used for handling large values

    return idf


def compute_tfidf(corpus):
    """
        Calculates the tf_idf for the entire corpus.

        :param corpus:
        :return:
    """
    # Compute IDF (needs to be done once for the whole corpus)
    idf = compute_idf(corpus)

    # Compute TF-IDF for each document
    tfidf_corpus = []
    for document in corpus:
        tf = compute_tf(document)
        tfidf = {word: tf[word] * idf[word] for word in tf}
        tfidf_corpus.append(tfidf)
    return tfidf_corpus


if __name__ == "__main__":
    file_path = "../train_data/dontpatronizeme_pcl.tsv"
    df = pd.read_csv(file_path, sep='\t', header=None,
                     names=['paragraph-id', 'keyword', 'countrycode', "paragraph", "label"])
    df_filtered = df[df['paragraph'].notna()]
    corpus = list(df_filtered['paragraph'])
    tf_idf = compute_tfidf(corpus)
    df_filtered['tf_idf'] = tf_idf
    print(df_filtered)
    df_filtered.to_csv("dontpatronize_tf_idfs.csv")
