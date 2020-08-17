import numpy as np
from sklearn.datasets import load_files
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, f1_score, confusion_matrix


def load_data(path):
    data = load_files(path, load_content=True, encoding="utf-8")
    return data.data, data.target


def preprocess(corpus, ground_truth):
    preprocessed_corpus = []
    for i in range(len(corpus)):
        text = corpus[i]

        preprocessed_text = word_tokenize(text)

        new_text = []
        for word in preprocessed_text:
            new_text.append(word.lower())

        preprocessed_text = new_text

        new_text = []
        for word in preprocessed_text:
            if word.isalnum():
                new_text.append(word)
        preprocessed_text = new_text

        # convert to text
        final_text = ""
        for t in preprocessed_text:
            preprocessed_text += t + " "
        preprocessed_corpus.append(final_text)

    # Split train y test
    X_train, X_test, y_train, y_test = train_test_split(preprocessed_corpus, ground_truth)

    vectorizer = CountVectorizer()
    vectorizer.fit(X_train)
    X_train = vectorizer.transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    corpus, ground_truth = load_data("C:/Users/03953138/Desktop/Medina/NLP/aclImdb/train")
    X_train, X_test, y_train, y_test = preprocess(corpus, ground_truth)

    clf = LinearSVC(verbose=1, random_state=7, tol=1e-5, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print("\t\t\t\tNot neutral\tNeutral")
    print("Not neutral\t\t\t{}\t{}".format(cm[0, 0], cm[0, 1]))
    print("Neutral\t\t\t\t{}\t{}".format(cm[1, 0], cm[1, 1]))

