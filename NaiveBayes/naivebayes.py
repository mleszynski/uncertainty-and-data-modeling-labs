# Naive Bayes
# Marcelo Leszynski


import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter
from scipy.stats import poisson
from sklearn.base import ClassifierMixin
from scipy.optimize import minimize, Bounds
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


# Problem 1-5 ##################################################################
class NaiveBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages in to spam or ham.
    '''
    def __init__(self):
        return


    def fit(self, X, y):
        '''
        Create a table that will allow the filter to evaluate P(H), P(S)
        and P(w|C)

        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels
        '''
        # initialize data structures ###########################################
        unique_words = set()
        for value in X.str.split():
            unique_words.update(set(value))
        unique_words = list(unique_words)
        array = np.zeros((2,len(unique_words)))
        self.data = pd.DataFrame(data=array, columns=unique_words)
        self.data.insert(0, 'Label', ['ham','spam'])
        self.data.set_index('Label', inplace=True)

        # fill in data into dataframe ##########################################
        for i in X.index:
            if y[i] == 'ham':
                for word in X[i].split():
                    self.data.loc['ham', word] += 1
            else:
                for word in X[i].split():
                    self.data.loc['spam',word] += 1

        # get total ham words and spam words ###################################
        self.total_h_words = self.data.loc['ham'].sum()
        self.total_s_words = self.data.loc['spam'].sum()

        # get unique words to ham and spam #####################################
        self.h_words = self.data.loc['ham', self.data.loc['ham'] > 0].index.tolist()
        self.s_words = self.data.loc['spam',self.data.loc['spam']> 0].index.tolist()
        self.P_h = y.value_counts(normalize=True)['ham']
        self.P_s = y.value_counts(normalize=True)['spam']


    def predict_proba(self, X):
        '''
        Find P(C=k|x) for each x in X and for each class k by computing
        P(C=k)P(x|C=k)

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Probability each message is ham, spam
                0 column is ham
                1 column is spam
        '''
        # populate array of predictions ########################################
        predictions = np.zeros((X.shape[0],2))
        for i,message in enumerate(X.str.split()):
            h_prod = []
            s_prod = []
            counts = Counter(message)
            for word in counts:
                if (word in self.h_words) and (word in self.s_words):
                    h_prod.append((self.data.loc['ham', word]/self.total_h_words) ** counts[word])
                    s_prod.append((self.data.loc['spam',word]/self.total_s_words) ** counts[word])
            predictions[i,0] = self.P_h * np.prod(h_prod)
            predictions[i,1] = self.P_s * np.prod(s_prod)

        return predictions


    def predict(self, X):
        '''
        Use self.predict_proba to assign labels to X,
        the label will be a string that is either 'spam' or 'ham'

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''
        # get predictions array ################################################
        predictions = self.predict_proba(X)
        argmax_array = np.argmax(predictions, axis=1)

        # translate results to ham or spam #####################################
        ham_or_spam = []
        for value in argmax_array:
            if value == 0:
                ham_or_spam.append('ham')
            else:
                ham_or_spam.append('spam')

        return np.array(ham_or_spam)


    def predict_log_proba(self, X):
        '''
        Find ln(P(C=k|x)) for each x in X and for each class k

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Probability each message is ham, spam
                0 column is ham
                1 column is spam
        '''
        # get predictions array ################################################
        predictions = np.zeros((X.shape[0],2))
        for i,message in enumerate(X.str.split()):
            h_sum = []
            s_sum = []
            counts = Counter(message)
            for word in counts:
                if word in self.h_words and word in self.s_words:
                    h_sum.append(np.log((self.data.loc['ham', word]/self.total_h_words) ** counts[word]))
                    s_sum.append(np.log((self.data.loc['spam',word]/self.total_s_words) ** counts[word]))
            predictions[i,0] = np.log(self.P_h) + np.sum(h_sum)
            predictions[i,1] = np.log(self.P_s) + np.sum(s_sum)

        return predictions


    def predict_log(self, X):
        '''
        Use self.predict_log_proba to assign labels to X,
        the label will be a string that is either 'spam' or 'ham'

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''
        # get predictions array ################################################
        predictions = self.predict_log_proba(X)
        argmax_array = np.argmax(predictions, axis=1)

        # translate results to ham or spam #####################################
        ham_or_spam = []
        for value in argmax_array:
            if value == 0:
                ham_or_spam.append('ham')
            else:
                ham_or_spam.append('spam')

        return np.array(ham_or_spam)


# Problem 6-7 ##################################################################
class PoissonBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages in to spam or ham.
    This classifier assumes that words are distributed like
    Poisson random variables
    '''
    def __init__(self):
        return


    def fit(self, X, y):
        '''
        Uses bayesian inference to find the poisson rate for each word
        found in the training set. For this we will use the formulation
        of l = rt since we have variable message lengths.

        This method creates a tool that will allow the filter to
        evaluate P(H), P(S), and P(w|C)


        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels

        Returns:
            self: this is an optional method to train
        '''
        # initialize data structures ###########################################
        NB = NaiveBayesFilter()
        NB.fit(X, y)
        self.data = NB.data
        self.P_h = NB.P_h
        self.P_s = NB.P_s
        self.h_words = NB.h_words
        self.s_words = NB.s_words
        self.N_h = NB.total_h_words
        self.N_s = NB.total_s_words

        # use maximizer and create dictionaries ################################
        self.ham_rates = dict()
        self.spam_rates= dict()

        # populate ham rates ###################################################
        for word, n_i in self.data.loc['ham'].items():
            self.ham_rates[word] = n_i / self.N_h

        # populate spam rates ##################################################
        for word, n_i in self.data.loc['spam'].items():
            self.spam_rates[word] = n_i / self.N_s


    def predict_proba(self, X):
        '''
        Find P(C=k|x) for each x in X and for each class

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Probability each message is ham or spam
                column 0 is ham, column 1 is spam
        '''
        # get predictions array ################################################
        predictions = np.zeros((X.shape[0],2))
        for i,message in enumerate(X.str.split()):
            h_sum = []
            s_sum = []
            n = len(message)
            counts = Counter(message)
            for word in counts:
                if (word in self.h_words) and (word in self.s_words):
                    h_sum.append(np.log(poisson.pmf(k=counts[word], mu=n*self.ham_rates[word])))
                    s_sum.append(np.log(poisson.pmf(k=counts[word], mu=n*self.spam_rates[word])))
            predictions[i,0] = np.log(self.P_h) + np.sum(h_sum)
            predictions[i,1] = np.log(self.P_s) + np.sum(s_sum)

        return predictions


    def predict(self, X):
        '''
        Use self.predict_proba to assign labels to X

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''
        # get predictions array ################################################
        predictions = self.predict_proba(X)
        argmax_array = np.argmax(predictions, axis=1)

        # translate results to ham or spam #####################################
        ham_or_spam = []
        for value in argmax_array:
            if value == 0:
                ham_or_spam.append('ham')
            else:
                ham_or_spam.append('spam')

        return np.array(ham_or_spam)


# Problem 8 ####################################################################
def sklearn_method(X_train, y_train, X_test):
    '''
    Use sklearn's methods to transform X_train and X_test, create a
    naïve Bayes filter, and classify the provided test set.

    Parameters:
        X_train (pandas.Series): messages to train on
        y_train (pandas.Series): labels for X_train
        X_test  (pandas.Series): messages to classify

    Returns:
        (ndarray): classification of X_test
    '''
    # initialize Vectorizer ####################################################
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X_train, y_train)

    # fit, transform, and test data ############################################ 
    clf = MultinomialNB().fit(X, y_train)
    test = vectorizer.transform(X_test)
    return clf.predict(test)
