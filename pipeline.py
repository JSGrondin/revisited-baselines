# Importing relevant packages
from data_load import load_imdb
from data_load import report
from data_load import lower_text
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from textprocessing import remove_urls
from textprocessing import remove_emails
from sklearn.model_selection import train_test_split
from textprocessing import lemmatization_stem_and_stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from nbsvm import NBSVM

# Extracting raw features (X) and targets(y)
X_train, y_train = load_imdb('train')
X_test, y_test= load_imdb('test')

###########################################################
#                  Text Pre-processing                    #
###########################################################

# Removing URLs
X_train = remove_urls(X_train)
X_text = remove_urls(X_test)

# Removing email addresses
# X_train = remove_emails(X_train)
# X_test = remove_emails(X_text)

# # Removing special characters
# X_train = char_preprocessing(X_train)
# X_test = char_preprocessing(X_test)

# with open('your_file.txt', 'w', encoding="utf-8") as f:
#     for item in X_train:
#         f.write("%s\n" % item)

# Processing text features on Xtrain, X_val and X_test
X_train = lemmatization_stem_and_stopwords(X_train,True,False,False)
X_test = lemmatization_stem_and_stopwords(X_test,True,False,False)

# Removing capital letters, i.e lowering all strings
X_train = lower_text(X_train)
X_test = lower_text(X_test)
#
# seed=123
# # Splitting the training set into a training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
#                                                  train_size=0.90, \
#                                                  test_size=0.1,
#                                                  random_state=seed)
#

###########################################################
#                           NBSVM                         #
###########################################################
#token_pattern = r'[a-zA-Z]+'
#token_pattern = r'\w+'
# token_pattern = r'[a-zA-Z]+|[\d+\/\d+]+'
#token_pattern = r'\w+|[%s]' % string.punctuation
# token_pattern = r'[a-zA-Z]+|[\d+\/\d+]+|[%s]' % string.punctuation
token_pattern = r'[\d+\/\d+]+|\w+|[%s]' % string.punctuation


pclf_NBSVM = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1,3),
                             token_pattern=token_pattern,
                             binary=True)),
    ('clf', NBSVM(beta=0.3, C=1, alpha=1.0, fit_intercept=False))
])

pclf_NBSVM.fit(X_train, y_train)
print('Test Accuracy: %s' % pclf_NBSVM.score(X_test, y_test))


params = {"vect__ngram_range": [(1,2)],
          "vect__binary": [True],
          "clf__alpha": [1],
          "clf__C": [1],
          "clf__beta": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
          "clf__fit_intercept": [False]
          }

# Perform randomized search CV to find best hyperparameters
random_search = RandomizedSearchCV(pclf_NBSVM, param_distributions = params,
                                   cv=3,
                                   verbose = 30, random_state = 123,
                                   n_iter = 10)
random_search.fit(X_train, y_train)

# Report results
report(random_search.cv_results_)
