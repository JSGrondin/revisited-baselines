# revisited-baselines
Improved baselines for sentence and document representations

This mini-project was undertaken as part of COMP-551 at McGill University. 

The goal of this project was to revisit statements made in the work of Le & al with regard to the performance of Paragraph vectors in natural language processing applications. The authos claimed that Paragraph vectors achieved state-of-the-art results on text classification and sentiment analysis tasks. To verify this statement, the best baselines referenced in this report were reproduce. All comparisons were made on the IMDB sentiment dataset. A NB-SVM baseline was used and improved. The latter achieved an accuracy of 92.096% on the test set. This is 0.876% above the baseline reported in the original article. 

The following scripts were used: 

data_load.py : to load review comments

textprocessing.py : to remove special characters, stop words, lemmatize or stem words, etc

pipeline.py : main file used to generate predictions

See the writeup.pdf for details on the methodology and results. 
