from preprocess.preprocess_arpit import *
from preprocess.preprocess_shashank import *
import time
import pickle

def preprocess(raw_docs, preprocessing_type, debug):
	'''
	- preprocessing_type: arpit or shashank - can add more by creating relevant python files
	'''

	print('### Starting pre-processing')
	start = time.time()
	if (preprocessing_type=='arpit'):
		dictionary, corpus, doc_list = preprocessing_docs_arpit(raw_docs, make_lowercase = True,
																punctuation_removal = True,    whitespace_removal = True, 
																lemmatization_tokenization = True, stopwords_removal = True, 
																make_ngrams=True, min_max_removal=True, pos_removal=True)

	elif (preprocessing_type=='shashank'):
		doc_list = preprocessing_docs(raw_docs, 
									make_lowercase = True, punctuation_removal = True, 
									whitespace_removal = True, store_alphanumeric = False, pos_removal_nltk = True,
									tokenization_nltk = False, lemmatization_tokenization_spacy = True, 
									stopwords_removal_nltk = True, stopwords_removal_spacy = False, 
									make_bigrams_gensim = False, bigrams_min_count = 10, bigrams_threshold = 10, 
									make_trigrams_gensim = False, trigrams_min_count = 10, trigrams_threshold = 10, 
									min_max_length_removal = False, mmlr_min_len = 3, mmlr_max_len = 50, mmlr_deacc = False,
									debug = False)
		doc_list = preprocessing_docs(doc_list, 
									make_lowercase = False, punctuation_removal = False, 
									whitespace_removal = False, store_alphanumeric = False, pos_removal_nltk = False, 
									tokenization_nltk = False, lemmatization_tokenization_spacy = False, 
									stopwords_removal_nltk = False, stopwords_removal_spacy = False, 
									make_bigrams_gensim = True, bigrams_min_count = 10, bigrams_threshold = 10, 
									make_trigrams_gensim = True, trigrams_min_count = 10, trigrams_threshold = 10, 
									min_max_length_removal = False, mmlr_min_len = 3, mmlr_max_len = 50, mmlr_deacc = False, 
									debug = True)
	
	print('~~~		pre-processing done in ', time.time()-start)


	print('### Creating dictionary and corpus')
	dictionary, corpus = to_dictionary_and_corpus(doc_list, 
													fe_no_below=1, 
													fe_no_above=0.15, 
													fe_keep_n=None)
	return dictionary, corpus, doc_list