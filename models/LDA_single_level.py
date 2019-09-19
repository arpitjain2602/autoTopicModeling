from preprocess.preprocess import preprocess
from utils.utils import get_topics, evaluate_model
import pickle
import time
import numpy as np
import operator
import pandas as pd
import gensim
from gensim.models import CoherenceModel
import time
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import pickle


def lda_model_single_level(
				raw_docs,
				num_topics_list_level_1,
				coherence='c_v', 
				debug = False,
				need_best_topic=True, 
				model_selection_metric='coherence'):
	'''
	- runs LDA at 1 level
	'''
	dictionary, corpus, doc_list = preprocess(raw_docs, 'shashank', False)

	lda_models_level_1 = {}
	coherence_list = []
	perplexity_list = []
	
	for topic in num_topics_list_level_1:
		
		print('### Running LDA for {} topic'.format(topic))
		start = time.time()
		
		lda_models_level_1[topic] = gensim.models.ldamodel.LdaModel(corpus=corpus,
														id2word=dictionary,
														num_topics=topic,
														random_state=100,
														update_every=1,
														chunksize=100,
														passes=10,
														alpha='auto',
														per_word_topics=True)

		print('LDA Done for {} topic! Time Taken is {}'.format(topic, time.time()-start))

		coherence, perplexity = evaluate_model(lda_models_level_1[topic], dictionary, corpus, doc_list, coherence = coherence)
		coherence_list.append(coherence)
		perplexity_list.append(perplexity)
	
	coherence_list = np.array(coherence_list)
	perplexity_list = np.array(perplexity_list)
	
	if (need_best_topic == True):
		coherence_index = np.argmax(coherence_list)
		perplexity_index = np.argmin(perplexity_list)

		if (model_selection_metric == 'coherence'):
			best_topic_index = coherence_index
		elif (model_selection_metric == 'perplexity'):
			best_topic_index = perplexity_index

	best_topic = num_topics_list_level_1[best_topic_index]
	coherence_score = coherence_list[best_topic_index]
	perplexity_score = perplexity_list[best_topic_index]
	best_lda_model = lda_models_level_1[best_topic]

	lda_level_1 = {'best_lda_model': best_lda_model,
							'best_topic': best_topic,
							'coherence_score': coherence_score,
							'perplexity_score': perplexity_score,
							'corpus': corpus,
							'dictionary': dictionary,
							'doc_list': doc_list}

	print('Done Single-Level LDA')
	return lda_level_1