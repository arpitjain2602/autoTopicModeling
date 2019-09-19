import pandas as pd
import numpy as np
import math
import time
import string
import nltk
import gensim
import warnings
warnings.filterwarnings('ignore')

import spacy
nlp_spacy = spacy.load('en') # Spacy

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
extra_words = ["a","about","above","after","again","against","ain","all","am","an","and","any","are","aren","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can","couldn","couldn't","d","did","didn","didn't","do","does","doesn","doesn't","doing","don","don't","down","during","each","few","for","from","further","had","hadn","hadn't","has","hasn","hasn't","have","haven","haven't","having","he","her","here","hers","herself","him","himself","his","how","i","if","in","into","is","isn","isn't","it","it's","its","itself","just","ll","m","ma","me","mightn","mightn't","more","most","mustn","mustn't","my","myself","needn","needn't","no","nor","not","now","o","of","off","on","once","only","or","other","our","ours","ourselves","out","over","own","re","s","same","shan","shan't","she","she's","should","should've","shouldn","shouldn't","so","some","such","t","than","that","that'll","the","their","theirs","them","themselves","then","there","these","they","this","those","through","to","too","under","until","up","ve","very","was","wasn","wasn't","we","were","weren","weren't","what","when","where","which","while","who","whom","why","will","with","won","won't","wouldn","wouldn't","y","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","could","he'd","he'll","he's","here's","how's","i'd","i'll","i'm","i've","let's","ought","she'd","she'll","that's","there's","they'd","they'll","they're","they've","we'd","we'll","we're","we've","what's","when's","where's","who's","why's","would","able","abst","accordance","according","accordingly","across","act","actually","added","adj","affected","affecting","affects","afterwards","ah","almost","alone","along","already","also","although","always","among","amongst","announce","another","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apparently","approximately","arent","arise","around","aside","ask","asking","auth","available","away","awfully","b","back","became","become","becomes","becoming","beforehand","begin","beginning","beginnings","begins","behind","believe","beside","besides","beyond","biol","brief","briefly","c","ca","came","cannot","can't","cause","causes","certain","certainly","co","com","come","comes","contain","containing","contains","couldnt","date","different","done","downwards","due","e","ed","edu","effect","eg","eight","eighty","either","else","elsewhere","end","ending","enough","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","except","f","far","ff","fifth","first","five","fix","followed","following","follows","former","formerly","forth","found","four","furthermore","g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","happens","hardly","hed","hence","hereafter","hereby","herein","heres","hereupon","hes","hi","hid","hither","home","howbeit","however","hundred","id","ie","im","immediate","immediately","importance","important","inc","indeed","index","information","instead","invention","inward","itd","it'll","j","k","keep","keeps","kept","kg","km","know","known","knows","l","largely","last","lately","later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","'ll","look","looking","looks","ltd","made","mainly","make","makes","many","may","maybe","mean","means","meantime","meanwhile","merely","mg","might","million","miss","ml","moreover","mostly","mr","mrs","much","mug","must","n","na","name","namely","nay","nd","near","nearly","necessarily","necessary","need","needs","neither","never","nevertheless","new","next","nine","ninety","nobody","non","none","nonetheless","noone","normally","nos","noted","nothing","nowhere","obtain","obtained","obviously","often","oh","ok","okay","old","omitted","one","ones","onto","ord","others","otherwise","outside","overall","owing","p","page","pages","part","particular","particularly","past","per","perhaps","placed","please","plus","poorly","possible","possibly","potentially","pp","predominantly","present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather","rd","readily","really","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted","resulting","results","right","run","said","saw","say","saying","says","sec","section","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sent","seven","several","shall","shed","shes","show","showed","shown","showns","shows","significant","significantly","similar","similarly","since","six","slightly","somebody","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","still","stop","strongly","sub","substantially","successfully","sufficiently","suggest","sup","sure","take","taken","taking","tell","tends","th","thank","thanks","thanx","thats","that've","thence","thereafter","thereby","thered","therefore","therein","there'll","thereof","therere","theres","thereto","thereupon","there've","theyd","theyre","think","thou","though","thoughh","thousand","throug","throughout","thru","thus","til","tip","together","took","toward","towards","tried","tries","truly","try","trying","ts","twice","two","u","un","unfortunately","unless","unlike","unlikely","unto","upon","ups","us","use","used","useful","usefully","usefulness","uses","using","usually","v","value","various","'ve","via","viz","vol","vols","vs","w","want","wants","wasnt","way","wed","welcome","went","werent","whatever","what'll","whats","whence","whenever","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","whim","whither","whod","whoever","whole","who'll","whomever","whos","whose","widely","willing","wish","within","without","wont","words","world","wouldnt","www","x","yes","yet","youd","youre","z","zero","a's","ain't","allow","allows","apart","appear","appreciate","appropriate","associated","best","better","c'mon","c's","cant","changes","clearly","concerning","consequently","consider","considering","corresponding","course","currently","definitely","described","despite","entirely","exactly","example","going","greetings","hello","help","hopefully","ignored","inasmuch","indicate","indicated","indicates","inner","insofar","it'd","keep","keeps","novel","presumably","reasonably","second","secondly","sensible","serious","seriously","sure","t's","third","thorough","thoroughly","three","well","wonder"," ", "user", "folder", "approval", "share", "time", "secs", "stryker", "manager", "dear", "team", "access", "good", "afternoon", "hello", "regard", "solve", "successfull"]
stop_words.extend(extra_words)

from nltk.stem import WordNetLemmatizer, SnowballStemmer, LancasterStemmer, PorterStemmer
nltk_lemmatizer = WordNetLemmatizer()
#nltk_stemmer = SnowballStemmer()
#nltk_stemmer = LancasterStemmer()
#nltk_stemmer = PorterStemmer()

# Min Max Removal
from gensim.utils import simple_preprocess

# POS Spacy
pos_removal_spacy = ['ADV','PRON','CCONJ','PUNCT','PART','DET','ADP','SPACE']
# POS NLTK
pos_removal_nltk = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  'PRP','PRP$',  'RB','RBR','RBS','RP',  'JJ','JJR','JJS',   'CC','DT','EX','IN',   'WDT','WP','WP$','WRB']

import gensim.corpora as corpora


def preprocessing_docs_arpit(data, make_lowercase = True,	punctuation_removal = True,	whitespace_removal = True, tokenization=False, lemmatization_tokenization = True, stopwords_removal = True, make_ngrams=True, min_max_removal=True, pos_removal=True):
	total_start = time.time()
	'''
	
	'''
	if (make_lowercase==True):
		start = time.time()
		data = [i.lower() for i in data]
		end = time.time()
		print('Lowercasing Done! Time Taken - ',end-start)
	
	if (punctuation_removal==True):
		start = time.time()
		data = [i.translate(str.maketrans('', '', string.punctuation)) for i in data]
		end = time.time()
		print('Punctuation removed! Time Taken - ',end-start)
	
	if (whitespace_removal==True):
		start = time.time()
		data = [i.strip() for i in data]
		end = time.time()
		print('Whitespace removed! Time Taken - ',end-start)

	if (tokenization == True):
		# Using NLTK
		start = time.time()
		data = [nltk.word_tokenize(i) for i in data]
		end = time.time()
		# Using Spacy - Spacy takes too much time
		#data = [[token.text for token in nlp_spacy(i)] for i in data]
		print('Tokenization Done! Time Taken - ', end-start)

	if (lemmatization_tokenization == True):
		# Spacy Lemmatizer
		start = time.time()
		data = [[i.lemma_ for i in nlp_spacy(doc)] for doc in data]
		# NLTK Lemmatizer
		#data = [[nltk_lemmatizer.lemmatize(j) for j in doc] for doc in data]
		end = time.time()
		print('Lemmatization and Tokenization Done! Time Taken - ',end-start)

	if (stopwords_removal == True):
		start = time.time()
		data = [[j for j in doc if not j in stop_words] for doc in data]
		end = time.time()
		print('Stopwords Removed! Time Taken - ',end-start)
	
	if (make_ngrams == True):	
		# Gensim
		start = time.time()
		bigram = gensim.models.phrases.Phrases(data, min_count=5, threshold=15)
		data = [bigram[doc] for doc in data]
		end = time.time()
		print('Bi-Gram Done! Time Taken - ',end-start)
	
	if (min_max_removal == True):
		# Gensim
		start = time.time()
		data = [simple_preprocess(' '.join(doc), min_len=3, max_len=50, deacc=False) for doc in data] 
		end = time.time()
		print('Min_Max Removal Done! Time Taken - ',end-start)
	
	if (pos_removal == True):
		# NLTK
		start = time.time()
		data = [nltk.pos_tag(doc) for doc in data]
		data = [[token[0] for token in doc if not token[1] in pos_removal_nltk] for doc in data]
		end = time.time()
		print('POS Removal Done! Time Taken - ',end-start)
	
	dictionary = corpora.Dictionary(data)
	dictionary.filter_extremes(no_below=1, no_above=0.05, keep_n=50000)
	corpus = [dictionary.doc2bow(doc) for doc in data]
	
	total_end = time.time()
	print('Total Time Taken ', total_end-total_start)
	return dictionary, corpus, data




'''
elif (stemming == True):
	data = [[nltk_stemmer.lemmatize(j) for j in doc] for doc in data]
	print('Stemming Done...')

elif (tokenize == True):
	# Using NLTK
	data = [nltk.word_tokenize(i) for i in data]
	# Using Spacy - Spacy takes too much time
	#data = [[token.text for token in nlp_spacy(i)] for i in data]
	print('Tokenization Done...')
'''


'''
CC coordinating conjunction
CD cardinal digit
DT determiner
EX existential there (like: “there is” … think of it like “there exists”)
FW foreign word
IN preposition/subordinating conjunction
JJ adjective ‘big’
JJR adjective, comparative ‘bigger’
JJS adjective, superlative ‘biggest’
LS list marker 1)
MD modal could, will
NN noun, singular ‘desk’
NNS noun plural ‘desks’
NNP proper noun, singular ‘Harrison’
NNPS proper noun, plural ‘Americans’
PDT predeterminer ‘all the kids’
POS possessive ending parent’s
PRP personal pronoun I, he, she
PRP$ possessive pronoun my, his, hers
RB adverb very, silently,
RBR adverb, comparative better
RBS adverb, superlative best
RP particle give up
TO, to go ‘to’ the store.
UH interjection, errrrrrrrm
VB verb, base form take
VBD verb, past tense took
VBG verb, gerund/present participle taking
VBN verb, past participle taken
VBP verb, sing. present, non-3d take
VBZ verb, 3rd person sing. present takes
WDT wh-determiner which
WP wh-pronoun who, what
WP$ possessive wh-pronoun whose
WRB wh-abverb where, when
'''