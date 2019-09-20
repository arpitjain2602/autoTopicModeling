import pandas as pd
import numpy as np
import math
import time
import string
import nltk
import gensim
import warnings
import regex as re
import spacy

import gensim.corpora as corpora

from bisect import bisect_left 
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer, LancasterStemmer, PorterStemmer

extra_words = ["a","about","above","after","again","against","ain","all","am","an","and","any","are","aren","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can","couldn","couldn't","d","did","didn","didn't","do","does","doesn","doesn't","doing","don","don't","down","during","each","few","for","from","further","had","hadn","hadn't","has","hasn","hasn't","have","haven","haven't","having","he","her","here","hers","herself","him","himself","his","how","i","if","in","into","is","isn","isn't","it","it's","its","itself","just","ll","m","ma","me","mightn","mightn't","more","most","mustn","mustn't","my","myself","needn","needn't","no","nor","not","now","o","of","off","on","once","only","or","other","our","ours","ourselves","out","over","own","re","s","same","shan","shan't","she","she's","should","should've","shouldn","shouldn't","so","some","such","t","than","that","that'll","the","their","theirs","them","themselves","then","there","these","they","this","those","through","to","too","under","until","up","ve","very","was","wasn","wasn't","we","were","weren","weren't","what","when","where","which","while","who","whom","why","will","with","won","won't","wouldn","wouldn't","y","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","could","he'd","he'll","he's","here's","how's","i'd","i'll","i'm","i've","let's","ought","she'd","she'll","that's","there's","they'd","they'll","they're","they've","we'd","we'll","we're","we've","what's","when's","where's","who's","why's","would","able","abst","accordance","according","accordingly","across","act","actually","added","adj","affected","affecting","affects","afterwards","ah","almost","alone","along","already","also","although","always","among","amongst","announce","another","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apparently","approximately","arent","arise","around","aside","ask","asking","auth","available","away","awfully","b","back","became","become","becomes","becoming","beforehand","begin","beginning","beginnings","begins","behind","believe","beside","besides","beyond","biol","brief","briefly","c","ca","came","cannot","can't","cause","causes","certain","certainly","co","come","comes","contain","containing","contains","couldnt","date","different","done","downwards","due","e","ed","effect","eg","eight","eighty","either","else","elsewhere","end","ending","enough","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","except","f","far","ff","fifth","first","five","fix","followed","following","follows","former","formerly","forth","found","four","furthermore","g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","happens","hardly","hed","hence","hereafter","hereby","herein","heres","hereupon","hes","hi","hid","hither","home","howbeit","however","hundred","id","ie","im","immediate","immediately","importance","important","inc","indeed","index","information","instead","inward","itd","it'll","j","k","keep","keeps","kept","kg","km","know","known","knows","l","largely","last","lately","later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","'ll","look","looking","looks","ltd","made","mainly","make","makes","many","may","maybe","mean","means","meantime","meanwhile","merely","mg","might","million","ml","moreover","mostly","mr","mrs","much","must","n","namely","nay","nd","near","nearly","necessarily","necessary","need","needs","neither","never","nevertheless","nine","ninety","nobody","non","none","nonetheless","noone","normally","nos","noted","nothing","nowhere","obtain","obtained","obviously","often","oh","ok","okay","omitted","one","ones","onto","ord","others","otherwise","overall","owing","p","particular","particularly","past","per","perhaps","placed","please","possible","possibly","potentially","pp","predominantly","present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","rather","rd","readily","really","recent","recently","regarding","regardless","regards","related","relatively","respectively","resulted","resulting","right","sec","seem","seemed","seeming","seems","seen","self","selves","seven","several","shall","shed","shes","significant","significantly","similar","similarly","since","six","slightly","somebody","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","still","strongly","sub","substantially","successfully","sufficiently","suggest","sup","sure","take","taken","taking","tell","tends","th","thank","thanks","thanx","thats","that've","thence","thereafter","thereby","thered","therefore","therein","there'll","thereof","therere","theres","thereto","thereupon","there've","theyd","theyre","think","thou","though","thoughh","thousand","throug","throughout","thru","thus","til","together","took","toward","towards","tried","tries","truly","try","trying","ts","twice","two","u","un","unfortunately","unless","unlike","unlikely","unto","upon","us","use","useful","usefully","usefulness","uses","using","usually","v","various","'ve","via","viz","vol","vols","vs","w","want","wants","wasnt","way","wed","welcome","went","werent","whatever","what'll","whats","whence","whenever","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","whim","whither","whod","whoever","whole","who'll","whomever","whos","whose","widely","willing","wish","within","without","wont","world","wouldnt","x","yes","yet","youd","youre","z","zero","a's","ain't","allow","allows","apart","appear","appreciate","appropriate","associated","best","better","c'mon","c's","cant","changes","clearly","concerning","consequently","consider","considering","corresponding","course","currently","definitely","described","despite","entirely","exactly","example","going","greetings","hello","help","hopefully","ignored","inasmuch","indicate","indicated","indicates","inner","insofar","it'd","keep","keeps","novel","presumably","reasonably","second","secondly","sensible","seriously","sure","t's","third","thorough","thoroughly","three","well","wonder", "time", "secs","dear","good", "afternoon", "hello", "regard", "solve", "successfull"]
#Added the following words to this list
#Removed the following words from this list: com, value, 


# POS Spacy
pos_removal_spacy_list = ['ADV','PRON','CCONJ','PUNCT','PART','DET','ADP','SPACE']
# POS NLTK
pos_removal_nltk_list = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  'PRP','PRP$',  'RB','RBR','RBS','RP',  'JJ','JJR','JJS',   'CC','DT','EX','IN',   'WDT','WP','WP$','WRB']

nlp_spacy = spacy.load('en') # Spacy

warnings.filterwarnings('ignore')

stop_words_nltk = stopwords.words('english')

#adding the extra words to nltk stopwords
stop_words_nltk.extend(extra_words)

stop_words_nltk.sort()

#adding the extra words to spacy stopwords as well, in case only one of the two is used
for word in extra_words:
    nlp_spacy.vocab[word].is_stop = True

nltk_lemmatizer = WordNetLemmatizer()

#used to search in nltk stop_words
def BinarySearch(a, x): 
    i = bisect_left(a, x) 
    if i != len(a) and a[i] == x: 
        return i 
    else: 
        return -1
    
#nltk_stemmer = SnowballStemmer()
#nltk_stemmer = LancasterStemmer()
#nltk_stemmer = PorterStemmer()



def to_dictionary_and_corpus(data, fe_no_below=1, fe_no_above=0.05, fe_keep_n=50000, debug=False):
    #converts the data to a dictionary and corpus
    
    total_start = time.time()
    dictionary = corpora.Dictionary(data)
    dictionary_created = time.time()
    if(debug):
        print("Dictionary created in ",dictionary_created- total_start)
    
    dictionary.filter_extremes(no_below=fe_no_below, no_above=fe_no_above, keep_n=fe_keep_n)
    dictionary_filtered = time.time()
    if(debug):
        print("Dictionary filtered in ",dictionary_filtered-dictionary_created)
    corpus = [dictionary.doc2bow(doc) for doc in data]
    corpus_created = time.time()
    if(debug):
        print("Corpus created in ",corpus_created-dictionary_filtered)
    
    total_end = time.time()
    if(debug):
        print('Total Time Taken ', total_end-total_start)
    return dictionary, corpus



def preprocessing_docs_shashank(data, 
                make_lowercase = True, 
                punctuation_removal = True, 
                whitespace_removal = True, 
                store_alphanumeric = True, 
                tokenization_nltk = False, 
                lemmatization_tokenization_spacy = True, 
                stopwords_removal_nltk = True, stopwords_removal_spacy = True, 
                make_bigrams_gensim = True, bigrams_min_count = 5, bigrams_threshold = 15, 
                make_trigrams_gensim = True, trigrams_min_count = 5, trigrams_threshold = 15,  
                min_max_length_removal = True, mmlr_min_len = 3, mmlr_max_len = 50, mmlr_deacc = False, 
                pos_removal_nltk = True, 
                debug = False):
    #args: True False args are used to toggle the respective processing techniques
    #spell check should be added to this
    #each toggle includes the name of the library, if used
    

    
    total_start = time.time()
    sample_to_print  = 2
    
    
    if (make_lowercase is True):
        if(debug):
            print("data_sample out of ",len(data)) 
            print(data[:sample_to_print])
        start = time.time()
        data = [i.lower() for i in data]
        end = time.time()
        print('\n       ##### Lowercasing Done! Time Taken - ',end-start)
        #returns a list of sentences
    
    if (punctuation_removal is True):
        if(debug):
            print("data_sample out of ",len(data)) 
            print(data[:sample_to_print])
        start = time.time()
        data = [i.translate(str.maketrans(string.punctuation,' '*len(string.punctuation))) for i in data]
        #returns a list of sentences
        end = time.time()
        print('\n       ##### Punctuation removed! Time Taken - ',end-start)
        #returns a list of sentences
    
    if (whitespace_removal is True):
        if(debug):
            print("data_sample out of ",len(data)) 
            print(data[:sample_to_print])
        start = time.time()
        data = [' '.join(mystring.split()) for mystring in data]
#         data = [i.strip() for i in data]
        end = time.time()
        print('\n       ##### Whitespace removed! Time Taken - ',end-start)
    
    
    if (pos_removal_nltk == True):
        if(debug):
            print("data_sample out of ",len(data)) 
            print(data[:sample_to_print])
        # NLTK
        start = time.time()
#         data = [nltk.pos_tag(doc.split()) for doc in data]
#         data = [" ".join([token[0] for token in doc if not token[1] in pos_removal_nltk_list]) for doc in data]
        new_data = list()
        #TO KEEP TRACK OF PROGRESS
        count = 0
        for doc in data:
            if(count%2000 == 0):
                if(debug):
                    print(count*100.0/len(data),"% Done")
            count += 1
            new_data.append(" ".join([token[0] for token in nltk.pos_tag(doc.split()) if not token[1] in pos_removal_nltk_list]))
        data = new_data
        end = time.time()
        print('\n       ##### POS Removal Done! Time Taken - ',end-start)
        print(' ')

    
    alpha_numeric_word_list = list()
    if (store_alphanumeric is True):
        if(debug):
            print("data_sample out of ",len(data)) 
            print(data[:sample_to_print])
        start = time.time()
        rx = re.compile(r'\S*\d+\S*')
        for sentence in data:
            alpha_numeric_word_list.append(rx.findall(sentence))
        alpha_numeric_word_list = [[x for x in alpha_numeric_word_item if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())] for alpha_numeric_word_item in alpha_numeric_word_list]
        end = time.time()
        print('\n       ##### Alphanumeric Words Stored! Time Taken - ',end-start)
        print(' ')
        
    if (tokenization_nltk == True):
        if(debug):
            print("data_sample out of ",len(data)) 
            print(data[:sample_to_print])
        # Using NLTK
        start = time.time()
        data = [nltk.word_tokenize(i) for i in data]
        end = time.time()
        # Using Spacy - Spacy takes too much time
        #data = [[token.text for token in nlp_spacy(i)] for i in data]
        print('\n       ##### Tokenization Done using NLTK! Time Taken - ', end-start)
        print(' ')

    if (lemmatization_tokenization_spacy == True):
        if(debug):
            print("data_sample out of ",len(data)) 
            print(data[:sample_to_print])
        # Spacy Lemmatizer
        start = time.time()
        new_data  = list()
#         data = [[i.lemma_ for i in nlp_spacy(doc)] for doc in data]
        #TO KEEP A TRACK OF Progress
        j = 0
        for doc in data:
            if(j%2000 == 0):
                if(debug):
                    print(j*100.0/len(data), "% done")
            j+=1
            new_data.append([i.lemma_ for i in nlp_spacy(doc)])
        data = new_data
        # NLTK Lemmatizer
        #data = [[nltk_lemmatizer.lemmatize(j) for j in doc] for doc in data]
        end = time.time()
        print('\n       ##### Lemmatization and Tokenization Done using Spacy! Time Taken - ',end-start)
        print(' ')

    if (stopwords_removal_nltk == True):
        if(debug):
            print("stopwords_removal_nltk data_sample out of ",len(data))
            print(data[:sample_to_print])
        #using NLTK
        start = time.time()
        data = [[j for j in doc if (BinarySearch(stop_words_nltk,j)<0)] for doc in data]
        data = [[x for x in word if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())] for word in data]
        end = time.time()
        print('\n       ##### Stopwords Removed using NLTK! Time Taken - ',end-start)
        print(' ')
        
    if (stopwords_removal_spacy == True):
        if(debug):
            print("stopwords_removal_spacy data_sample out of ",len(data)) 
            print(data[:sample_to_print])
        #using SPACY
        counter = 0
        start = time.time()
        new_data = list()
        for doc in data:
            if(counter%1000 == 0):
                if(debug):
                    print(counter*100.0/len(data), "% done")
            counter += 1
            new_data.append([string(j) for word in doc for j in nlp_spacy(word) if not j.is_stop ])
            if(counter<2):
                print(new_data)
        data = new_data#[[j for word in doc for j in nlp_spacy(word) if not j.is_stop ] for doc in data]
        end = time.time()
        print('\n       ##### Stopwords Removed using Spacy! Time Taken - ',end-start)
        print(' ')
    
    if (make_bigrams_gensim == True):  
        if(debug):
            print("data_sample out of ",len(data)) 
            print(data[:sample_to_print]) 
        # Gensim
        start = time.time()
        bigram = gensim.models.phrases.Phrases(data, min_count=bigrams_min_count, threshold=bigrams_threshold)
        data = [bigram[doc] for doc in data]
        end = time.time()
        print('\n       ##### Bi-Grams made using Gensim! Time Taken - ',end-start)
        print(' ')
    
    if (make_trigrams_gensim == True):  
        if(debug):
            print("data_sample out of ",len(data)) 
            print(data[:sample_to_print]) 
        # Gensim
        start = time.time()
        trigram = gensim.models.phrases.Phrases(data, min_count=trigrams_min_count, threshold=trigrams_threshold)
        data = [trigram[doc] for doc in data]
        end = time.time()
        print('\n       ##### Tri-Grams made using Gensim! Time Taken - ',end-start)
        print(' ')
    
    if (min_max_length_removal == True):
        if(debug):
            print("data_sample out of ",len(data)) 
            print(data[:sample_to_print])
        # Gensim
        start = time.time()
        data = [simple_preprocess(' '.join(doc), min_len=mmlr_min_len, max_len=mmlr_max_len, deacc=mmlr_deacc) for doc in data] 
        end = time.time()
        print('\n       ##### Min_Max Word Length Removal Done using Gensim! Time Taken - ',end-start)
        print(' ')
    
        
    if (store_alphanumeric is True):
        if(debug):
            print("data_sample out of ",len(data)) 
            print(data[:sample_to_print])
            print("alphanumeric_sample out of ",len(alphanumeric)) 
            print(alphanumeric[:4])
            
        start = time.time()
        print(len(data))
        print(len(alpha_numeric_word_list))
        data = [data[i] + alpha_numeric_word_list[i] for i in range(len(data))]
        end = time.time()
        print('\n       ##### Alphanumeric Words added back to the text! Time Taken - ',end-start)
    
    total_end = time.time()
    print('       Total Time Taken ', total_end-total_start)
    print(' ')
    
    return data
    



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