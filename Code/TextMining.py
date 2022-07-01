import re, os, itertools, docx2txt
from tqdm import tqdm
#from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize
from spacy.lang.id import Indonesian
from html import unescape
from unidecode import unidecode
import pandas as pd
#from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
#from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
#from nltk.tokenize import TweetTokenizer; Tokenizer = TweetTokenizer(reduce_len=True)
#from nltk.corpus import wordnet as wn
#from nltk.stem import PorterStemmer;ps = PorterStemmer()
from string import punctuation
#from pattern.web import PDF
from bz2 import BZ2File as bz2
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.decomposition import LatentDirichletAllocation as LDA
import numpy as np
from textblob import TextBlob 
import openewfile as of

def crawlFiles(dPath, types = None):
    #dPath = 'C:/Temp', types ='pdf'
    if types:
        return [dPath+'/'+f for f in os.listdir(dPath) if f.endswith('.'+types)]
    else:
        return [dPath+'/'+f for f in os.listdir(dPath)]

def readBz2(file):
    with bz2(file, "r") as bzData:
        txt = []
        for line in bzData:
            try:
                txt.append(line.strip().decode('utf-8','replace'))
            except:
                pass
    return ' '.join(txt)

def LoadDocuments(dPath=None,types=None, file = None): # types = ['pdf','doc','docx','txt','bz2']
    Files, Docs = [], []
    if types:
        for tipe in types:
            Files += crawlFiles(dPath,tipe)
    if file:
        Files = [file]
    if not types and not file: # get all files regardless of their extensions
        Files += crawlFiles(dPath)
    for f in Files:
#        if f[-3:].lower()=='pdf':
#            try:
#                Docs.append(PDF(f).string)
#            except:
#                print('error reading{0}'.format(f))
        if f[-3:].lower()=='txt':
            try:
                df=open(f,"r",encoding="utf-8", errors='replace')
                Docs.append(df.readlines());df.close()
            except:
                print('error reading{0}'.format(f))
        elif f[-3:].lower()=='bz2':
            try:
                Docs.append(readBz2(f))
            except:
                print('error reading{0}'.format(f))
        elif f[-4:].lower()=='docx':
            try:
                Docs.append(docx2txt.process(f))
            except:
                print('error reading{0}'.format(f))
        elif f[-3:].lower()=='csv':
            Docs.append(pd.read_csv(f))
        else:
            print('Unsupported format {0}'.format(f))
    if file:
        Docs = Docs[0]
    return Docs, Files


def LoadStopWords(lang, sentiment = True):
    L = lang.lower().strip()
    if sentiment:
        if L == 'en' or L == 'english' or L == 'inggris':
            lemmatizer = WordNetLemmatizer()
            stops = set([t.strip() for t in LoadDocuments(file = of.openfile(path = 'Stopwords'))[0]])
        elif L == 'id' or L == 'indonesia' or L == 'indonesian':
            lemmatizer = Indonesian()
            stops = set([t.strip() for t in LoadDocuments(file = of.openfile(path = 'Stopwords'))[0]])
        else:
            print('Warning! Languange not recognized. Empty stopword given')
            stops = set(); lemmatizer = None
    else:
        if L == 'en' or L == 'english' or L == 'inggris':
            lemmatizer = WordNetLemmatizer()
            stops = set([t.strip() for t in LoadDocuments(file = of.openfile(path = 'Stopwords_eng'))[0]])
        elif L == 'id' or L == 'indonesia' or L == 'indonesian':
            lemmatizer = Indonesian()
            stops = set([t.strip() for t in LoadDocuments(file = of.openfile(path = 'Stopwords'))[0]])
        else:
            print('Warning! Languange not recognized. Empty stopword given')
            stops = set(); lemmatizer = None
    return stops, lemmatizer

def fixTags(T):
    getHashtags = re.compile(r"#(\w+)")
    pisahtags = re.compile(r'[A-Z][^A-Z]*')
    t = T
    tagS = re.findall(getHashtags, T)
    for tag in tagS:
        proper_words = ' '.join(re.findall(pisahtags, tagS[0]))
        t = t.replace('#'+tag, proper_words)
    return t

def getTags(T):
    getHashtags = re.compile(r"#(\w+)")
    tagS = re.findall(getHashtags, T)
    isitag = []
    for tag in tagS:       
        tag = '#'+tag
        isitag.append(tag)
    
    return ', '.join(isitag)

def cleanText(T, fix={},pattern2 = False, lang = 'id', lemma=None, stops = set(), symbols_remove = False, numbers_remove = False, hashtag_remove = False, min_charLen = 0):
    # lang & stopS only 2 options : 'en' atau 'id'
    # symbols ASCII atau alnum
    pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    if pattern2:
        pattern2 = re.compile(r'@(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+') #remove@
        t = re.sub(pattern2, ' ',T)
    else:
        t = T
    t = re.sub(pattern,' ',t) #remove urls if any
    t = unescape(t) # html entities fix
    if hashtag_remove:
        t = getTags(t) # fix abcDef
    else:
        t = t
    t = t.lower().strip() # lowercase
    t = unidecode(t)
    t = ''.join(''.join(s)[:2] for _, s in itertools.groupby(t)) # remove repetition
    t = sent_tokenize(t) # sentence segmentation. String to list
    for i, K in enumerate(t):
        if symbols_remove:
            K = re.sub(r'[^.,a-zA-Z0-9 \n\.]',' ',K)
            K = K.replace(',',' ').replace('.',' ')
            K = ''.join(c for c in K if c not in punctuation)
            K = re.sub('\s+',' ',K).strip()
        
        if numbers_remove:
            K = re.sub(r'[0-9]',' ',K)
            K = re.sub('\s+',' ',K)
            
        cleanList = []
        if lang =='en':
            listKata = word_tokenize(K) # word tokenize
            for token in listKata:
                if token in fix.keys():
                    token = fix[token]
                if lemma:
                    token = lemma.lemmatize(token)
                if stops:
                    if len(token)>=min_charLen and token not in stops:
                        cleanList.append(token)
                else:
                    if len(token)>=min_charLen:
                        cleanList.append(token)
            t[i] = ' '.join(cleanList)
        else:
            if lemma:
                K = lemma(K)
                listKata = [token.text for token in K]
            else:
                listKata = TextBlob(K).words
                
            for token in listKata:
                if token in fix.keys():
                    token = fix[token]
                
                if lemma:
                    token = lemma(token)[0].lemma_
                    #token = stemmer.stem(token)
                if stops:    
                    if len(token)>=min_charLen and token not in stops:
                        cleanList.append(token)
                else:
                    if len(token)>=min_charLen:
                        cleanList.append(token)
            t[i] = ' '.join(cleanList).lstrip()
    return ' '.join(t) # Return kalimat lagi

def cleanText_(T, fix={}, min_charLen = 0):
    t = T
    t = t.lower().strip() # lowercase
    t = unidecode(t)
    t = ''.join(''.join(s)[:2] for _, s in itertools.groupby(t)) # remove repetition
    t = sent_tokenize(t) # sentence segmentation. String to list
    for i, K in enumerate(t):
        cleanList = []
        listKata = TextBlob(K).words
        for token in listKata:
            if token in fix.keys():
                token = fix[token]
                cleanList.append(token)
            else:
                if len(token)>=min_charLen:
                    cleanList.append(token)
        t[i] = ' '.join(cleanList).lstrip()
    return ' '.join(t) # Return kalimat lagi


def handlingnegation (text):
    match = re.compile(r'(tidak|kurang|bukan|jangan|tapi|tetapi) (\w+)').findall(text)
    match = list(set(match))
    for i,word in enumerate(match): 
        if ' '.join(match[i]) in text:
            kata = text.replace(' '.join(match[i]), str(match[i][0])+' '+'negx'+str(match[i][1]))
            text = kata
    return text

def handlingporn (text):
    match = re.compile(r'(video|foto|photo) (\w+)').findall(text)
    match = list(set(match))
    for i,word in enumerate(match): 
        if ' '.join(match[i]) in text:
            kata = text.replace(' '.join(match[i]), str(match[i][0])+' '+'pornx'+str(match[i][1]))
            text = kata
    return text