import TextMining as tm
import time
import openewfile as of
from tqdm import tqdm
#import swifter


def cleanningtext(data, both = True, onlyclean = False, sentiment = False):
    print('Cleaning Text')
    fSlang = of.openfile(path = '../Slangwords')
    bahasa = 'id'
    stops, lemmatizer = tm.LoadStopWords(bahasa, sentiment = sentiment)
    sw=open(fSlang,encoding='utf-8', errors ='ignore', mode='r');SlangS=sw.readlines();sw.close()
    SlangS = {slang.strip().split(':')[0]:slang.strip().split(':')[1] for slang in SlangS}
  
    start_time = time.time()
    tqdm.pandas()
    
    if both:
        data['text'] = data['text'].astype('str')
        data['text'] = data['text'].str.lower()
        data = data[~data.text.str.contains('unavailable')]
        data['cleaned_text'] = data['text'].progress_apply(lambda x : tm.cleanText(x,fix=SlangS, pattern2 = True, lang = bahasa, lemma=lemmatizer, stops = stops, 
                              symbols_remove = True, numbers_remove = True, hashtag_remove=False, min_charLen = 2))
        data['cleaned_text'] = data['text'].progress_apply(lambda x : tm.handlingnegation(x))
        #data['cleaned_text'] = data['cleaned_text'].progress_apply(lambda x : tm.handlingporn(x))
    elif onlyclean: 
        data['cleaned_text'] = data['text'].progress_apply(lambda x : tm.cleanText(x, fix=SlangS, pattern2 = True, lang = bahasa, 
                               lemma=lemmatizer, stops = stops, symbols_remove = True, numbers_remove = True, hashtag_remove=False, min_charLen = 3))
    else:
        data['cleaned_text'] = data['text'].progress_apply(lambda x : tm.handlingnegation(x))
    
    data = data[data['cleaned_text'].notna()]
    print("%s seconds" %(time.time()-start_time))
    
    return data