from __future__ import print_function
import os

def openfile(path):
    results = [] 
    files = os.listdir(path)
    for name in files:
        results.append(name)
    results.sort(reverse = True)

    return(path+'/'+results[0])
