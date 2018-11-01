#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 19:08:32 2018

@author: pabloruizruiz
"""


database = ['a', 'abracadara', 'al', 'alice', 'alicia', 'allen', 'alter', 
            'altercation', 'az', 'azz', 'azzz', 'azzzz', 'azzzzz', 'azzzzzz', 
            'azzzzzzz', 'azzzzzzzz', 'azzzzzzzzz', 'bob', 'element', 'ello', 
            'eve', 'evening', 'event', 'eventually', 'mallory', 'z', 'zz', 'zzz', 
            'zzzz', 'zzzzz', 'zzzzzz', 'zzzzzzz', 'zzzzzzzz', 'zzzzzzzzz']
query = lambda prefix: [d for d in database if d.startswith(prefix)][:5]


def extract(query):
    
    db = set() # Use a Python set for our output backup to ensure we don't have repeated elements

    # Brute search
    import string
    alphabet = string.ascii_lowercase
    
    def add(db, r):
        ''' F. to add new values to db'''
        [db.add(i) for i in r]
        return db
    
    def next_letter(l):
        ''' Return next letter of the alphabet '''
        global alphabet
        return list(alphabet)[list(alphabet).index(l)+1]
    
    def root (s1, s2):
    
        root = ''
        for i, l in enumerate(s1):
            if s2[i] == l: 
                root += l
            else:
                break
            
        return root

    # Problems? --> Where there are more than 5 results for the same letter
    
    for l in alphabet:
                    
        r = query(l)
        if len(r) > 0: ## Avoid computations on none-results
            
            c = 1 
            second = r[1]              
            while True:  
                
                r_ = query(second[:c])
                db = add(db,r_)
                c += 1
                second = r_[-1]
                if len (r_) < 5:
                    break
            else:
                db = add(db, r)
    return db
        
result = extract(query)

        