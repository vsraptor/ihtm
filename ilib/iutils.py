from __future__ import print_function
import random 
import numpy as np
import string

WORD_DICT = []

WORD = 1
CHAR = 2
INT  = 3
SENT = 4
CORPUS = 5

say = print
log = print

def rand_words(nwords=1):
	global WORD_DICT
	if len(WORD_DICT) == 0 :
		for w in open("/usr/share/dict/words").read().splitlines() :
			WORD_DICT.append( filter(lambda x: x  in string.ascii_lowercase, w.lower()) )
	return list(np.random.choice(WORD_DICT,size=nwords, replace=False))

#given sparsity percent OR num-bits, returns both
def spaORnbits(vsize, spaOnbits): 
	spa, spa_nbits = None, None
	if isinstance(spaOnbits,float) :
		spa = spaOnbits
		spa_nbits = int(vsize * spa)
	elif isinstance(spaOnbits, int) :
		spa_nbits = spaOnbits
		spa = spa_nbits/float(vsize)
	else :
		raise Exception("spaORnbits: Wrong parameters ...")
	
	return spa, spa_nbits

def char_range(c1, c2):
    """Generates the characters from `c1` to `c2`, inclusive."""
    for c in xrange(ord(c1), ord(c2)+1): yield chr(c)


def tokenize(string, level=WORD, split_sym=' ' ):
    if level is WORD : return string.split(split_sym)
    else: return list(string)

#python strings are immutable, this fun changes
# a character at position ix and returns new-updated string
def change_char(line, ix, new_char):
	tmp = list(line)
	tmp[ix] = new_char
	return ''.join(tmp)

#convert slice to xrange
def slice2range(s):	
	start = s.start if s.start is None else 0
	end = s.end if s.end is None else 0
	stop = s.stop if s.stop is None else 0
	return xrange(start,end,stop)

#detects is this a string-of-chars OR list-of-words OR string-of-separated-words
#  OR str-of-integers OR list-of-integers
def detect_tokens(tokens, sep=',', tpl_sym='.'):
	if sep in tokens : #seems like WORDS
		if tpl_sym in tokens : tokens = tokens.replace(tpl_sym, sep+tpl_sym)
		tokens = filter(None, tokens.split(sep))

	if isinstance(tokens[0], int) or str.isdigit(tokens[0]) :
		return INT, [ (t if t == '.' else int(t) ) for t in tokens]
	elif len(tokens) == 1 and isinstance(tokens[0], str) : 
		return CHAR, tokens
	else :
		return WORD, tokens


#Calculate minimum edit distance, you can provide the cost of edit ops
def med(tokens1, tokens2, subst=1, delete=1, insert=1):
	n,m = len(tokens1), len(tokens2)
	D = np.zeros((n+1, m+1), dtype=np.uint16)
	D[0,1:] = range(1, m+1)
	D[1:,0] = range(1, n+1)

	for i in xrange(1,n+1):
		for j in range(1,m+1):
	# for i,j in np.ndindex(D.shape) :	
			subst_cost = 0 if tokens1[i-1] == tokens2[j-1] else subst
			D[i,j] = min(D[i-1, j] + insert, D[i, j-1] + delete, D[i-1, j-1] + subst_cost)

	return D[n,m]

#generate primes in number range
def primes(start=1, end=100):
	rv = []
	for x in range(start, end+1) :
		for y in range(2,x) :
			if x % y == 0 : break
		else : rv.append(x)
	return rv

#generate n-primes starting from a number
def nprimes(start=1, cnt=100):
	rv = []
	x = start
	while len(rv) < cnt :
		for y in range(2,x) :
			if x % y == 0 : break
		else : rv.append(x)
		x += 1
	return rv

#coprime generation
def coprime(n, a=1, b=1): 
### generates all relatively prime pairs <= n.  The larger number comes first. 
	yield (a,b) 
	k = 1 
	while a*k+b <= n: 
			for i in coprime(n, a*k+b, a): 
				yield i 
			k += 1 

#coprime generation
def farey(limit): 
	'''Fast computation of Farey sequence as a generator''' 
	# n, d is the start fraction n/d (0,1) initially                             
	# N, D is the stop fraction N/D (1,1) initially                              
	pend = [] 
	n = 0 
	d = N = D = 1 
	while True: 
		mediant_d = d + D 
		if mediant_d <= limit: 
			mediant_n = n + N 
			pend.append((mediant_n, mediant_d, N, D)) 
			N = mediant_n 
			D = mediant_d 
		else: 
			yield n, d 
			if pend: n, d, N, D = pend.pop()
			else: break


def flatten(lol): return sum(lol,[])

#one step of the decay
def decay(val, rate): return val * np.exp(-rate)

def decay_plot(ticks=1000, pulse=[5,25], rate=0.01):
	data = [ ]
	value = 0
	pulse = np.array(pulse)
	for t in xrange(ticks) :
		if np.any(t  == pulse) : value += 1
		else : value = decay(value, rate)
		data.append(value)
	return data	
