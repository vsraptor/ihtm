import numpy as np
from isdp import *
from dyn_mem import *
from itertools import permutations, islice
import string
import warnings
import inspect
import pprint
from iutils import *

ERASED = '__erased__'

class iLexicon:
	__slots__ = 'lex','lex_inv','items','vsize','spa','spa_nbits', 'mem'

	def __init__(self, items=100, vsize=NBITS, spaOnbits=SPA) :
		self.lex = {}
		self.lex_inv = []
		self.items = items
		self.vsize = vsize

		if isinstance(spaOnbits,float) :
			self.spa = spaOnbits
			self.spa_nbits = int(self.vsize * self.spa)
		elif isinstance(spaOnbits, int) :
			self.spa_nbits = spaOnbits
			self.spa = self.spa_nbits/float(self.vsize)

		self.mem = DynMem((self.items, self.spa_nbits), vsize=self.vsize)
		self.add("!")


	def __contains__(self, key) : return key in self.lex

	def __getattr__(self, key) : return self.get(key)
	def __getitem__(self, key) : return self.get(key)
	def __setitem__(self, key, value=None) :
		if value is None : self.add(key)
		else : self.add(key, value)

	#count the matching bit-ixs between the vector/iSDP and every row in the memory 
	def olaps(self, vec1d): return np.isin(self.mem.ary[:self.mem.ix_empty,:], vec1d, assume_unique=True).sum(axis=1)
	#get back row ixs with best overlaps
	def bixs(self, vec, topn=1) :
		olaps = self.olaps(vec)
		if len(olaps[olaps > (self.spa_nbits * SIM_THRESH)]) == 0 : return None
		return np.argsort( olaps )[::-1][:topn] #first-bigger
	best_ixs = bixs

	#return best symbols
	def bt(self, vec, topn=1) :
		ixs = self.bixs(vec, topn=topn)
		# print ixs
		if ixs is None : return ["!"]
		return [ self.lex_inv[ix] for ix in ixs ]
	best_top = bt

	#get the best symbol
	def bm(self, vec) : 
		return self.bt(vec,topn=1)[0]
	best_match = bm		

	def add(self, name, vec=None, spa=None):
		if name in self.lex : raise Exception(">%s< already exists" % name)
		if spa is None : spa = self.spa
		ix, new = self.mem.add(vec)
		#print "add> %s[%s] new:%s" % (name, ix,new)
		self.lex[name] = ix
		#did we used new memory slot OR reused old one
		if new : self.lex_inv.append(name)
		else : self.lex_inv[ix] = name
		return ix

	def remove(self, name):
		self.mem.remove(self.lex[name])
		ix = self.lex.pop(name) #remove the old name
		self.lex_inv[ix] = ERASED

	def erase(self):
		self.mem.erase()
		self.lex = {}
		self.lex_inv = []

	def add_items(self, lst, spa=None, vivify=False) :
		if spa is None : spa = self.spa
		for el in lst :
			if vivify : self.vivify(el) 
			else : self.add(el, spa=spa)

	def get(self, idn):
		if idn is None : return None
		if isinstance(idn,int) : return iSDP(vsize=self.vsize, vsn=self.mem.get(idn))
		if idn not in self.lex : 
			warnings.warn("Item '%s' does not exist" % idn)
			return None
		return iSDP(vsize=self.vsize, vsn=self.mem.get( self.lex[idn] ))

	def set(self, name, value) : self.mem.set( self.lex[name], value)

	def exists(self, name): return name in self.lex
	def rename(self, old_name, new_name) :
		self.lex[new_name] = self.lex.pop(old_name)

	def vivify(self, name, value=None, reset=False):
		if value is None : value = iSDP(vsize=self.vsize, vsn=self.spa_nbits)
		if not self.exists(name) : self.add(name, value) 
		else :
			if reset : self.set(name, value)
			else : return None #do nothing if exists and dont want to reset
		return value

	#generate random list of hypervectors, name them from 'a' to 'z'
	def az(self): self.add_items(list(string.ascii_lowercase), vivify=True)
	
	#generate alpha symbols in the range
	def sym_range(self, c1, c2): 
		for sym in char_range(c1, c2) : yield self.get(sym)

	def syms(self):#symbols used as special cases
		self.add_items([' ', '.', '-','=', '+','?', '$', '%' , '@', '_' ], vivify=True)

	#generates all|some permutation of the provided characters
	def permuted_syms(self, syms='abc',repeat=2, cnt=None):
		for tup in islice(permutations(syms,r=repeat), cnt) :
			name = ''.join(tup)
			pre = tuple([ self.get(n) for n in tup ])
			bundled = isdp.thin(isdp.union(pre),new_spaOnbits=self.spa_nbits)
			self.vivify(name, bundled)

	def rand_items(self, cnt=10):
		for i in xrange(cnt):
			name = '_' + ''.join([ random.choice(string.ascii_lowercase) for _ in xrange(5)])
			self.add(name)

	#apply operation on a bunch of operands
	def apply(self, op, rng1, rng2):
		res = []
		#character range
		if isinstance(rng1, str) : rng1 = list(self.sym_range(rng1[0],rng1[1]))
		#if range2 is single sym .. multiply it to match the lrn of range1
		if not isinstance(rng2, list) : rng2 = [rng2] * len(rng1)
		#collect the results
		for a1, a2 in zip(rng1,rng2) : res.append( op(a1,a2) )
		return res	


	def __repr__(self):
		s = '' 
		for key,ix in self.lex.items() :
			s += "%s: %s\n" % (key,self.get(ix))
		return s	

	# @property
	def info(self) :
		s = "Shape : %s\n" % ((self.items, self.spa_nbits),)
		s += "Sparsity : %s\n" % self.spa
		s += self.mem.info
		return s