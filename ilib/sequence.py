from isdp import *
import numpy as np

"""
  
  Manages iSDP sequence : union of shifted iSDP's

"""


class Sequence(object):

	def __init__(self, start=1, end=10, lex=None):

		self.seq = None
		self.start = start 
		self.current = start
		self.end = end
		self.lex = lex


	def add(self, element):
		assert self.current < self.end, "the sequence is full"

		rolled = isdp.roll(element, roll_nbits=self.current)
		if self.seq is None : self.seq = rolled
		else : self.seq = isdp.union([self.seq, rolled])
		self.current += 1
 
	def add_items(self, lst):
		for el in lst : self.add(el)

	def isin(self, element, ix=None):

		if ix is not None : return isdp.roll(element,ix) in self.seq

		found, rv = False, []
		for n in xrange(self.start, self.end) :   
			found = isdp.is_sim( isdp.roll(element,n), self.seq, thresh=0.8 )
			if found : rv.append(n)
		return rv	

	def __contains__(self, element): return self.isin(element)
	def __len__(self): return self.current - 1

	# def __eq__(self, right) :
	# 	if isinstance(right, (iSDP,i2DSDP)) : return self.seq == right
	# 	return self.seq == right.seq	

	def __getattr__(self, name):
		return getattr(self.seq, name)

	def __getitem__(self, ix):
		assert self.lex is not None, "The sequence need to have Lexicon"
		if isinstance(ix,int) :
			assert self.start <= ix < self.current, "Outside of range : %s <=> %s" % (self.start, self.current-1)
			rolled = isdp.roll(self.seq, roll_nbits=-ix)
			return self.lex.bm(rolled)

		if isinstance(ix,slice) :
			rv = []
			start = self.start if ix.start is None or ix.start < self.start else ix.start
			stop = self.current if ix.stop is None or ix.stop > self.current else ix.stop
			step = 1 if ix.step is None else ix.step

			for i in range(start,stop,step) :
				rolled = isdp.roll(self.seq, roll_nbits=-i)
				rv.append(self.lex.bm(rolled))
			return rv	

	def roll(self, ix): return isdp.roll(self.seq, roll_nbits=ix)


