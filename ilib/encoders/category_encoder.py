import numpy as np
import math
from encoder import Encoder
from isdp import iSDP
from iutils import *

class CategoryEncoder(Encoder) :

	def __init__(self, vsize, ncats, start_category=0):
		self.ncats = ncats
		self.vsize = vsize
		self.start_category = start_category
		self.end_category = start_category + self.ncats
		self.spa_nbits = self.vsize / ncats #width
		assert self.vsize % self.ncats == 0, "Categories cannot overlap, reminder must be zero : %s " % (self.vsize % self.ncats)
		self.spa, self.spa_nbits = spaORnbits(self.vsize, self.spa_nbits)

	@property
	def info(self):
		s = "> Category encoder -----\n"
		s += "Num of categories : %s\n" % self.ncats
		s += "Num of bits : %s\n" % self.vsize
		s += "sparsity : %.2f, spa_nbits:%s\n" % (self.spa, self.spa_nbits)
		return s

	def encode(self, value):
		i = int( math.floor((value) * self.spa_nbits) )
		assert self.start_category <= value < self.end_category, "category value outside of range : %s : [%s <=> %s]" % (value, self.start_category, self.end_category-1)
		return iSDP(vsize=self.vsize, vsn=np.arange(i, i+self.spa_nbits))

	def decode(self, data):
		i = data[0]
		value = int( math.floor( i / self.spa_nbits ))
		return value+1

	def random(self): return self.encode(np.random.randint(self.start_category, self.start_category + self.ncats))
	
	#Encode some random value close to the passed value
	def nboor(self, value, radius=0.1, rv=None):
		x = int(radius * self.ncats)
		if value > self.ncats : value = self.ncats
		if value < 0 : value = 0
		bottom = 0 if value-x < 0 else value-x
		top = self.ncats if value+x > self.ncats else value+x
		nval = top
		if bottom != top : nval = np.random.randint(bottom, top)
		pct_diff = abs(nval-value)/float(self.ncats)
		if rv == 'isdp' : return self.encode(value)
		return value, nval, pct_diff

	#LoL: [ [[value, neighboor-value, percent-diff ], .. ],  [[ .. ], ....], [ ... ], ... ]
	def nboors(self, values, samples=1, radius=0.05):
		rv = []
		for v in values :
			vals = []
			for i in xrange(samples) :
				val, nval, pct_diff = self.nboor(v,radius)
				vals.append([val, nval, pct_diff])
			rv.append(vals)
		return rv
