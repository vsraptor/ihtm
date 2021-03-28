import numpy as np
import math
from encoder import Encoder
from iutils import *
from isdp import *

#Concatenate encoded results
class MultiEncoder(Encoder):

	def __init__(self, encoders):
		self.encoders = encoders
		self.vsize = 0
		self.lens = []
		self.spa_nbits = 0
		for enc in self.encoders :
			self.vsize += enc.vsize
			self.lens.append(self.vsize)
			self.spa_nbits += enc.spa_nbits

		self.spa, _ = spaORnbits(self.vsize, self.spa_nbits)


	@property
	def info(self):
		s = "=====================================\n"
		for e in self.encoders :
			s += e.info
			s += "-----------------------------------\n"
		s += "Total number of bits : %s\n" % self.vsize
		s += "Sparsity : %s\n" % self.spa
		s += "Sparse bits : %s\n" % self.spa_nbits
		return s

	#one data item per encoder
	def encode(self, data):
		if isinstance(data, (int,str)) : data = [data]
		assert len(data) == len(self.encoders), "Data <=> Encoder size mistmatch"
		encoded = [ self.encoders[i].encode(d) for i,d in enumerate(data) ]
		return isdp.concat(encoded)

	def decode(self, sdr):
		raise NotImplementedError #!fixme : must deconcatenate/split
		assert len(sdr) == self.vsize, "Mismatch in the size of SDR (%s) expected %s" % (sdr.size, self.vsize)
		rv = []
		# for i, enc in enumerate(self.encoders) :
		# 	#find start-end range of bits to change
		# 	start = 0 if i == 0 else self.lens[i-1]
		# 	rv.append( enc.decode(sdr[start: self.lens[i]]) )
		return rv


	#Encode some random value close to the passed value
	def nboor(self, values, radius=0.1, rv=None):
		return [ enc.nboor(val,radius,rv) for enc, val in zip(self.encoders, values) ]

	def nboors(self, values, samples=1, radius=0.05):
		rv = []
		for v in values :
			vals = []
			for i in xrange(samples) :
				lst = self.nboor(v,radius)
				vals.append(lst)
			rv.append(vals)
		return rv



	def process(self, data): return self.encode(data)
	
