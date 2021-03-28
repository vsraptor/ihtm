
import numpy as np
import math
import warnings
from encoder import Encoder
from memoization import cached
from isdp import *
from iutils import spaORnbits

class ScalarEncoder(Encoder):

	def __init__(self, minimum,maximum,vsize,buckets=100,spaOnbits=5):

		self.vmin = minimum
		self.vmax = maximum
		self.vrange = self.vmax - self.vmin
		self.vsize = vsize
		self.spa, self.spa_nbits = spaORnbits(self.vsize, spaOnbits)

		if (vsize is None) :
			self.buckets = buckets
			self.vsize = buckets + self.spa_nbits  #+ 1
		else :
			self.vsize = vsize
			self.buckets = vsize - self.spa_nbits  #+ 1

		#what range of values, single bucket covers
		self.resolution = self.vrange/float(self.buckets)


	@property
	def info(self):
		s = "> Scalar encoder -----\n"
		s += "min-max/range : %s-%s/%s\n" % (self.vmin,self.vmax,self.vrange)
		s += "buckets,width,n : %s,%s,%s\n" % (self.buckets,self.spa_nbits,self.vsize)
		s += "resolution : %.2f, %.4f%%\n" % (self.resolution, self.resolution/float(self.vrange))
		s += "sparsity : %.2f, spa_nbits:%s\n" % (self.spa, self.spa_nbits)
		return s

	def pos(self, value) :
#		if value == 0 : return 0
		return int( math.floor(self.buckets * ((value - self.vmin)/float(self.vrange)) ) )

#	@cached(max_size=100)
	def encode(self, value):
		if value < self.vmin or value > self.vmax : 
			warnings.warn("Value '%s' outside of range : [%s <=> %s]" % (value, self.vmin, self.vmax))
		i = self.pos(value)
		return iSDP(vsize=self.vsize, vsn=np.arange(i, i+self.spa_nbits))
		
	def random(self): return self.encode(np.random.randint(self.vmin, self.vmax))

#	@cached(max_size=100)
	def decode(self, data):
		value = ( (data[0] * self.vrange) / float(self.buckets) ) + self.vmin
		if data[0] > 0 : value += 1
		return math.floor(value)

	#Encode some random value close to the passed value
	def nboor(self, value, radius=0.1, rv=None):
		x = int(radius * self.vrange)
		if value > self.vmax : value = self.vmax
		if value < self.vmin : value = self.vmin
		bottom = self.vmin if value-x < self.vmin else value-x
		top = self.vmax if value+x > self.vmax else value+x
		nval = np.random.randint(bottom, top)
		pct_diff = abs(nval-value)/float(self.vrange)
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


