import numpy as np
# from encoder import Encoder
from isdp import *
from iutils import *
#from memoization import cached


"""

the input to the GCE is a number


"""

class GridCellLikeEncoder(object):


	def __init__(self, vsize=NBITS, spaOnbits=None, gm_size=None, modules=None):

		# if spaOnbits is not None :
		# 	self.spa, self.spa_nbits = spaORnbits(vsize, spaOnbits)
		# else :
		# 	assert gm_size is not None, "Either spaOnbits or gm_size is required argument"

		# if gm_size is None : gm_size = vsize/self.spa_nbits

		# if not hasattr(self, 'spa_nbits'): 
		# 	self.spa, self.spa_nbits  = spaORnbits(vsize, vsize/gm_size)

		# self.gm_size = gm_size
		# self.gm_num = self.spa_nbits
		# self.vsize = vsize

		# assert self.vsize == self.gm_num * self.gm_size, "vsize:%s == GM size * count:%s" % (self.vsize, self.gm_num * self.gm_size)

		self.vsize = vsize
		self.spa, self.spa_nbits = spaORnbits(vsize, spaOnbits)

		if modules is None : 
			self.modules = np.array(nprimes(start=11,cnt=self.vsize), dtype=DTYPE)
			np.random.shuffle(self.modules)
		else : self.modules = np.array(modules, dtype=DTYPE)	
		self.min_prime, self.max_prime = self.modules.min(), self.modules.max()


	def encode(self, value) :
		reminders = (value) % self.modules
		return iSDP(vsize=self.vsize, vsn=np.argsort(reminders)[:self.spa_nbits])

	# @cached(max_size=100)
	def decode(self, sdpi, rng=range(0,100)) :
		olap = []
		for i in rng :
			enc = self.encode(i)
			olap.append(isdp.olap(enc, sdpi))
		return np.argmax(olap)	


	def similarity_map(self, rng=range(100)):
		size = len(list(rng))
		self.sims = np.zeros((size,size))
		for i,x in enumerate(rng):
			for j,y in enumerate(rng):
				self.sims[i,j] = self.encode(x) // self.encode(y)
	