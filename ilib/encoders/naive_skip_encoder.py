import math
import numpy as np
from isdp import *
from iutils import spaORnbits
from ilexicon import iLexicon

"""

The normal procedure is to Encode the values and then pass it trough Spatial Pooler, 
but it is expensive operation. What if we can encode directly and skip SP.

This is a Naive algorithm I came up to do just that, here is how it works.

The algorithm depends on 3 parameters.

	- vsize : the number of bits of the SDP
	- spaOnbits : sparsity percent OR number of bits
	- olap_nbits : how many bits do neighboring items overlap

First we have to calculate capacity i.e. how many items we can represent.
For example if the SDP is 1000 bits, spa-bits is 20 and neighboring items dont overlap,
then the capacity is 1000/20 = 50 items i.e. we can generate randomly 50 items the	 
51st will necessarily generate item that will overlap with some other item.

With me so far ? 
Now if we allow 10 bits overlap, we can generate 1000/(20-10) = 100 items, before we 
generate item that will create item with bigger overlap.

Now that we know the capacity, how do we do it :

We keep a set of all free for use bits. In the beginning its all of the 1 ... 1000.
We generate spa-number-of-bits randomly and remove them from the free-set.
Next we pick randomly olap-number of bits from the previous item (in this case the first one)
and non-olap-number of bits from the free-set, where :

		self.non_olap = self.spa_nbits - self.olap_nbits

Then we rinse and repeat until we exhaust all the possible items..

"""

class NaiveSkipEncoder:

	def __init__(self, minimum,maximum,vsize,spaOnbits=10,olap_nbits=5):
		self.vmin = minimum
		self.vmax = maximum
		self.vrange = self.vmax - self.vmin
		self.vsize = vsize
		self.spa, self.spa_nbits = spaORnbits(self.vsize, spaOnbits)

		assert olap_nbits < self.spa_nbits
		self.olap_nbits = olap_nbits
		#number of bits that dont overlap in neighboring items
		self.non_olap = self.spa_nbits - self.olap_nbits

		#number of items that can be expressed with the selected overlap
		self.nitems = int( self.vsize / (self.spa_nbits - self.olap_nbits) )
		#lookup table ( +1 :first elem of lex is for system use)
		self.lex = iLexicon(items=self.nitems+1,vsize=vsize,spaOnbits=spaOnbits)
		self.build_table()

	#create the iSDP lookup table
	def build_table(self):
		#list of bit-ixs which are free for use
		unused_ixs = set(range(0, self.vsize))
		#add the first item 
		prev_ixs = set( np.random.choice(list(unused_ixs), size=self.spa_nbits,replace=False) )
		self.lex.add(0,sorted(prev_ixs))

		for i in range(1,self.nitems) :
			#next-item = 50% of prev item bits + 50% of unused/free bits
			ixs_olap = np.random.choice(list(prev_ixs), size=self.olap_nbits,replace=False)
			ixs_new = np.random.choice(list(unused_ixs), size=self.non_olap,replace=False)
			prev_ixs = set(list(ixs_olap) + list(ixs_new))
			unused_ixs.difference_update(prev_ixs) # remove the bits that got used
			# print(sorted(prev_ixs))
			self.lex.add(i,sorted(prev_ixs))

	def pos(self, value) :
#		if value == 0 : return 0
		return int( math.floor(self.nitems * ((value - self.vmin)/float(self.vrange)) ) )

	def encode(self, value):
		if value < self.vmin or value > self.vmax : 
			warnings.warn("Value '%s' outside of range : [%s <=> %s]" % (value, self.vmin, self.vmax))
		i = self.pos(value)
		return self.lex.get(i+1) # +1 because first elem of lex is for system use

	def decode(self, data): 
		key = self.lex.best_match(data)
		# if key == '!' : key = 0
		value = ( (key * self.vrange) / float(self.nitems) ) + self.vmin
		return math.floor(value)


	@property
	def info(self):
		s = "> Naive Skip encoder -----\n"
		s += "min-max/range : %s-%s/%s\n" % (self.vmin,self.vmax,self.vrange)
		s += "vsize,olap/non_olap : %s,%s/%s\n" % (self.vsize,self.olap_nbits,self.non_olap)
		s += "nitems : %s\n" % (self.nitems)
		s += "sparsity : %.2f, spa_nbits:%s\n" % (self.spa, self.spa_nbits)
		return s

	def similarity_map(self):
		self.sims = np.zeros((self.nitems,self.nitems))
		for i in range(1,self.nitems):
			for j in range(1,self.nitems):
				self.sims[i-1,j-1] = self.lex[i] // self.lex[j]

	def similarity_map2(self):
		self.sims2 = np.zeros((self.vrange,self.vrange))
		for i in range(self.vmin,self.vmax):
			for j in range(self.vmin,self.vmax):
				self.sims2[i-self.vmin,j-self.vmin] = self.encode(i) // self.encode(j)

