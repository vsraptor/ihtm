import numpy as np
from modules.isdp_module import iSDPModule
from isdp import *
from encoders.scalar_encoder import *
import matplotlib.pylab as plt
import math
from memoization import cached
from ilexicon import *
from iutils import *


#!fixme: input data bits must > in sparsity bits

class iSpatialMapper(iSDPModule):


	def __init__(self, shape, spa=(0.05,0.05), learn_rates=(0.1,0.01), olap_match_pct=0.5, 
								cell_connectivity_pct=0.85, permanence_thresh=0.1, boost_lrate=0.01, encoder=None, elex=None):

		self.shape = shape
		self.in_vsize, self.out_vsize = self.shape
		self.in_spa, self.in_spa_nbits = spaORnbits(self.in_vsize, spa[0])
		self.out_spa,self.out_spa_nbits = spaORnbits(self.out_vsize, spa[1])
		self.spa = (self.in_spa, self.out_spa)
		self.spa_nbits = (self.in_spa_nbits, self.out_spa_nbits)
		self.vsize = shape

		#possible synapse states : 
		#    never-connected, may-connect-but-disconnected, connected
		# .. above thresh the synapse is assummed connected
		self.permanence_thresh = permanence_thresh
		# .. what % of all probable connections can be established
		self.cell_connectivity_pct = cell_connectivity_pct
		#how fast synapses learn/unlearn connections
		self.learn_rate = learn_rates[0] # 0.05
		self.unlearn_rate = learn_rates[1] #0.008

		#PERMANENCE/weights rectangual matrix with permanence
		#  between every in and out cell
		self.w = np.zeros(shape, dtype=np.float16)

		#boosting is used to make sure the least unused  
		# .. connections are used too
		self.do_boost = True if boost_lrate else False
		#boost weights 
		self.boost = np.ones(self.out_vsize, dtype=np.float16)
		self.boost_lrate = boost_lrate if boost_lrate else 0.01
		#how many times was this column used
		self.boost_count = np.ones(self.out_vsize, dtype=np.uint16)

		self.randomize_weights()

		#what % of bits must overlap to vote it a match
		self.olap_match_pct = olap_match_pct
		self.olap_match_bits = int(self.olap_match_pct * self.in_spa_nbits)

		#Encode/Decode functionality
		self.pre_trained = 0
		self.encoder = encoder # provides encode()/decode()
		self.elex = elex #lex-mem so : sm.rev_map => e.decode() => original
		if encoder is not None :
			assert encoder.vsize == self.in_vsize and encoder.spa_nbits == self.in_spa_nbits

		if elex is True : #not provided but create one
			self.elex = iLexicon(items=100, vsize=self.out_vsize, spaOnbits=self.out_spa_nbits)
		elif elex is not None : #check Lex cfg to match the SMapper
			assert elex.vsize == self.in_vsize and elex.spa_nbits == self.in_spa_nbits

	#randomize SP memory
	def randomize_weights(self, full=False, around_perm=False):
		if full :
			self.w = np.random.random((self.in_vsize, self.out_vsize))
		else : #part of the conns are zero
			for col in xrange(self.out_vsize) :
				cnt = int(self.in_vsize * self.cell_connectivity_pct)
		 		idxs = np.random.choice(xrange(0,self.in_vsize), cnt, replace=False )
				#select random weights/permanences
				if around_perm :
					self.w[idxs, col] = np.random.normal(self.permanence_thresh, scale=self.permanence_thresh/2., size=cnt) #around thresh
				else :
					self.w[idxs, col] = np.random.random(cnt) # in the 0 - 1 range


	def update_boost(self, top_ixs):
		#create boost vec ONE|1.5|2 when not in topN, ZERO when in topN
		boost_vec = np.ones(self.out_vsize)
		boost_vec[top_ixs] = 0 #smaller values penalize TOPN bits
		self.boost += self.boost_lrate * (boost_vec - self.boost)
		self.boost_count[top_ixs] += 1

	def pertrube(self, pmax=0.1):
		self.w += np.random.uniform(-pmax,pmax, self.w.shape)

	#Winner-take-all for sparsity algorithm
	def topN(self, vec, boost=True, calc_omb=False):
		# vec ! multiplication by 1 keeps the weight, by 0 disregards it
		#when not zero & permanence < thresh then sum perm verticaly to get the overlap
		olap = np.sum((vec.reshape(-1,1) * self.w) > self.permanence_thresh, axis=0)
		if boost : olap = (olap * self.boost)
		#exclude matches which have lower overlap than required
		olap[ olap < self.olap_match_bits ] = 0
		# print olap, vec
		assert ~ np.all(olap == 0), "probably wrong sparsity"

		col_ixs = np.argsort(olap) #sort by bigger overlap
		#winners-take-all up to output sparsity
		return col_ixs[::-1][:self.out_spa_nbits]


	def train(self, vec):
		assert vec.size >= self.in_spa_nbits, "spa-nbits: %s >= expected:%s" % (vec.size, self.in_spa_nbits)
		if isinstance(vec, (iSDP,i2DSDP)) : vec = vec.as01

		#given active rows (input), find the winning cols
		top_cols = self.topN(vec, boost=self.do_boost)
		#nudge positivily++ cross b/w 1s-rows and winning columns, and decrement-- the other rows
		ones  = np.ix_(vec.nonzero()[0],      top_cols) #coords: (rows,cols)
		zeros = np.ix_(np.where(vec == 0)[0], top_cols) #coords

		self.w[ones ] += self.learn_rate   * ( 1 - self.w[ones ])
		self.w[zeros] += self.unlearn_rate * ( 0 - self.w[zeros])

		if self.do_boost : self.update_boost(top_cols)

	def batch_train(self, data):
		for d in data : self.train(d)

	#encode and store the result for reverse-mapping i.e. decoding
	def encode(self, value, encoder=None) :
		if encoder is None and self.encoder is not None : encoder = self.encoder
		else : raise Exception("No Encoder specified !") 
		sdpi = encoder.encode(value)	
		rv = self.predict(sdpi)
		if self.elex is not None : self.elex.vivify(value, rv)
		return rv

	def decode(self, sdpi) :
		if self.elex is None : raise Exception("Decode requires lexer memory !") 
		rv = self.elex.bm(sdpi)
		return rv
	
	#prep the SM training it with random data
	def pre_train(self, nsamples, encoder=None, once=False):
		if self.pre_trained and once : return
		print "Pretraining Spatial mapper/Encoder... : ", nsamples
		if encoder is None and self.encoder is not None : encoder = self.encoder
		else : raise Exception("No Encoder specified !") 
		self.pre_trained += nsamples
		for i in xrange(nsamples): 
			if i % 1000 == 0 : log("... %s" % (i))
			self.train(encoder.random())

	def ed_train(self, nsamples):
		print "Encode/Decode training ... : ", nsamples
		rand = True
		if isinstance(nsamples,int):
			nsamples = xrange(nsamples) 
			rand = False

		for i in nsamples: 
			if i % 100 == 0 : log("... %s" % (i))
			if rand : self.train(self.encoder.random())
			else : self.train(self.encoder.encode(i))

	#train by constant learn rates with clipping
	def train_clip(self, vec):
		if isinstance(vec, (iSDP,i2DSDP)) : vec = vec.as01
		#given active rows, find the winning cols
		top_cols = self.topN(vec, boost=self.do_boost)
		#nudge positivily++ cross b/w 1s-rows and winning columns, and decrement-- the other rows
		ones  = np.ix_(vec.nonzero()[0],      top_cols) #coords: (rows,cols)
		zeros = np.ix_(np.where(vec == 0)[0], top_cols) #coords

		self.w[ones ] += self.learn_rate 
		self.w[zeros] += self.unlearn_rate 
		#clip to 0-1 range
		self.w[:,top_cols] = np.where(self.w[:,top_cols] > 1, 1, self.w[:,top_cols])
		self.w[:,top_cols] = np.where(self.w[:,top_cols] < 0, 0, self.w[:,top_cols])

		if self.do_boost : self.update_boost(top_cols)


	#dont touch ZERO perm, they should be disconnected at all times
	def train_skip0s(self, vec):
		if isinstance(vec, (iSDP,i2DSDP)) : vec = vec.as01
		#print vec
		#given active rows, find the winning cols
		top_cols = self.topN(vec, boost=self.do_boost)
		#print top_cols
		#nudge positivily++ cross b/w 1s-rows and winning columns, and decrement-- the other rows
		#exclude ZERO-0 perm !!
		ones  = np.ix_(vec.nonzero()[0],      top_cols) #coords: (rows,cols)
		ones_mask = self.w[ones] != 0
		zeros = np.ix_(np.where(vec == 0)[0], top_cols) #coords
		zeros_mask = self.w[zeros] != 0

		#print self.w[zeros]
		self.w[ones ] = np.where(ones_mask , self.w[ones]  + self.learn_rate   * ( 1 - self.w[ones]), self.w[ones])
		self.w[zeros] = np.where(zeros_mask, self.w[zeros] + self.unlearn_rate * ( 0 - self.w[zeros]), self.w[zeros])
		#print self.w[zeros]

		if self.do_boost : self.update_boost(top_cols)



	def predict(self, vec, astype='isdp'):
		# print 'p> ', vec
		if isinstance(vec, (iSDP,i2DSDP)) : vec = vec.as01

		#given active rows, find the winnivng cols
		cols = self.topN(vec, boost=True)
		rv = np.zeros(self.out_vsize, dtype=np.uint8)
		rv[cols] = 1
		if astype == 'isdp' : return isdp.np012sdp(rv) 
		return rv

	@cached(max_size=100,thread_safe=False)
	def mem_predict(self, vec, astype='sdp'): return self.predict(vec,astype)


	@property
	def info(self):
		print """
Shape: %s
Sparsity: %s
ON Bits count : (%s,%s)
Learn rates : (%s, %s)
Boost learn rate : %s (on:%s)
Overlap match %% : %s
Cell conn. %% : %s
Permanence thresh : %s
""" % (self.shape, self.spa, self.in_spa_nbits, self.out_spa_nbits, self.learn_rate, self.unlearn_rate, self.boost_lrate, self.do_boost,
		 self.olap_match_pct, self.cell_connectivity_pct, self.permanence_thresh)


##========================================= MODULE ===========================================


	def outs(self): return {'type':iSDP, 'vsize':self.vsize[1], 'spa_nbits': self.spa_nbits[1]}

	def check(self, prev_module) :
		info = prev_module.outs()
		if info['type'] != iSDP : return (True, "expecting iSDP, got %s" % info['type'])
		if self.vsize[0] != info['vsize'] : return (True, "vsize: mismatch %s => %s" % (self.vsize[0], info['vsize']))
		if self.spa_nbits[0] != info['spa_nbits'] : return (True, "spa_nbits: mismatch %s => %s" % (self.spa_nbits[0], info['spa_nbits']))
		return (False, "no err")

	def process(self, data, learn=True): 
		if learn : self.train(data)
		return self.predict(data)
		