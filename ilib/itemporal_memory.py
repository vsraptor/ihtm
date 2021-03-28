import numpy as np
from modules.isdp_module import iSDPModule
import matplotlib.pylab as plt
from isdp import *
from isegmented_memory import *
from iutils import *



class iTemporalMemory(iSDPModule) :

	def __init__(self, vsize=NBITS, spaOnbits=SPA_NBITS, nsegments=5, sm={}):

		self.vsize = vsize
		self.nsegments = nsegments
		self.vsize2D = self.nsegments * self.vsize
		self.spa, self.spa_nbits = spaORnbits(self.vsize, spaOnbits)

		assert vsize * nsegments < 65000, "Too many segments %s ... use max: %s " % (nsegments, 65000/vsize)

		#init Memory
		sm_cfg = {'shape': (self.nsegments, self.vsize, self.spa_nbits)}
		self.sm = iSegmentedMemory(**sm_cfg)
	
		#TM state arrays
		step = self.nsegments + 1
		self.predicted = self.buf2D(step, step)
		self.now       = self.buf2D(self.spa_nbits * step + step, step)
		self.before    = self.buf2D(self.spa_nbits * 2 * step + step, step)

		self.backup = {} #store i2DSDP

		self.start_sym = self.asym
		self.end_sym   = self.asym

		self.syms = {}

	#initialize a buffer
	def buf2D(self, start, step): 
		return i2DSDP(vsize=self.vsize2D, nrows4col=self.nsegments, #vsn=self.start_sym)
			vsn=isdp.full(vsize=self.vsize2D, spa_nbits=self.spa_nbits, start=start, step=step) )
		#, empty='null')

	@property
	def info(self) :
		s = "> Temporal Memory =========================================\n"
		s += ">>>  Buffers : \n"
		s += self.now.info
		s += ">>>  Memory : "	
		s += self.sm.info
		return s

	@property
	def asym(self): return isdp.rand(vsize=self.vsize2D, spaOnbits=self.spa_nbits, nrows4col=self.nsegments, ret='i2DSDP')

	#pre/post initiastart_symlization for sequences
	def prep(self, start_sym=None):
		if start_sym is None : start_sym = self.start_sym.copy()
		self.predicted = start_sym
		self.now = self.start_sym.copy()
		self.before = self.start_sym.copy()

	def store(self):
		self.backup['predicted'] = self.predicted.copy()
		self.backup['now'] = self.now.copy()
		self.backup['before'] = self.before.copy()

	def restore(self):
		self.predicted = self.backup['predicted'].copy()
		self.now = self.backup['now'].copy()
		self.before = self.backup['before'].copy()


	#store current before state for learning purposes (VAL)
	def copy(self, buf1, buf2): buf2[:] = buf1.copy()

	def clean_slate(self):
		self.before.clear()
		self.predicted.clear()
		self.now.clear()

	#given bursted columns return the offset, we don't burst columns instead
	# we just find the next available segment/cell 
	def find_segments(self, bursted) :
		assert np.all(bursted < self.vsize), "%s < %s" % (bursted, self.vsize) 
		offsets = np.zeros(len(bursted), dtype=DTYPE) #updated segments
		for i, row in enumerate(bursted) :
			offsets[i] = self.sm.bit_learn(self.before, row)
		return offsets	

	#merge DATA(1D) + PREDICTED(2D) => now(2D)
	#the goal is to transpose 1D SDP to 2D SDP, where 
	def merge(self, data, data_buf, dest_buf):

		#Activate whole col or specific cell in col, depending on the predicitve state
		dest_buf.clear() # dest is .now
		self.data_buf_cols = data_buf.as1d

		_ , self.pred_match_ixs, self.data_match_ixs = np.intersect1d(self.data_buf_cols, data, assume_unique=True, return_indices=True)
	
		#... for matched data-data_buf-cols copy values from data_buf
		self.matched_cols = data_buf[self.pred_match_ixs]

		#... for non-matched/bursted, scale data * nseg + adjusted-cols
		inv_mask = np.ones(len(data), np.bool)
		inv_mask[self.data_match_ixs] = 0
		self.bursted_cols = data[inv_mask]

		#mini-learn: update/burst-bit (row,col/seg)
		self.offsets = self.find_segments(self.bursted_cols)
		#adjust bursted-col to bits-in-col based on the segment which 'absorbed' the data
		self.adjusted = np.unique(np.asarray(self.bursted_cols) * self.nsegments + self.offsets)
		
		assert len(self.adjusted) == len(self.bursted_cols), "%s : %s " % (self.adjusted, self.bursted_cols)
		# print self.matched_cols, self.adjusted

		#combine matched and bursted-adjusted cells into the destination buffer
		dest_buf[:] = np.sort(np.hstack((self.matched_cols, self.adjusted)))

		return self.adjusted #train only the bursted

	def predict(self, data, setit=True, ret='2D') :
		pred = self.sm.predict(data, ret=ret)
		if setit and ret == '2D' : self.predicted[:] = pred 
		return pred

	def step(self, data, learn=True, lex=None) :

		# if lex is not None : print self.status(data,lex)

		#merge incoming data with PREDICTED =into=> NOW
		adjusted = self.merge(data, data_buf=self.predicted, dest_buf=self.now)

		if learn : 
			# self.sm.learn(data=self.before, pred=self.now, kind='2D') #resize-much
			#train only segments in the bursted/adjusted column
			self.sm.learn(data=self.before, pred=adjusted, kind='2D') #resize-much
			# print "learn> b:%s:%s => n:%s:%s" % (lex.bm(self.before.as1d), self.before[0],lex.bm(self.now.as1d),self.now[0])

		# print "copy> n:%s => b:%s" % (lex.bm(self.now.as1d), lex.bm(self.before.as1d))
		self.copy(self.now, self.before) #copy before predict

		#set the cells in predicted state
		rv = self.predict(self.now, setit=True, ret='2D') 
		# print "predict> n:%s:%s => p:%s:%s" % (lex.bm(self.now.as1d), self.now[0], lex.bm(rv.as1d), rv[0])

		# if lex is not None : print self.status(data,lex)

		return rv
	
	#open-loop
	def steps(self,  data, learn=True, nsteps=1) :
		prev = data
		rv = [prev]
		for s in xrange(nsteps) :
			prev = self.step(prev,learn)
			rv.append(prev)
		return rv

	#buffer and lex-resolved info
	def status(self, data, lex):
		rv = "D:%s | " % lex.bm(data)
		for c, buf in zip(('N', 'B', 'P'), (self.now, self.before, self.predicted) ) :
			rv += "%s:%s:%s | " % (c,lex.bm(buf.as1d),buf[0])
			self.syms[lex.bm(buf.as1d) + str(buf[0])] = 1
		return rv


		
##========================================= MODULE routines ===========================================

	
	def outs(self): return {'type':iSDP, 'vsize':self.vsize, 'spa_nbits': self.spa_nbits}

	def check(self, prev_module) :
		info = prev_module.outs()
		if info['type'] != iSDP : return (True, "expecting iSDP, got %s" % info['type'])
		if self.vsize != info['vsize'] : return (True, "vsize: mismatch %s => %s" % (self.vsize, info['vsize']))
		if self.spa_nbits != info['spa_nbits'] : return (True, "spa_nbits: mismatch %s => %s" % (self.spa_nbits, info['spa_nbits']))
		return (False, "no err")

	def process(self, data, learn=True): self.step(data, learn=learn)	