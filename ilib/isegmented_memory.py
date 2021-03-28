import numpy as np
from modules.isdp_module import iSDPModule

from isdp import *
from dyn_mem import *

def filter_zeros(ary): return ary[ ~np.all(ary == 0, axis=1) ]

class iSegmentedMemory(iSDPModule):
	#__slots__ = "shape","vsize","vsize2D","nsegments","spa_nbits","spa","nrows"#,"segs"

	def __init__(self, shape=(3, NBITS, SPA_NBITS), override_spa_nbits=None) :

		# print "SM:shape: %s" % (shape,)
		if isinstance(shape[2],float) : #if spa is %
			shape = (shape[0],shape[1], int(shape[1] * shape[2]) )

		#if override_spa_nbits is specified, store only this many bits, from the whole pattern.
		# In the brain 8-20 syn are enough to recognize pattern
		self.partial_bits = False
		if override_spa_nbits is not None : #reformat shape
			shape = (shape[0],shape[1],override_spa_nbits)
			self.partial_bits = True

		#( number of segements, output-bit-size, number-of-active-input-bits)
		self.shape = shape
		self.vsize = self.shape[1] #Y-axis: virtual-size
		self.match_thresh = 0.1 #below this thresh is assumed noise
		self.nsegments = self.shape[0] #Z-axis
		self.spa_nbits = self.shape[2] #X-axis
		self.spa = self.spa_nbits / float(self.vsize)

		self.vsize2D = self.vsize * self.nsegments
		#holds the count of items stored in every cell
		self.fill = np.zeros(shape[:2], dtype=np.byte) #for debugging purposes
		self.stored_transitions = 0

		self.segs = np.zeros(shape, dtype=DTYPE) + NIL  #cant be ZERO
		#The arrays are reset with NIL/65535 as a value instead of ZERO, because
		#.. zero messes up the np.isin() in .olaps() calculations when the 
		#.. input vector contains ZERO .. which means zeroth-bit is set to 1
		#because when the ary row is all zeros and the input vector has a ZERO
		#.. the calculated overlap is the row length, rather than 0

		self.tmp = [] #!fixme

	#count the matching bit-ixs between the vector/iSDP and every row in the memory 
	def olaps(self, vec1d, seg): return np.isin(self.segs[seg,:,:], vec1d).sum(axis=1)
	#get back row ixs with best overlaps
	def bixs(self, vec, seg, topn=1) :
		sim = self.olaps(vec, seg) / self.spa_nbits
		return np.argsort( sim )[::-1][:topn] #first-bigger
	best_ixs = bixs


##================================= LEARN ===========================================

	#find the best segment/col for (row, data)

	def bit_learn(self, data, row):
		segment = None
		items = self.segs[:, row, :]
		#first calc similarity
		olap = np.isin(items, data).T.sum(axis=0) 
		sim = olap / float(self.spa_nbits)
		#either pick max-similar if there is perfect match OR all segments are filled/non-zero
		if sim.max() > 0.90 :#or np.all( sim != 0) : 
			segment = sim.argmax()
		else : #pick randomly one of the zero segments
			is_empty = np.argwhere(items[:,0] == NIL).T[0]
			if len(is_empty) > 0 :
				segment = np.random.choice(is_empty)
			else : 
				segment = sim.argmax()
				# segment = np.random.choice(np.argwhere(olap == 0).T[0])

			self.fill[segment, row] += 1	#update counter! 
			# if self.fill[segment, row] > 1 : self.tmp.append([row,segment])#, self.segs[segment, row, :], data, sim]) 

		return segment

	#updates segments at a row 
	def update_row(self, data, row, kind):
		#!fixme: assert 0 <= row < self.vsize, "'row' outside of the boundary: %s" % row

		#depending on whether the input is 1D or 2D
		if kind == '1D' : 
			segment = self.bit_learn(data, row) #pick a segment/offset
			vsize = self.vsize
		else : #2D .. bit coords already calculated convert to (row,segment) 
			segment = row % self.nsegments
			row = int(row/self.nsegments) #recalc row
			vsize = self.vsize2D

		new_value = data 
		#if the segment is not empty merge with the input data
		if ~ np.any(self.segs[segment, row, :] == NIL) :
			union = isdp.union((self.segs[segment, row, :], data), vsize=vsize)
			new_value = isdp.thin(union, new_spaOnbits=self.spa_nbits)

		#store the new merge-thinned value
		if self.partial_bits : #pat detection work even with 10-20 bits 
			self.segs[segment, row, :] = isdp.thin(new_value, new_spaOnbits=self.spa_nbits)	
		else : #keep the full value	
			self.segs[segment, row, :] = new_value	

		return segment

	#for every row and offset do update 
	def learn(self, data, pred, kind='1D'):
		self.stored_transitions += 1
		offset = np.zeros(len(pred), dtype=DTYPE) #updated segments
		for i, row in enumerate(pred) :
			offset[i] = self.update_row(data, row, kind)
		return offset	


##================================= PREDICT ===========================================


	def calc_similarity(self, data, fun=np.sum, ret='1D') :
		# olap / spa_nbits
		self.sim = np.isin(self.segs, data).sum(axis=2).T / float(self.spa_nbits)
		self.sim[ self.sim < self.match_thresh ] = 0 #filter noise

		# print filter_zeros(self.sim)	#only non-zero rows

		#default winner-take all, no-adjustments
		self.rows = fun(self.sim, axis=1).argsort()[::-1][:self.spa_nbits].astype(DTYPE)
		if ret == '1D' : return self.rows
		#2D case ...
		cols = self.sim[self.rows].argmax(axis=1)
		#adjust: row-col =2=> bit indexes
		return np.sort(self.rows * self.nsegments + cols)
	

	def predict(self, data, ret='1D') :
		self.winners = self.calc_similarity(data, ret=ret) # winner-take all
		if ret == '1D' : return iSDP(vsize=self.vsize, vsn=self.winners)
		return i2DSDP(vsize=self.vsize2D, vsn=self.winners, nrows4col=self.nsegments)


##================================= UTILS ===========================================

	def nil2zero(self, data) : #convert 65535 => 0
		if np.all(data == NIL) : return np.zeros(len(data), dtype=DTYPE) 
		return data	

	def __repr__(self): return self.mem_map(seg=0)

	#get a row and segments as SDP
	def r2i2(self, segs, row) :
		return i2DSDP(vsize=self.vsize2D, vsn=self.segs[segs,row,:], nrows4col=self.nsegments)
	row2i2dsdp = r2i2

	def has_dup(self, segs, row) :
		item = self.segs[segs,row,:]
		self.sim = np.isin(self.segs, item).sum(axis=2).T / float(self.spa_nbits)
		return np.where(self.sim == 1)

	def is_dup_row(self, row) :
		for seg in xrange(self.nsegments) :
			item = self.segs[seg,row,:]
			self.sim = np.isin(self.segs, item).sum(axis=2).T / float(self.spa_nbits)
			if self.sim.max() > 0.99 : return True
		return False

	def mem_map(self, segs, lex=None, filter_sym=None, fun=lambda i:i[0], width=4, kind='1D'):
		s = ''; nrows = 0;
		for row in xrange(self.segs.shape[1]) : #rows
			line = ''
			if ~ np.all(self.segs[segs,row,0] == NIL) : 
				line = "%04d :" % (row)
				nrows += 1
				keep = False
				for i, item in enumerate(self.segs[segs,row,:]) :
					sym, first = "  *  ", ""
					if lex is not None and ~np.any(item == NIL) : 
						val = iSDP(vsize=self.vsize, vsn=item)
						if kind == '2D' : val = i2DSDP(vsize=self.vsize2D, vsn=item, nrows4col=self.nsegments).as1d
						sym = lex.bm(val) 
						first = "%04d" % (int(fun(item)),)
						if (filter_sym and sym == filter_sym) or not filter_sym : keep = True
					line +=  " %s%s" % (sym,first)

				if keep : s += line + "\n"	
		return s + "\n%s\n" % nrows
				

	#get back either the segments rows lex.bm of the data-binary or of the predicted value
	# .. where ONE bits is used as index
	# f.e.: lex_row(x.a) => prints segs-lex-best-match a-rows
	# lex_row(x.b,predicted=True) => whichever is the predicted SDP segs-bm rows
	def lex_row(self, data, lex, predicted=False, rng=6):
		if predicted :
			pred = self.predict(data)
			print "predicted: ", lex.bm(pred)
			rows = self.winners
		else :
			rows = data
		rv = []
		for row in rows[:rng] :
			for seg in self.segs :
				best = lex.bt(sdp.resize(seg[row,:].bmap, lex.nrows, lex.sparsity), ret='both')
			 	rv.append( [row, [list(x) for x in best ] ] )
		return rv

	def used(self):
		nil_tf = (self.segs[:,:,0] == NIL).sum(axis=0)
		used_segs4rows = self.nsegments - nil_tf
		non_zero = used_segs4rows[used_segs4rows > 0]
		if len(non_zero) == 0 : return 0,0,0
		used_segs = non_zero.mean()
		#at least one used seg
		used_rows = non_zero.size 
		full = (nil_tf == 0).sum()
		return used_rows, used_segs, full

	@property
	def info(self):

		used = (self.segs[:,:,0] != NIL).sum()
		free = (self.segs[:,:,0] == NIL).sum()
		total = used + free
		percent_used = used/float(total)
		mb = self.segs.nbytes / float(1024*1024)

		used_rows, used_segs, full = self.used() 
		tcapacity = self.nsegments/self.spa

		return """
Seg-Shape (segs,vsize,bits) : %s, Segments : %s
Sparsity/bits: %s/%s
Match-thresh: %s, vsize: %s
used: rows: %s, full:%s, segs:%.2f, stored-pat: %s
      cells: %d%% (%s of %s)
Uniq patterns Capacity: ~%d patterns
Mem: %.2f MB, pat/mb:%.2f
		""" % (self.shape, self.nsegments, self.spa, self.spa_nbits,
				self.match_thresh, self.vsize,
				used_rows, full,used_segs, self.stored_transitions,
				percent_used*100,used, total,
				tcapacity, 
				mb, tcapacity/mb,)





##========================================= MODULE routines ===========================================

	def outs(self): return {'type':iSDP, 'vsize':self.vsize[1], 'spa_nbits': self.spa_nbits[1]}

	def check(self, prev_module) :
		info = prev_module.outs()
		if info['type'] != iSDP : return (True, "expecting iSDP, got %s" % info['type'])
		if self.vsize != info['vsize'] : return (True, "vsize: mismatch %s => %s" % (self.vsize, info['vsize']))
		if self.spa_nbits != info['spa_nbits'] : return (True, "spa_nbits: mismatch %s => %s" % (self.spa_nbits, info['spa_nbits']))
		return (False, "no err")

	def process(self, data, learn=True): 
		if learn : self.learn(data[0], data[1])
		else: self.predict(data, ret='1D')
