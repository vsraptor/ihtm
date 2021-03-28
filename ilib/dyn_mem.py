import numpy as np
from isdp import isdp, NBITS, SPA, SPA_NBITS, DTYPE, NIL

class DynMem:

	def __init__(self, shape=(100,SPA_NBITS), grow_percent=0.5, grow_step=0.8, vsize=NBITS): 
		self.ary = np.zeros(shape, dtype=DTYPE) + NIL
		self.grow_percent = grow_percent
		self.grow_step = grow_step
		self.vsize = vsize
		self.erase()

	@property
	def ix_empty(self) : return self.ix + 1

	def grow(self, rows=None, cols=None) :
		print("+ rows:%s, cols:%s" % (rows,cols))
		if cols : #add cols
			self.ary = np.append( self.ary, np.zeros((self.ary.shape[0], cols), dtype=DTYPE), axis=1 )

		if rows : #add rows
			self.ary = np.append( self.ary, np.zeros((rows, self.ary.shape[1]), dtype=DTYPE), axis=0 )

	def get(self, idx):	return self.ary[ idx, : ]
	def set(self, idx, value): self.ary[idx, :] = value

	def add(self, vec=None):
		ix = None
		poped = False
		#if there are free items, use them instead of allocating new
		if len(self.free) > 0 :
			poped = True
			ix = self.free.pop()
		else:
			self.ix += 1
			ix = self.ix
			if self.ix >= self.ary.shape[0] : 
				self.grow(rows=int(self.grow_percent * self.ary.shape[0])) 
				self.grow_percent *= self.grow_step
				if self.grow_percent < 0.1 : self.grow_percent = 0.1


		if vec is not None :
			self.ary[ ix, : ] = vec
		else :
			self.ary[ ix, : ] = isdp.rand(vsize=self.vsize, spaOnbits=self.ary.shape[1], ret='numpy')

		if poped : return ix, False #old/reused
		return ix, True #new
	
	def erase(self):
		self.ary[:] = 0
		self.ix = -1
		self.free = [] #list of free indexes

	#put the item in the free-list for reuse
	def remove(self, ix):
		self.ary[ix,:] = 0
		self.free.append(ix)

	def add_items(self, lst) :
		for el in lst : self.add(el)

	def __repr__(self):
		return self.mem.__repr__()

	@property
	def info(self):
		s  = "Dynamic Memory :==================\n"
		s += "items:%s, max items:%s, spa-nbits:%s\n" % (self.ix+1, self.ary.shape[0], self.ary.shape[1])
		s += "grow %% : %s, size: %s" % (self.grow_percent, self.vsize)
		return s
