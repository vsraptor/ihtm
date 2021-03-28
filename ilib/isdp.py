import os, sys
# sys.path.extend(['lib', '../lib', 'ilib', '../ilib'])

# from sdp import sdp 
import numpy as np
import warnings
import functools
import inspect
import pprint
from iutils import *


#Global defaults
NBITS = 1000 #size
SPA  = 0.02 #sparsity
SPA_NBITS = int(SPA * NBITS)
SIM_THRESH = 0.12
DTYPE = np.uint16
#no-data characters
NIL = np.uint16(-1) #what is clean memory for np.isin() to work
ZERO = 0


class isdp(object):


	@staticmethod
	def null(vsize): return iSDP(vsize, np.zeros(0,dtype=DTYPE))
	@staticmethod
	def full(vsize, spa_nbits, start=0, step=2):
		return iSDP(vsize=vsize, vsn=np.arange(start, start+spa_nbits*step, step, dtype=DTYPE))

	@staticmethod #convert from numpy-binary 0|1 to iSDP
	def np012sdp(ary): return iSDP(vsize=ary.size, vsn=ary.nonzero()[0])

	@staticmethod
	def rand(vsize, spaOnbits, nrows4col=None, ret='iSDP'):
		if isinstance(spaOnbits,float) : spaOnbits = int(vsize * spaOnbits)
		assert 0 < spaOnbits < vsize
		rv = np.sort( np.random.choice( np.arange(0,vsize, dtype=DTYPE), spaOnbits, replace=False ) )
		if ret == 'i2DSDP' : return i2DSDP(vsize=vsize, vsn=rv, nrows4col=nrows4col)
		return rv # iSDP(vsize=vsize, vsn=rv)

	@staticmethod
	def roll(data, roll_nbits, check=False) :
		if roll_nbits == 0 : return data
		val = np.add(data, roll_nbits)
		if check and ~ np.any(np.equal(val,0)) : raise Exception("roll: iSDP cant have index ZERO : %s" % val)
		if roll_nbits > 0 : val[val > val.vsize] -= val.vsize
		else : val[val < 0 ] += val.vsize
		return val

	@staticmethod
	def union(tpl, vsize=None) :
		val = np.unique(np.concatenate(tpl))
		if isinstance(tpl[0], i2DSDP) :
			return i2DSDP(vsize=tpl[0].vsize, nrows4col=tpl[0].nrows4col, vsn=val)
		if isinstance(tpl[0], iSDP) :
			return iSDP(vsize=tpl[0].vsize, vsn=val)
		if isinstance(tpl[0], np.ndarray) : 
			if vsize is None : raise Exception("union: when numpy 'vsize' arg required")
			return iSDP(vsize=vsize, vsn=val)

	@staticmethod # switch OFF some of the ON bits
	def sparseit(ixs, keep_cnt):
		return ixs[np.random.choice(len(ixs), keep_cnt, replace=False)]

	@staticmethod
	def resize(sdpi,change):
		scale = 1
		if isinstance(change,int) : scale = sdpi.vsize + change
		else : scale += change
		sdpi *= scale
		sdpi.vsize *= scale
		return sdpi

	@staticmethod #CDT using random selection
	def thin(union, new_spaOnbits):
		assert np.all(union < union.vsize), "%s < %s" % (union, union.vsize)

		_, spa_nbits = spaORnbits(union.vsize, new_spaOnbits)
		 # int(union.vsize * new_spaOnbits) if isinstance(new_spaOnbits, float) else new_spaOnbits
		assert union.size >= spa_nbits, "Low bit count: union:%s > spa_nbits:%s !!" % (union.size,spa_nbits) 

		if union.size == spa_nbits : 
			warnings.warn("Sparsity of the Union equals Output sparsity !!")
			return union

		return iSDP(vsize=union.vsize, vsn=np.random.choice(union, size=spa_nbits,replace=False))


	#pick randomly proportional % of bits : Context Dependent Thinning
	@staticmethod #thinning, similarity-binding, normalization
	def cdt(union, new_spaOnbits):
		assert np.all(union < union.vsize), "%s < %s" % (union, union.vsize)

		spa_nbits = int(union.vsize * new_spaOnbits) if isinstance(new_spaOnbits, float) else new_spaOnbits
		assert union.size >= spa_nbits, "Low bit count: union:%s > spa_nbits:%s !!" % (union.size,spa_nbits) 

		if union.size == spa_nbits : 
			warnings.warn("Sparsity of the Union equals Output sparsity !!")
			return union

		thinning = union.copy() #permute-vector
		if spa_nbits > union.size :
			warnings.warn("Requested sparsity:%s must be smaller than %s of the union of the src vectors" % (spa_nbits,union.size))
			spa_nbits = union.size

		i = 0
		rv = np.zeros(0, dtype=DTYPE)
		#stop when there is no longer zeros
		while len(rv) < spa_nbits: #pick bits
			i += 1
			if i > 200 : warnings.warn("Too many loops : %s: %s" % (len(rv), rv))

			roll_nbits = np.random.randint(0, spa_nbits)
			rolled = isdp.roll(thinning, roll_nbits=roll_nbits)
			conj = np.intersect1d(union, rolled, assume_unique=True)
			rv = isdp.union((conj, rv), vsize=union.vsize)

		#if more than requested bits are ON, switch OFF some of them
		# ... happens when you unionize many vectors with small target sparsity
		if len(rv) > spa_nbits : rv = isdp.sparseit(rv, spa_nbits) #!fixme : add jiggle, may be! +/- very-small-%bits

		return iSDP(vsize=union.vsize, vsn=rv) 

	@staticmethod
	def olap(left, right) :	return np.intersect1d(left, right).size
	@staticmethod
	def sim(left, right) : return isdp.olap(left,right) / float(len(left))
	@staticmethod
	def is_sim(left, right, thresh=SIM_THRESH) : return isdp.sim(left, right) > thresh

	@staticmethod
	def permute(sdpi, pmx) : pass #!todo

	@staticmethod	#flip bits
	def flip(sdpi, nbits):
		all_bits = set(xrange(0, sdpi.vsize))
		#pick idx not in the sources
		new_nums = np.random.choice(list(all_bits.difference(sdpi)), nbits, replace=False)
		new = sdpi.copy()
		new[np.random.randint(0,sdpi.size,nbits)] = new_nums #substitute
		return np.sort(new)

	#like union but adjusts the vsize and every item is shifted by prev elem vsize
	@staticmethod
	def concat(lst):
		#collect the iSDP sizes
		sizes = np.array([ item.vsize for item in lst ]) 
		start = np.cumsum(sizes) - lst[0].vsize # move ixs 
		new_vsize = start[-1] + lst[0].vsize
		#increment and join
		new = [ start[i] + item for i,item in enumerate(lst) ]
		sdpi = iSDP(vsize=new_vsize, vsn=new)
		assert np.any(sdpi < new_vsize), "concat: idxs can not be bigger than the size"
		return sdpi


class iSDP(np.ndarray):
	__slots__ = "vsize", "spa", "spa_nbits", "buf", "sym"

	def __new__(cls, vsize=NBITS, vsn=SPA, sym=None) : #value|spa|spa_nbits

		buf, spa, spa_nbits = None, None, None

		# print pprint.PrettyPrinter(indent=3).pprint( inspect.stack() )
		if isinstance(vsn, np.ndarray) :
			buf = np.sort(vsn.astype(DTYPE))
			spa_nbits = vsn.size
			spa = spa_nbits / float(vsize)
		elif isinstance(vsn, list) :
			buf = np.sort(np.hstack(vsn))
			spa_nbits = buf.size
			spa = spa_nbits / float(vsize)
		elif isinstance(vsn, (iSDP,i2DSDP)) :
			buf = np.sort(np.asarray(vsn))
			spa_nbits = vsn.spa_nbits
			spa = spa_nbits / float(vsize)
		elif isinstance(vsn,float) :
			spa = vsn
			spa_nbits = int(vsize * spa)
			buf = isdp.rand(vsize=vsize, spaOnbits=spa_nbits)
		elif isinstance(vsn, int) :
			spa_nbits = vsn
			spa = spa_nbits/float(vsize)
			buf = isdp.rand(vsize=vsize, spaOnbits=spa_nbits)
		else :
			raise Exception("iSDP: Wrong parameters ...")

		obj = super(iSDP, cls).__new__(cls, shape=(spa_nbits,), buffer=buf, dtype=DTYPE)
		obj.vsize = vsize
		obj.spa = spa
		obj.spa_nbits = spa_nbits
		obj.sym = sym

		return obj


	def __array_finalize__(self, obj):
		if obj is None: return
		self.vsize = getattr(obj, 'vsize', None)
		self.spa = getattr(obj, 'spa', None)
		self.spa_nbits = getattr(obj, 'spa_nbits', None)
		self.sym =  getattr(obj, 'sym', None)

	def __array_wrap__(self, out_arr, context=None):
		return super(iSDP, self).__array_wrap__(out_arr, context)

	# def __array_function__(self, func, types, args, kwargs):
	# 	if func == np.concatenate:
	# 		< do stuff here for concatenating your class >
	# 		return < result of stuff done of type MyClass>
	# 	else:
	# 		return NotImplemented

	@property #return as numpy array
	def asnp(self): return np.asarray(self)
	@property # as numpy array of 0s and 1s
	def as01(self):
		rv = np.zeros(self.vsize, dtype=np.uint8)
		rv[self] = 1
		return rv

	def copy(self, order='K'): return type(self)(vsize=self.vsize, vsn=np.asarray(self))
	def clear(self): self[:] = 0

	# @property
	# def i2s(self):
	# 	rv = SDP(self.vsize)
	# 	sdp.set_by_ixs(rv,self)
	# 	return rv

	@property
	def info(self):
		return "Size: %s, spa/bits: %s/%s\n" % (self.vsize, self.spa, self.spa_nbits)

	def __repr__(self) : 
		ncols =  int(os.getenv('COLUMNS', 80)) - 10
		rv, line, tmp = '','', ''
		for s in self :
			tmp = "%s:" % s
			if (len(line) + len(tmp)) < ncols : line += tmp
			else : 
				rv += line[:-1] + "\n"
				line = tmp
		if len(rv) == 0 : return line[:-1]
		return rv + line[:-1] + "\n"

	def __str__(self): return self.__repr__()

	def __mul__(self, rhs) : 
		assert self.vsize == rhs.vsize, "Size mistmatch"
		union = isdp.union((self, rhs))
		return isdp.thin(union, new_spaOnbits=self.spa_nbits)
	def __or__(self, rhs) : 
		assert self.vsize == rhs.vsize, "Size mistmatch"
		return isdp.union((self, rhs))
	def __xor__(self, rhs) : 
		assert self.vsize == rhs.vsize, "Size mistmatch"
		return iSDP(vsize=self.vsize, vsn=np.setxor1d(self, rhs, assume_unique=False))
	def __add__(self, rhs) : 
		return isdp.concat((self, rhs))
	def __truediv__(self, rhs) : 
		assert self.vsize == rhs.vsize, "Size mistmatch"
		return isdp.olap(self, rhs)
	def __floordiv__(self, rhs) : 
		assert self.vsize == rhs.vsize, "Size mistmatch"
		return isdp.sim(self, rhs)
	def __rshift__(self, rbits) : return isdp.roll(self, roll_nbits=rbits)
	def __lshift__(self, spaOnbits) : # << is thin to size 
		_, spa_nbits = spaORnbits(self.vsize, spaOnbits)
		return isdp.thin(self, new_spaOnbits=spa_nbits)

	def __contains__(self, rhs) : 
		assert self.vsize == rhs.vsize, "Size mistmatch"
		return isdp.is_sim(self, rhs)

	def __eq__(self, rhs) : 
		assert self.vsize == rhs.vsize, "Size mistmatch"
		return np.all(np.equal(self, rhs).asnp)
	def __ne__(self, rhs) : 
		assert self.vsize == rhs.vsize, "Size mistmatch"
		return ~np.all(np.equal(self, rhs).asnp)


class i2DSDP(iSDP) : # used for 2D buffers

	def __new__(cls, *args, **kwargs) :
		assert 'nrows4col' in kwargs, "Provide number of rows per column : nrows4col"
		nrows4col = kwargs['nrows4col']
		del kwargs['nrows4col']
		obj = super(i2DSDP, cls).__new__(cls, *args, **kwargs)
		obj.nrows4col = nrows4col
		return obj

	def __array_wrap__(self, out_arr, context=None):
		return super(i2DSDP, self).__array_wrap__(out_arr, context)

	@property
	def as1d(self):#all cols that have at least one bit set
		rv = np.divide(self,self.nrows4col)
		return iSDP(vsize=int(self.vsize/self.nrows4col), vsn=np.asarray(rv)) 

	def copy(self, order='K'): return type(self)(vsize=self.vsize, vsn=np.asarray(self), nrows4col=self.nrows4col)

	def __div__(self, rhs) : 
		if isinstance(rhs, iSDP) : return iSDP(vsize=self.vsize, vsn=self.to_cols).olap(rhs)
		return self.olap(rhs)
	def __floordiv__(self, rhs) : 
		if isinstance(rhs, iSDP) : return iSDP(vsize=self.vsize, vsn=self.to_cols).sim(rhs)
		return self.sim(rhs)
