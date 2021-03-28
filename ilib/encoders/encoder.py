import os, sys
sys.path.extend(['./', '../', './modules', '../modules'])
from module import Module
from isdp import iSDP
from iutils import *

import numpy as np
import math
# from base import Base
# import utils
from pipe import Pipe, PerPipe

class Encoder(Module):

	@property
	def pencode(self): return PerPipe(self.encode)
	pe = pencode
	@property
	def pdecode(self): return PerPipe(self.decode)
	pd = pdecode

	
	# @property
	# def vsize(self): raise NotImplementedError
	# @property
	# def spa(self): raise NotImplementedError
	# @property
	# def spa_nbits(self): raise NotImplementedError


	def encode(self, data): raise NotImplementedError
	def decode(self, data): raise NotImplementedError


##========================================= MODULE ===========================================

	#Module stuff
	def outs(self): return {'type': iSDP, 'vsize':self.vsize, 'spa_nbits': self.spa_nbits}
	def check(self, prev_module) : return (False, "no error")
	def process(self, data): return self.encode(data)

	
#	def __lt__(self, other):
#		if isinstance(other, (list, BMap2D)) : return self.batch_encode(other)
#		return self.encode(other)
#	def __gt__(self, other): 
#		if isinstance(other, (list, BMap2D)) : return self.batch_decode(other)
#		return self.decode(other)
