
# This class is used as a base for the framework
# It establishes the interface for providing information
# needed to connect modules

# Here is the basic informational structure 
#  { input : { shape : (2000,), spa: 0.02, spa_nbits : 40 }, output : {...} }

class Module(object): 

	#provides information about input arguments	
	def ins(self): raise NotImplementedError
	def outs(self): raise NotImplementedError	
	def check(self, prev_module): raise NotImplementedError	

	def no_error() : return (False, "no error")

	def check_type(self, info, xtype):
		if info['type'] != xtype : return (True, "expecting %s, got %s" % (xtype,info['type']))
		return (False,)

	def check_vsize(self, info):
		if self.vsize != info['vsize'] : return (True, "vsize: mismatch %s => %s" % (self.vsize, info['vsize']))
		return (False,)

	def check_spa_nbits(self, info):
		if self.spa_nbits != info['spa_nbits'] : return (True, "spa_nbits: mismatch %s => %s" % (self.spa_nbits, info['spa_nbits']))
		return (False,)

	def check3(self, info):
		rv = self.check_type(info)
		if rv[0] is True : return rv 
		rv = self.check_vsize(info)
		if rv[0] is True : return rv
		rv = self.check_spa_nbits(info)
		if rv[0] is True : return rv
		return self.no_error()
