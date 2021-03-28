from itemporal_memory import *
from iutils import *
from collections import OrderedDict
from corpus import Corpus

ri = np.random.randint
rc = np.random.choice

class LexTM(iTemporalMemory):

	# def __init__(self, *args, **kwargs) :
	# 	assert 'lex' in kwargs, "lexicon required"
	# 	self.lex = kwargs['lex']
	# 	del kwargs['lex']
	# 	super(LexTM, self).__init__(*args, **kwargs)
	# 	self.word_count = 0

	def set_lex(self, lex):
		self.lex = lex
		self.lex.az()
		self.lex.syms()
		self.word_count = 0

		self.corpus_data = None
		self.corpus = Corpus()

	def set_encoder(self, encoder) :
		self.encoder = encoder

	#converts a sequence to integer-idxs, so that we can calc ENTROPY
	def token2ints(self, seq):	
		ttype, seq = detect_tokens(seq)
		if ttype == INT : return seq
		ids = {e:i for i,e in enumerate(set(seq))}
		return [ ids[tok] for tok in seq ]

	#generates the CHARACTER, WORD or INT tokens for sequence generation
	def gen_tokens(self, ntokens=5, ttype=CHAR, int_range=(0,100), start=None):
		if ttype == CHAR :
			assert ntokens <= 26, "alphabet is 26 characters : %s" % ntokens
			tokens = rc(list(string.ascii_lowercase), ntokens, replace=False)
		elif ttype == WORD :
			tokens = rand_words(ntokens)
		elif ttype == INT and int_range is not None :
			assert int_range[1] - int_range[0] >= ntokens, "num of tokens must be bigger than the int range : %s" % ntokens
			tokens = rc(xrange(int_range[0], int_range[1]), ntokens, replace=False)
		
		return tokens

	#generate random sequence
	def random_seq(self, ntokens=5, length=50, ttype=CHAR, int_range=(0,100)) :

		tokens = self.gen_tokens(ntokens=ntokens, ttype=ttype, int_range=int_range)
		if length == ntokens : return tokens
		seq = rc(tokens, length, replace=True)
		return list(seq)

	#given tokens and seq template return dict of pattern
	def gen_patterns(self, seq_tpl, ntokens, pat_size, ttype, int_range):

		tokens = self.gen_tokens(ntokens=ntokens, ttype=ttype, int_range=int_range)

		pats = OrderedDict.fromkeys(seq_tpl) #key=None dict
		for p in pats.keys() :
			pats[p] = list(rc(tokens, ri(*pat_size), replace=False))

		return pats

	#return sequence build from PATTERNS
	def gen_seq(self, 
		seq_tpl='ABAC', # sub-pattern definition, '*' for random
		ntokens=7, # number or xrange of tokens used to generate sub-pat/seq
		pat_size=(2,5), # sub-pattern sizes
		ttype=CHAR, # tokens type : CHAR, WORDS, INT 
		pats = None, # "overrides" the prev args
		length=50, # sequence lengths 
		start = None,
		int_range=(0,100) # when INT the range of possible values
	):
		assert ntokens > pat_size[1]

		#!fixme pats requires the original seq_tpl !!!

		if seq_tpl == '*' :
			return self.random_seq(ntokens=ntokens, length=length, ttype=ttype, int_range=int_range), 'rand'
	
		if ttype == CORPUS : #extract sequence of words from a corpus
			if self.corpus_data is None : 
				self.corpus.process()
				self.corpus_data = self.corpus.tokens
			size = len(self.corpus_data)
			if start is None : start = ri(0, size)
			end = start + length
			if end > size : end = size
			return self.corpus_data[start : end], 'corpus'

		if pats is None :
			pats = self.gen_patterns(seq_tpl=seq_tpl, ntokens=ntokens, pat_size=pat_size, ttype=ttype, int_range=int_range)

		#build a lbl for display purposes
		label = ''.join([k + str(len(pats[k])) for k in seq_tpl ])

		#tpl =pats=> sub_seq and flatten
		sub_seq = sum([ pats[p] for p in seq_tpl ], [] ) 

		if len(sub_seq) >= length : return sub_seq[:length], '-'+label #warn!

		repetition = length/len(sub_seq)
		rest = length % len(sub_seq)
		seq = sub_seq * repetition + sub_seq[:rest]
		return seq, label

 			

	#convert a sym-sequence to sdp-sequence
	#acccepts: numbers, characters, words
	def seq2sdp(self, seq,  sep=','):

		ttype, seq = detect_tokens(seq)

		sdp_seq = []
		for i, item in enumerate(seq) :
			#if number use encoder to encode
			if ttype == INT and hasattr(self, 'encoder') :
				sdp = self.encoder.encode(int(item))
				sdp_seq.append(sdp)
			elif ttype in [CHAR, WORD] and item in self.lex :
				sdpi = self.lex.get(item)
				sdp_seq.append(sdpi)
			else :
				print("sequence unknown item:%s (ix:%s)" % (item,i))

		return sdp_seq, seq	


	#convert SDP seq to lex-syms
	def resolve_seq(self, sdp_seq, ttype, sep='', as1d=True):

		syms = []
		for sdpi in sdp_seq :
			if ttype in [CHAR, WORD,] :
				sym = self.lex.bm(sdpi.as1d) if isinstance(sdpi, i2DSDP) else self.lex.bm(sdpi)
			elif ttype == INT :
				sym = self.encoder.decode(sdpi)
			else : raise Exception("resolve : Unknown token type")				
			syms.append(sym) 
	
		if ttype == CHAR : return sep.join(syms)
		return syms


	#store state, play SDP sequence, restore state
	def train_sequence(self, seq, learn=True, repeat=1, sep=','):
		#make sure it is sequence of SDP's
		if not isinstance(seq[0],(iSDP,i2DSDP)) and self.lex is not None :
			sdp_seq, seq = self.seq2sdp(seq)
		else : sdp_seq = seq

		self.store()
		self.prep()
		predictions = []
		for r in xrange(repeat) :
			for i,item in enumerate(sdp_seq) :
				# print "> ", seq[i]
				# if lex is not None and isinstance(item, str) : item = lex.get(item)
				pred = self.step(item, learn=learn)
				if r == 0 : predictions.append(pred)
		self.restore()
		return predictions

	#word list OR string (char-sequences)
	def train_tokens(self, tokens, repeat=1, sep=','):

		#figures the tokens type
		ttype, tokens = detect_tokens(tokens, sep)

		if ttype in [CHAR, WORD] :#first, add the tokens to the lexicon
			toks = [  self.lex.vivify(t) for t in tokens ]
			for r in xrange(repeat):
				res = self.train_sequence(seq=tokens, repeat=1, learn=True)
				# res = self.resolve_seq( ts, ttype=ttype )

		elif ttype == CHAR :
			for r in xrange(repeat):
				for token in tokens :
					res = self.train_sequence(seq=token, repeat=1, learn=True)

		elif ttype == INT :
			for r in xrange(repeat):
				res = self.train_sequence(seq=tokens, repeat=1, learn=True)

		else :
			raise Exception("Unknown tokens type : %s" % ttype)

		return res		

	#templated sequence, where tpl-sym is replaced with prediction
	def templated_seq(self, tpl, tpl_sym='.', sep=',', fwd_steps=0) :
	
		tpl += tpl_sym * fwd_steps
		ttype, tpl = detect_tokens(tpl)
		assert tpl[0] != tpl_sym, "prediction need initial symbol"

		print "> tpleted seq %s : %s... " % (ttype, tpl[:10])

		self.store()
		self.prep()
		predictions = []
		prev = None
		for i in tpl : #loop the template
			item = i

			#use the prev-prediction to substitute the template character
			if item == tpl_sym : 
				if ttype == INT: item = self.encoder.decode(prev.as1d)
				else:	item = self.lex.bm(prev.as1d)

			#still string ? then convert sym to iSDP
			if isinstance(item, int) : item = self.encoder.encode(item)
			if isinstance(item, str) : item = self.lex.get(item)
			#do the prediction
			pred = self.step(item, learn=False, lex=self.lex)
			# print " > %s:%s => %s" % (i, abc, self.lex.bm(pred.as1d))
			prev = pred
			predictions.append(pred)
		self.restore()	

		#fix the pred using the template
		corrected = tpl[0] if ttype == CHAR else [tpl[0]]
		predicted = tpl[0] if ttype == CHAR else [tpl[0]]
		#except the last predicted which is outside the range
		for i,s in enumerate(predictions[:-1]) :
			if ttype == INT: sym = self.encoder.decode(s.as1d)
			else : sym = self.lex.bm(s.as1d)
			predicted += sym if ttype == CHAR else [sym]
			#if prediction contradicts tpl, use template
			if i+1 < len(tpl) and tpl[i+1] != '.' and sym != tpl[i+1] : 
				print "%s> %s => %s" % (i+1,sym,tpl[i+1])
				sym = tpl[i+1]
			corrected += sym if ttype == CHAR else [sym] 
		return corrected, predicted


	def seq2seq(self, seq, learn=True, repeat=1, sep=','):
		sdp_seq = self.train_sequence(seq,learn=learn,repeat=repeat,sep=sep)
		return self.seq2sdp(sdp_seq)