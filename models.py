class Formulas():
	def __init__(self):
		self.formulas = {}
		self.formulas["polynomial"] = []
		self.formulas["circles"] = []
	def put(self, formula):
		if (type(formula[0])==list):
			self.formulas["circles"].append(formula)
		else:
			self.formulas["polynomial"].append(formula)
	def get(self, catogary, index):
		return self.formulas[catogary][index]	

formulas = Formulas()
formulas.put([1,2])
formulas.put([[[1,1],[-1,-1]],[0.5,0.5]])
# formulas.put([[[12,0],[-12,0]],[4,4]])