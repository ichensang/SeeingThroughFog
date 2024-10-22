import numpy as np
from matplotlib.pyplot import *
from fuzzylab2 import *
import time



def fuzzy_dynamic(inputdynamic, inputstd):
	fis = sugfis()
	fis.addInput([0, 3], Name = 'dynamic_range')
	fis.addMF('dynamic_range', 'trapmf',[0, 0, 1.7, 1.8], Name = 'Too_small')
	fis.addMF('dynamic_range', 'trapmf', [2, 2.1, 3, 3], Name = 'Good')
	fis.addMF('dynamic_range', 'gaussmf', [0.06, 1.9], Name = 'Little_small')
	#plotmf(fis,'input',0)

	fis.addInput([0, 100], Name = 'standard')
	fis.addMF('standard', 'trapmf',[0, 0, 25, 35], Name = 'Good')
	fis.addMF('standard', 'trapmf', [30, 40, 100, 100], Name = 'Too_big')

	fis.addOutput([-50, 100], Name = 'output1')
	fis.addMF('output1', 'trimf',[-10, 0, 10], Name = 'Zero')
	fis.addMF('output1', 'trimf',[20, 30, 40], Name = 'Add_little')
	fis.addMF('output1', 'trimf',[50, 60, 70], Name = 'Add_some')
	fis.addMF('output1', 'trimf',[-40, -30, -20], Name = 'Minus_little')
	#plotmf(fis, 'output', 0)
	ruleList = [[1, -1, 1, 1,1], [2 ,-1 ,3, 1,1], [0,1,3, 1,1], [-1 ,1, 3, 1,1], [0,0, 1, 1,1]]
	fis.addRule(ruleList)

	#std = inputstd

	#plotcontrol = evalfis(fis, [dL,dF,dR])
	pecontrol = evalfis(fis, [inputdynamic, inputstd])
	#print('output = ', plotcontrol)

	return pecontrol

def fuzzy_canny(inputline):
	#fis = sugfis()
	fis = mamfis()
	fis.addInput([0, 35000], Name = 'Line_number')
	fis.addMF('Line_number', 'trapmf',[0, 0, 38000, 42000], Name = 'Too_few')
	fis.addMF('Line_number', 'gaussmf', [1000, 48000], Name = 'Good')
	fis.addMF('Line_number', 'trapmf',[52000, 58000, 70000, 70000], Name = 'Too_many') #40000 40000
	fis.addMF('Line_number', 'gaussmf', [1000, 44000], Name = 'Few')
	fis.addMF('Line_number', 'gaussmf', [1000, 52000], Name = 'Many')
	#plotmf(fis,'input',0)

	fis.addOutput([-6, 6], Name = 'output1')
	#fis.addMF('output1', 'trimf',[-2.5, -2, -1.5], Name = 'Minus3')
	fis.addMF('output1', 'trimf',[-1.5, -1, -0.5], Name = 'Minus_Some')
	fis.addMF('output1', 'trimf',[-1, -0.5, 0], Name = 'Minus_Little')
	fis.addMF('output1', 'trimf',[-0.5, 0, 0.5], Name = 'Zero')
	fis.addMF('output1', 'trimf',[0, 0.5, 1], Name = 'Add_Little')
	fis.addMF('output1', 'trimf',[0.5, 1, 1.5], Name = 'Add_Some')
	#fis.addMF('output1', 'trimf',[3.5, 4, 4.5], Name = 'Add2') #1.5 2 2.5
	#plotmf(fis,'output',0)
	ruleList = [[0,0,1,1], [1,2,1,1], [2,4,1,1], [3,3,1,1], [4,1,1,1]]
	fis.addRule(ruleList)


	#plotcontrol = evalfis(fis, [dL,dF,dR])
	edgecontrol = evalfis(fis, [inputline])
	#print('output = ', plotcontrol)

	return edgecontrol

def fuzzy_linenumber(inputstd):
	fis = sugfis()
	fis.addInput([0, 2], Name = 'Line_std')
	#fis.addMF('Line_std', 'trapmf',[0, 0, 0.2, 0.3], Name = 'Good')
	fis.addMF('Line_std', 'trapmf',[0, 0, 0.5, 0.7], Name = 'Good')
	#fis.addMF('Line_std', 'trapmf', [0.5, 0.6, 2, 2], Name = 'Too_big')
	fis.addMF('Line_std', 'trapmf', [1.5, 1.6, 2, 2], Name = 'Too_big')
	#fis.addMF('Line_std', 'gaussmf', [0.05, 0.4], Name = 'A_little_big')
	fis.addMF('Line_std', 'gaussmf', [0.1, 0.9], Name = 'A_little_big')
	#plotmf(fis,'input',0)

	fis.addOutput([-2, 2], Name = 'output1')
	fis.addMF('output1', 'trimf',[-1.8, -1.5, -1.2], Name = 'Minus_some')
	fis.addMF('output1', 'trimf',[-1, -0.85, -0.7], Name = 'Minus_little')
	fis.addMF('output1', 'trimf',[-0.3, 0, 0.3], Name = 'Zero')
	fis.addMF('output1', 'trimf',[0.7, 0.85, 1], Name = 'Add_little')
	fis.addMF('output1', 'trimf',[1.2, 1.5, 1.8], Name = 'Add_some')
	#fis.addMF('output1', 'constant',-1.5, Name = 'Minus_some')
	#fis.addMF('output1', 'constant',-0.85, Name = 'Minus_little')
	#fis.addMF('output1', 'constant',0, Name = 'Zero')
	#fis.addMF('output1', 'constant',0.85, Name = 'Add_little')
	#fis.addMF('output1', 'constant',1.5, Name = 'Add_some')
	#plotmf(fis, 'output', 0)
	ruleList = [[0,3,1,1], [1,0,1,1], [2,1,1,1]]
	fis.addRule(ruleList)

	std = inputstd

	#plotcontrol = evalfis(fis, [dL,dF,dR])
	plotcontrol = evalfis(fis, [std])
	#print('output = ', plotcontrol)

	return plotcontrol

def fuzzy_fusion(input_array):
	fis = mamfis()
	#fis = sugfis()
	fis.addInput([0, 40], Name = 'distance')
	fis.addMF('distance', 'trapmf',[0, 0, 3, 8], Name = 'Near')
	fis.addMF('distance', 'trapmf', [15, 25, 40, 40], Name = 'Far') #12 17 40 40
	fis.addMF('distance', 'gaussmf', [2, 12], Name = 'Mid')
	#plotmf(fis,'input',0)

	fis.addInput([0, 1], Name = 'feature')
	fis.addMF('feature', 'trapmf',[0, 0, 0.1, 0.2], Name = 'Small') #0 0 0.5 0.7
	fis.addMF('feature', 'trapmf', [0.3, 0.4, 1, 1], Name = 'Big') #0.5 0.7 1 1
	#plotmf(fis, 'input', 1)

	fis.addOutput([0, 1], Name = 'output1')
	fis.addMF('output1', 'trapmf',[0.8, 0.9, 1, 1], Name = 'Yes') #0.4 0.6 1 1
	fis.addMF('output1', 'trapmf',[0, 0, 0.4, 0.5], Name = 'No')
	#plotmf(fis, 'output', 0)
	ruleList = [[1, -1, 1, 1,1], [0 ,1 ,0, 1,1], [0 ,0 ,1, 1,1], [2 ,1 ,0, 1,1], [2 ,0 ,1, 1,1]]
	fis.addRule(ruleList)

	#std = inputstd

	#plotcontrol = evalfis(fis, [dL,dF,dR])
	fusion_result = evalfis(fis, input_array)
	#print('output = ', plotcontrol)

	return fusion_result

def fuzzy_fusion_single(feature):
	fis = mamfis()
	#fis = sugfis()
	fis.addInput([0, 1], Name = 'feature')
	fis.addMF('feature', 'trapmf',[0, 0, 0.6, 0.8], Name = 'Small') #0 0 0.7 0.8
	fis.addMF('feature', 'trapmf', [0.78, 0.95, 1, 1], Name = 'Big') #0.8 0.9 1 1
	#plotmf(fis, 'input', 0)

	fis.addOutput([0, 1], Name = 'output1')
	fis.addMF('output1', 'trapmf',[0.4, 0.6, 1, 1], Name = 'Yes')
	fis.addMF('output1', 'trapmf',[0, 0, 0.2, 0.4], Name = 'No')
	#plotmf(fis, 'output', 0)
	ruleList = [[0, 1, 1,1], [1 ,0, 1,1]] #Antecedents, Consequent, Weight, Connection(1=And, 0=Or)
	fis.addRule(ruleList)

	#std = inputstd

	#plotcontrol = evalfis(fis, [dL,dF,dR])
	fusion_result = evalfis(fis, feature)
	#print('output = ', plotcontrol)

	return fusion_result

#print(fuzzy_dynamic(1,100))
#a = fuzzy_fusion([10, 20], [0, 1])
#print(trapmf([2.0,3.0, 10], [1,5,10, 11]))
#a = fuzzy_fusion_single(np.asarray([[0.2], [0.4], [0.6]]))
#input_array = np.asarray([[4, 0.3],[11, 0.6], [18, 0.9]])
x = ([20, 0.2])
y = np.asarray([[1.0], [1.0]])
#print(y.shape)
#x = np.random.rand(10000, 2)
#start = time.time()
#a = fuzzy_fusion(x)
#b = fuzzy_fusion_single(y)
#end = time.time()
#print(end-start)
#c = fuzzy_canny(15000)
#vfunc = np.vectorize(fuzzy_fusion)
#a = vfunc([10,20,30], [0, 1, 2])
#a = fuzzy_fusion(matrix)
#print(a)
#print(b)