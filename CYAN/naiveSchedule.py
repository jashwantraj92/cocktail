from __future__ import absolute_import, division, print_function
from collections import defaultdict
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants, signature_def_utils_impl
from matplotlib import pyplot as plt
from signal import signal, SIGINT
from sys import exit
import efficientnet.keras as efn
import tensorflow as tf
import numpy as np
import scipy as sp
import os,sys
import math
import time
import random
import collections
import logging
import csv

log = logging.getLogger('spam_application')
log.setLevel(logging.INFO)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
import argparse
#end imports

'''
Terms:

cur_latency:	max latency amont all instances, 
cur_accuracy:	aggreagted aacuracy of all instances, 
cur_cost:		total cost of all instances 

slo_latency, slo_accuracy, slo_cost : constraints provided by user. (FIXME: Need some defaut values for each)

latecy_margin, accuracy_margin, cost_margin: margins provided by user. Defaut value = 0 if not provided

Scaling Priority: User specified. Cost > Accuracy > latency (Default)

README
	Function Lists:
		instace.init					:	initializes base object of class instance
		instance.add_model_to_instance	:	keeps track of all models in an instace and adds models as scale-in is called #can be called instance_model_manager
											returns 1 in case of successful scale in, returns 0 if fail
		scale_up						: 	adds one more instace to the instance list
		
	TODO:
		scale_down_policy				: incase the user changes the SLO in the runtime
'''
###############################################################
# PDFs

matched = 0
weighted_match = 0
not_matched=0
BLmatch = 0
step_matching_pred = 0
total = 0
images = defaultdict(list)
file1 = open('/home/cc/cocktail/CYAN/ground-truth-classes', 'r') 
lines = file1.readlines()
for line in lines:
    label = line.split(" ")[1].strip('\n')
    label = label.strip('\t')
    name = line.split(" ")[0]
    images[name].append(label)
#logging.info(images)
def check_ground_truth(imgcls,imgname):
    global matched,not_matched,BLmatch,total,step_matching_pred,weighted_match
    total +=1
    match=0
    #print("matching ground truth", BLmatch, imgcls, imgname, images[imgname][0])
    for i in range(len(imgcls)):
        if imgcls[i] == images[imgname][0]:
            if i == 0:
                matched+=1
                print(f'majority voting ground truth matched {matched} {imgcls} {imgname} {images[imgname]}')
                step_matching_pred +=1
                #logging.info(f"Prediction accuracy {matched/(total)*100}")       
            elif i ==1:
                BLmatch+=1
                print(f'Baseline ground truth matched {matched} {imgcls} {imgname} {images[imgname]}')
            elif i ==2:
                weighted_match+=1
                print(f'Weighted voting  ground truth matched {matched} {imgcls} {imgname} {images[imgname]}')
   #logging.info(f"Prediction accuracy {BLmatch/(total)*100}")       
            match=1
            #logging.info(f"Prediction accuracy {matched/(matched+not_matched)*100} {BLmatch/(BLmatch+not_matched)*100}")       
    if match==0:
        not_matched+=1
        print(f'ground truth not matched {not_matched} {imgcls} {imgname} {images[imgname]}')
    print(f'Prediction accuracy {matched} {BLmatch} {not_matched} {matched/total*100} {BLmatch/total*100}')       
    #logging.info(f"Prediction accuracy {matched/(matched+not_matched)*100}")       

def pdf_fun(accuracy, verbose):
	pos			=	int(accuracy*100);
	arr			=	np.zeros(10000);
	arr[:pos]	=	1;
	#arr = np.hstack((np.ones(pos), np.zeros(10000 - pos)))
	np.random.shuffle(arr);
	
	if (verbose == 1):
		print(sum(arr)/10000);
	return arr;
MobileNetV1			=	pdf_fun(69.36,0);
MobileNetV2			=	pdf_fun(69.56,0);
InceptionV3			=	pdf_fun(68,0);
ResNet50V2			=	pdf_fun(66,0);
DenseNet201			=	pdf_fun(72.8,0);
DenseNet121			=	pdf_fun(69.95,0);
Xception			=	pdf_fun(69.75,0);
NASNetMobile			=	pdf_fun(71.14,0);
NASNetLarge			=	pdf_fun(76,0);
InceptionResnetV2		=	pdf_fun(73.6,0);
vgg16				=	pdf_fun(66.52,0);
ResNet50			=	pdf_fun(64,0);

"""MobileNetV1			=	pdf_fun(70.40,0);
MobileNetV2			=	pdf_fun(71.30,0);
InceptionV3			=	pdf_fun(77.90,0);
ResNet50V2			=	pdf_fun(74.90,0);
DenseNet201			=	pdf_fun(76.00,0);
DenseNet121			=	pdf_fun(77.30,0);
Xception			=	pdf_fun(75.00,0);
NASNetMobile			=	pdf_fun(79.00,0);
NASNetLarge			=	pdf_fun(82.00,0);
InceptionResnetV2		=	pdf_fun(80.30,0);
vgg16				=	pdf_fun(71.30,0);
ResNet50			=	pdf_fun(74.90,0);"""

# End PDFs
###############################################################
###############################################################
# Global Vars

#SLOs
def set_option(which_gpu, fraction_memory):
    #config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = fraction_memory
    #config.gpu_options.visible_device_list = which_gpu
    #set_session(tf.Session(config=config))
    if which_gpu == -1:
        print("setting CPU********************")
        my_devices = tf.config.list_physical_devices(device_type='CPU')
        tf.config.set_visible_devices([], 'GPU')
        #tf.config.set_visible_devices(devices= my_devices, device_type='CPU')
        return

    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)
    tf.config.set_visible_devices(physical_devices[which_gpu], 'GPU')
    return


def parse_arguments():
    try:
        args_parser = argparse.ArgumentParser(description="cocktail combinations")
        args_parser.add_argument('-l', "--latency", default='', action='store',type=float, dest='latency',help="Target SLO")
        args_parser.add_argument('-c', "--cost", default='', action='store', dest='cost',type=float,help="Target Cost")
        args_parser.add_argument('-a', "--accuracy", default='', action='store', type=float, dest='accuracy',help="Target Accuracy")
        args_parser.add_argument('-b', "--batch", default=10, action='store', type=float, dest='batch',help="Target Accuracy")
        args_parser.add_argument('-s', "--scheme", default='', action='store', type=str, dest='scheme',help="Scheme infaas")
        args_parser.add_argument('-p', "--policy", default='constant', action='store', type=str, dest='policy',help="aggressive")
        args_parser.add_argument('-g', "--gpu", default=-1, action='store', type=int, dest='gpu',help="aggressive")
        args = args_parser.parse_args()

        return args

    except:
            class ArgsDefault:
                latency = 0
                cost = 0
                accuracy = 0
                scheme = "ensembling"
                batch=10
            args = ArgsDefault() 
            args.latency = 40
            args.accuracy = 70
            args.cost = 0.5
            print("default arguments added ", args)
            #log.info("default arguments added ", args)
#args = parse_arguments()
#slo_accuracy			=	0.75;#float(args.accuracy)
#slo_cost			=	1; #float(args.cost)
#slo_latency			=	50;#float(args.latency)
args = parse_arguments()
print(args)
slo_accuracy			=	float(args.accuracy)
slo_cost			=	float(args.cost)
slo_latency			=	float(args.latency)
scheme				=	args.scheme
policy				=	args.policy
batch				= 	args.batch
gpu                             =       args.gpu
accuracy_margin			=	0.05
infaas 				=	False
cost_margin			=	0.1
latecy_margin			=	10
set_option(gpu, 0.9)
weights = defaultdict(lambda: defaultdict(int))
with open('all-models-classes', mode='r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
        model = rows[0]
        Class = rows[1]
        predictions = rows[2]
        weights[model][Class]=predictions   


# Vars
inst_list			=	[]; # list of all active instances - a list of class instace

correct_predictions	= 	defaultdict(list)
class_weights		= 	defaultdict(lambda: defaultdict(int))
active_instaces		=	len(inst_list);
base_spot_cost		=	0.02;		#FIXME Jash
current_latency		=	0;		# max of all instances latency
current_accuracy	=	0;		#
current_cost		=	0;		# sum of all instance cost
instance_list		=	[];
model_lat_list		=	[315,151.96, 119.2, 74, 152.21, 89.5, 102.35, 98.22, 78.18, 41.5, 259, 43.45]
#top_accuracy_list	=	[82.3,80.30, 79.00, 77.90, 77.30, 76.00, 75.00, 74.90, 74.40, 71.30, 71.30, 70.40]
top_accuracy_list	=	[74.6,73, 69.75, 67.9, 72.83, 66, 70, 65,71.1, 68.05, 71.30, 68.36 ]
model_name_list		=	['NASNetLarge','InceptionResNetV2', 'Xception', 'InceptionV3', 'DenseNet201', 'ResNet50V2', 'DenseNet121', 'ResNet50', 'NASNetMobile', 'MobileNetV2', 'VGG16', 'MobileNet']
accuracy_list=defaultdict(float)
for i in range(len(top_accuracy_list)):
	accuracy_list[model_name_list[i]] = top_accuracy_list[i]

active_model_list	=	[];
union_model_list	=	[];

def handler(signal_received, frame):
    # Handle any cleanup here
	print('SIGINT or CTRL-C detected. Exiting gracefully')
	filename = str(time.time())+"-output.csv"
	#dict = {'Python' : '.py', 'C++' : '.cpp', 'Java' : '.java'}
	for k, val in class_weights.items():
		for key,value in val.items():
    			#w.writerow([k,key, value])
    			exit(0)

	exit(0)


def write_to_file(logfile, results):
    # Handle any cleanup here
	print('Writing to fle class weights', logfile)
	batch_file_path = "/home/cc/"+str(slo_latency)+ "-" + str(slo_accuracy) + "-"  + str(batch) + "-results.csv"
	batch_file = open(batch_file_path,"w")
	batch_writer = csv.writer(batch_file)
	batch_writer.writerows(results)

#dict = {'Python' : '.py', 'C++' : '.cpp', 'Java' : '.java'}
	w = csv.writer(open(logfile, "w"))
	for k, val in class_weights.items():
		for key,value in val.items():
    			w.writerow([k,key, value])


# Debugs
verbosity			= 	1;

# End Global Vars
###############################################################

###############################################################
# Classes
def infaas_select_model():
		global slo_latency, slo_accuracy
		candidate_models = []
		for itr in range(len(model_lat_list)):
				if model_lat_list[itr] <= slo_latency:
					candidate_models.append([itr,top_accuracy_list[itr]])
		print(candidate_models)
		if candidate_models:
			
			model = max(candidate_models, key=lambda x:x[1])
			print(model, model_name_list[model[0]])
			return [model_name_list[model[0]]]

def find_model(models):
		global slo_latency, slo_accuracy
		candidate_models = []
		print("finding in models", models)
		for model in models:
				if model_lat_list[model_name_list.index(model)] <=  slo_latency:
					candidate_models.append([model_name_list.index(model),top_accuracy_list[model_name_list.index(model)]])
		if not candidate_models:
			return None
		print(candidate_models)
		if candidate_models:
			
			model = max(candidate_models, key=lambda x:x[1])
			print(model, model_name_list[model[0]])
			return [model_name_list[model[0]]]

class instance:
	def __init__(self):
		print("New Spot Instce Created")
		print("Instance Info ... 4vCPU, memory = xxx, GPU = YYY")
		
		#self.id = active_instaces + 1
		self.my_latency			=	0
		#self.my_cost			=	base_spot_cost
		self.my_accuracy		=	0;
		self.my_model_list		=	[];
		self.my_latency_list	=	[];
		self.packingNumber		=	0;
	#current_cost			=	current_cost + base_spot_cost
		
	print ('Init Complete...')
		
	def add_model_to_instance(self):#forceScaleIn):
		global model_lat_list
		global model_name_list
		global slo_latency
		global active_model_list
		if (self.my_latency < slo_latency):
			#for itr in range(len(model_lat_list)):
			for itr in range(len(model_lat_list)):
				obey_latency		=	model_lat_list[itr] < (slo_latency + latecy_margin);
				#obey_latency		=	slo_latency + latecy_margin
				obey_duplication	=	model_name_list[itr] not in active_model_list	
				print("Adding more models to instance ",active_model_list, self.my_latency, obey_latency, obey_duplication)		
				
				if (obey_latency and obey_duplication):
					self.my_latency	=	max(self.my_latency , obey_latency);
					active_model_list.append(model_name_list[itr]);
					self.my_model_list.append(model_name_list[itr]);
					self.my_latency_list.append(model_lat_list[itr]);
					forceIt			=	0;
					self.my_latency = max(self.my_latency_list);
					print(str(model_name_list[itr]) + " Added #####################################")
					print(self.my_model_list, self.my_latency)
					print("Scale in Successful...")
					
					return 1;
				
				'''
				elif (model_lat_list[itr] < (slo_latency + latecy_margin - self.my_latency)):# and forceScaleIn == 1):
					self.my_latency	=	self.my_latency + model_lat_list[itr];
					self.my_model_list.append(model_name_list[itr]);
					self.my_latency_list.append(model_lat_list[itr]);
					forceIt			=	0;
					self.my_latency = sum(self.my_latency_list);
					
					return 1;
				''' #force model repitition, add it later	
		else: 
			#forceIt = 1;
			print("cannot add more models ", self.my_latency, slo_latency)
			return 0
	
	def my_info(self):
		print(self.my_model_list)
		print(self.my_latency_list)
		print(self.my_latency)
					

# End Classes
###############################################################

###############################################################
# Function defs

#def scale_in(instance_num):	
	#if((current_accuracy < slo_accuracy) and current_latency < )
		
def scale_up():
	#if I can scale in, I'll always scale-in, as getting more instances is more costly
	global inst_list
	global current_cost
	global current_accuracy
		
	scale_up_success = 0;
	print("Values before scale up currentcost " + str(current_cost + base_spot_cost) + "slo cost " + str (slo_cost + cost_margin))
	if ((current_cost + base_spot_cost < slo_cost + cost_margin)): #condition to call scale up
		
		inst_list.append(instance())
		scale_up_success = inst_list[-1].add_model_to_instance();
		current_cost = current_cost + base_spot_cost; #already called by instace.init()
		current_accuracy = get_global_accuracy()
	
	#check success
	if (scale_up_success == 1):
		return 1
	else:
		return 0
			
def get_global_accuracy():
	
	global inst_list;
	scores				=	0;
	score				=	[];
	joint_model_list	=	[];
	global union_model_list
	for inst in inst_list:
		joint_model_list	=	sorted(joint_model_list + inst.my_model_list); # union with repetition and order
		#print(joint_model_list)
		union_model_list	=	list(set().union(union_model_list, inst.my_model_list)); # union without repitition
		#print(union_model_list)
	
	for i in range(1000):
		#for inst in union_model_list:
		for inst in joint_model_list:
			#scores = scores + joint_model_list.count(inst) * random.choice(get_model_key(inst, 0))
			score.append(random.choice(get_model_key(inst, 0)))
		#if(score.count(1) > score.count(0)):
			#scores = scores + 1
		#print("Score of instance " + str(i) + " is " + str(scores))
	scores = score.count(1)
	#print("Average score is " + str(scores/1000))
	
	#print ("score is :" + str(scores))
	accuracy = 	scores/(1000 * len(joint_model_list))
	#print(scores)
	#print(len(joint_model_list))
	#accuracy 	=	 scores/1000
	print("Current Global Accuracy is: " + str(accuracy))
	return accuracy
		
	

def get_model_key(model_name, verbose):

	if (model_name == 'MobileNet'):
		model = MobileNetV1
	
	elif (model_name == 'MobileNetV2'):
		model = MobileNetV2
	
	elif (model_name == 'InceptionV3'):
		model = InceptionV3
		
	elif (model_name == 'ResNet50V2'):
		model = ResNet50V2
		
	elif (model_name == 'DenseNet201'):
		model = DenseNet201
		
	elif (model_name == 'DenseNet121'):
		model = DenseNet121
		
	elif (model_name == 'Xception'):
		model = Xception
		
	elif (model_name == 'NASNetMobile'):
		model = NASNetMobile
	
	elif (model_name == 'NASNetLarge'):
		model = NASNetLarge
		
	elif (model_name == 'InceptionResNetV2'):
		model = InceptionResnetV2
		
	elif (model_name == 'VGG16'):
		model = vgg16
	
	elif (model_name == 'ResNet50'):
		model = ResNet50
	
	else:
		model = [];
		
	if (verbose == 1):
		print("model accuracy" + str(sum(model)/len(model)))
		
	return model
	
def get_global_latency():
	global_latency_list	=	[];
	global inst_list
	for inst in inst_list:
		global_latency_list.append(inst.my_latency);
	global current_latency
	current_latency	=	max(global_latency_list);
	return current_latency;
	

def init_scale():
	scale_ok	=	scale_up();
	if (scale_ok == 1):
		print("##### Autoscaler initialized #####");
	else:
		print("Could not meet requirements");
		
def baseline_optimizer():
	global slo_accuracy		
	global slo_cost			
	global slo_latency		
 
	global accuracy_margin	
	global cost_margin		
	global latecy_margin	
 
	global model_lat_list	
	global top_accuracy_list
	global model_name_list	

	norm_lat_dist				=	[]
	norm_acc_dist				=	[]
	
	for lat in model_lat_list:
		norm_lat_dist.append(1-(lat/(slo_latency + latecy_margin)))
	
	for acc in top_accuracy_list:
		norm_acc_dist.append(1- (acc/slo_accuracy - accuracy_margin))
	
	#dist_metric	=	
	
def printv(string):
	global verbosity;
	if (verbosity == 1):
		print(string);
# End Function Defs
###############################################################

###############################################################
# Main()

def vote_based_scaling(step_accuracy,overall_accuracy,correct_predictions,pretrained_model_list, MajorityVote):
	print("vote scaling " ,step_accuracy, overall_accuracy, (slo_accuracy + 0.02)*100, MajorityVote, len(MajorityVote))
	drop = []
	if ((step_accuracy) >= ((slo_accuracy + 0.02)*100)) and len(correct_predictions) > 1:
		excess = max(MajorityVote) -  math.ceil(0.51*len(pretrained_model_list))
		if(excess >= 1):
			for i in range(excess):	
				index,drop_model = min((len(correct_predictions[key]),key) for key in correct_predictions )
				drop.append([e for e in pretrained_model_list if e[0] == drop_model])
				print("least model is " ,index, drop_model,drop)
				pretrained_model_list.remove(drop[0])
                #union_model_list.remove(drop_model)
				del correct_predictions[drop_model]
			return drop 
	elif ((step_accuracy) <= ((slo_accuracy - 0.02)*100)):		
		remaining_models = set(model_name_list) - set([x[0] for x in pretrained_model_list])
		print("remaining_models",remaining_models,set(model_name_list),set(pretrained_model_list[0]))
		if remaining_models:
			model = find_model(remaining_models)
			if model:
				print("model added", model)
				cmd = 'tf.keras.applications.' + str(model[0]) + '()';
				pretrained_model = eval(cmd)
				pretrained_model_list.append([model[0],pretrained_model])
				return model
			else:
				print("***no model available to add *********")
				return "None"

def aggressive_scaling(step_accuracy,overall_accuracy,correct_predictions,pretrained_model_list):
        print("aggressive_scaling " ,step_accuracy, overall_accuracy, (slo_accuracy + 0.02)*100)
        if ((step_accuracy) >= ((slo_accuracy + 0.02)*100)) and len(correct_predictions) > 1:
                index,drop_model = min((len(correct_predictions[key]),key) for key in correct_predictions )
                drop = [e for e in pretrained_model_list if e[0] == drop_model]
                print("least model is " ,index, drop_model,drop)
                pretrained_model_list.remove(drop[0])
                #union_model_list.remove(drop_model)
                del correct_predictions[drop_model]
                return drop 
        elif ((step_accuracy) <= ((slo_accuracy - 0.02)*100)):
                remaining_models = set(model_name_list) - set([x[0] for x in pretrained_model_list])
                print("remaining_models",remaining_models,set(model_name_list),set(pretrained_model_list[0]))
                if remaining_models:
                    model = find_model(remaining_models)
                    if model:
                        print("model added", model)
                        cmd = 'tf.keras.applications.' + str(model[0]) + '()';
                        pretrained_model = eval(cmd)
                        pretrained_model_list.append([model[0],pretrained_model])
                        return model
                    else:
                        print("***no model available to add *********")
                        return "None"
	

def main():
	global inst_list, current_latency, current_cost, infaas, matched, BLmatch, not_matched, step_matching_pred,weights
	print('main invoked',slo_accuracy,slo_latency,slo_cost)


	if scheme == "infaas":
		infaas = True
	signal(SIGINT, handler)
	time_Scale = 1
	init_scale();
	if infaas:
		return infaas_select_model() 
	for ts in range(time_Scale): # 1,000,000 itterations of autoscaling
		global_accuracy 	=	get_global_accuracy();
		while (global_accuracy < slo_accuracy):
			print("Not meeting slo accuracy", global_accuracy, slo_accuracy);
			scale_in_succ	=	inst_list[len(inst_list)-1].add_model_to_instance()
			if(scale_in_succ == 1):
				print("Scaling in instace " + str(len(inst_list)))
				global_accuracy 	=	get_global_accuracy();
			else:
				print("Trying to Scale-Up... ")
				scale_ok			=	scale_up();
				global_accuracy 	=	get_global_accuracy();
				if (scale_ok == 1):
					print("scaling Up... ")
				else:
					print("Could NOT meet requirements",global_accuracy,get_global_latency(), current_cost);
					print("################################################")
					print("#################### FAILED ####################")
					print("################################################")
					break;
		print("Completed TimeScale Num:  " + str(ts))
	print("Done for Target slo accuracy and cost", slo_accuracy, slo_latency, slo_cost, len(inst_list))
	

	############################ Testing the entire test Suite ####################################
	
	pretrained_model_list	=	[]
	results = []
	accuracies = defaultdict()
	if len(union_model_list)-5 > 0:
		for i in union_model_list:
			accuracies[i]=(accuracy_list[i])

                
		print(accuracies)
		for i in range(len(union_model_list)-5):
			min_model = min(accuracies,key=accuracies.get)
			print(min_model)
		#for i in range(len(union_model_list)-5:
			union_model_list.remove(min_model)
			del accuracies[min(accuracies,key=accuracies.get)]
			

	for model in union_model_list:
		cmd = 'tf.keras.applications.' + str(model) + '()';
		pretrained_model = eval(cmd)
		pretrained_model_list.append((model,pretrained_model))

	fcount = 0;
	votearray = []
	MajorityVote = []
	voteclassarray = []
	num_matching_pred	=	0;
	num_non_matching_pred	=	0;
	weighted_matching	=	0;
	baselineModel = eval('tf.keras.applications.NASNetLarge()')
	step_length = batch
	correct_predictions = defaultdict(list)
	logfile = str(time.time()) + "-" + str(slo_latency)+ "-" + str(slo_accuracy) + "-"  + str(batch) +  "-output.csv"
	for filename in os.listdir('/home/cc/val'):
		stime	=	time.time()
		file = '/home/cc/val/' + str(filename)
	#file = tf.keras.utils.get_file("grace_hopper.jpg","https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg")
		img = tf.keras.preprocessing.image.load_img(file, target_size=[224, 224])
	# plt.imshow(img)
	# plt.axis('off')
		x = tf.keras.preprocessing.image.img_to_array(img)
		x = tf.keras.applications.mobilenet.preprocess_input(
		np.array(img)[tf.newaxis, ...])
	# images will be the input key name
	# scores will be the out key name
		#for model in union_model_list:
		#	cmd = 'tf.keras.applications.' + str(model) + '()';
		#	pretrained_model = eval(cmd)
			#pretrained_model = eval(f'tf.keras.applications.{model}()')
			##print(pretrained_model)
		i=0
		weighted_vote = defaultdict(int)
		for smodels in pretrained_model_list:
			result_before_save = smodels[1](x)
			vote_result = tf.keras.applications.mobilenet.decode_predictions(result_before_save.numpy())[0][0][1]
			vote_class = tf.keras.applications.mobilenet.decode_predictions(result_before_save.numpy())[0][0][0]
			#print("Result before saving",tf.keras.applications.mobilenet.decode_predictions(result_before_save.numpy())[0][0][1])
			print("Result before saving",smodels, smodels[0] ,vote_result, vote_class,weights[smodels[0]][vote_class],result_before_save.numpy()[0][0])
			if vote_class == images[filename.strip('.JPEG')][0]:
				correct_predictions[smodels[0]].append([fcount,vote_class])			
				class_weights[smodels[0]][vote_class]+=1
			#votearray.append(tf.keras.applications.mobilenet.decode_predictions(result_before_save.numpy())[0][0][1]));
			votearray.append(vote_result)
			voteclassarray.append(vote_class)
			weighted_vote[vote_class]+= int(weights[smodels[0]][vote_class])
			i+=1
		
		##print(baselineModel);
		resultBLModel	=	baselineModel(x);
		maxvoteclass = []
		maxvotename = []
		BLClass		=	tf.keras.applications.mobilenet.decode_predictions(resultBLModel.numpy())[0][0][1]
		BLclass		=	tf.keras.applications.mobilenet.decode_predictions(resultBLModel.numpy())[0][0][0]
		maxVoteClass	=	max(set(votearray), key = votearray.count)
		maxVoteclass	=	max(set(voteclassarray), key = voteclassarray.count)
		weighted_result =       max(weighted_vote, key= lambda x: weighted_vote[x])
		counts = {x:votearray.count(x) for x in votearray}
		maxcount = counts[maxVoteClass]
		for i in counts:
		    if counts[i] == maxcount:
		        maxvotename.append(i)
		counts = {x:voteclassarray.count(x) for x in voteclassarray}
		maxcount = counts[maxVoteclass]
		for i in counts:
		    if counts[i] == maxcount:
		        maxvoteclass.append(i)
		MajorityVote.append(len(maxvoteclass))
		if len(maxvoteclass) > 1:
                    print("tied ",maxvotename,maxvoteclass)
		print("weighted voted are ", weighted_vote)
		print("$$$ Class Info $$$ " + str(collections.Counter(votearray)) + " ### VS ### " + str(BLClass), BLclass, maxVoteclass,weighted_result ,tf.keras.applications.mobilenet.decode_predictions(resultBLModel.numpy())[0][0])
                    
		fcount 	=	fcount	+ 1
		if (maxVoteclass == BLclass):
			num_matching_pred	=	num_matching_pred + 1;
			#step_matching_pred+=1
			print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
			print("Matched For " + str(filename) + " Matching Pred ====== " + str(num_matching_pred) + " ### Completed: " + str(fcount) + "/50,000", maxVoteClass,BLClass,maxVoteclass,BLclass)
			print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
			print("\n")
		if (weighted_result == BLclass):
			weighted_matching+= 1;
			#step_matching_pred+=1
			print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
			print("Matched For " + str(filename) + " Weighted Matching Pred ====== " + str(weighted_matching) + " ### Completed: " + str(fcount) + "/50,000", BLClass,weighted_result,BLclass)
			print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
			print("\n")

		if(maxVoteclass!=BLclass and weighted_result!=BLclass):
			num_non_matching_pred	=	num_non_matching_pred + 1;
			print("-------------------------------------------------------------------------------")
			print("Did NOT Match For " + str(filename) + " NOT Matching Pred ====== " + str(num_non_matching_pred) + " ### Completed: " + str(fcount) + "/50,000",maxVoteClass,BLClass,maxVoteclass,BLclass)
			print("-------------------------------------------------------------------------------")
			print("\n")
		
		etime		=	time.time()
		print("Time to Process Image	=	" + str(etime - stime),[maxVoteclass,BLclass])
		check_ground_truth([maxVoteclass,BLclass,weighted_result],filename.strip('.JPEG'))
		if (fcount%step_length == 0):
			step_accuracy = step_matching_pred/step_length * 100
			overall_accuracy = matched/fcount *100
			result=""
			if policy=="aggressive":
                                result = aggressive_scaling(step_accuracy,overall_accuracy,correct_predictions,pretrained_model_list)
			elif policy=="vote":
                                result = vote_based_scaling(step_accuracy,overall_accuracy,correct_predictions,pretrained_model_list, MajorityVote)


			for key in correct_predictions.keys():
				print(key, len(correct_predictions[key]))
				correct_predictions[key] = []
			MajorityVote = []
			pretrained_models = [e for e,m in pretrained_model_list]
			print("overall prediction accuracy is ", overall_accuracy, policy)
			print("current step prediction accuracy is ", step_accuracy, step_matching_pred, step_length)
			#original = sys.stdout
			#sys.stdout = f
			results.append(["batchId ," , str(fcount/step_length), ",overall_accuracy, ",str(overall_accuracy) ,",step_accuracy, " ,str(step_accuracy), ",models, ",str(collections.Counter(pretrained_models)),",",str(len(pretrained_models)),",",str(result)])
                        #sys.stdout = original
			write_to_file(logfile,results)
			#batch_file.close()
			#print(correct_predictions)
			
			step_matching_pred=0
		voteclassarray = []
		votearray=[]

	text_file = open("sample.txt", "wt")
	n = text_file.write(str(num_matching_pred))
	text_file.close()
	print ("###$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$###")	
	#print(len(inst_list))
	#print(current_cost)
	#print(global_accuracy)
'''	
	g_acc = 0;
	init_call();
	for i in range(10000):
		g_acc	=	g_acc + get_global_accuracy();
	
	print("Average Accuracy")
	print(g_acc/10000)
	
	for item in inst_list:
		item.my_info()
'''	
# end Main
###############################################################
if __name__ == "__main__":
	main()	

