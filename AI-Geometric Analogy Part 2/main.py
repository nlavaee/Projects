import sys
import os
import math
import string
import re
import numpy as np
from collections import defaultdict, OrderedDict, Counter
from copy import deepcopy
import itertools
from itertools import permutations, repeat
from A1_solution_update import main as a1
import shutil

def main():



	directory = os.getcwd() + "/" + sys.argv[1] + "/"
	# print(sys.argv[1].split("/"))
	path = sys.argv[1].split("/")
	print(path[0])
	output_dir = os.getcwd() + "/" + path[0] + "/output"
	new_dir = os.getcwd() + "/" + path[0] + "/new" # change the name of this to A1 results
	# print("path is ", os.path.dirname(output_dir))
	if(os.path.exists(output_dir)):
		shutil.rmtree(output_dir)
	if(os.path.exists(new_dir)):
		shutil.rmtree(new_dir)
	os.makedirs((output_dir), exist_ok=True)
	os.makedirs((new_dir), exist_ok=True)
	# os.mkdir(temp_dir + "output")
	# output_dir = temp_dir + "output"
	# print("directory is ", directory)
	finalList = []
	all_shapes = []
	file_names = []

	for file in os.listdir(directory):
		# print("file is ", file)
		file_names.append(file)
		# print(file) # This is when it goes to the next file
		if(file.endswith(".txt")):
			
			elem = os.getcwd() + "/" + sys.argv[1] + "/" + file
			# print("elem is ", elem)
			# elem1 = os.getcwd() + "/" 
			temp_list = a1(elem, output_dir)
			a1(elem, new_dir)

			alpha = 97
			for interpret in temp_list:
				shape_name, garbage = file.split(".")
				# print(shape_name)
				
				# if(shape_name[0] != 'k'):
				shape_num = 1
				for obj in interpret:
					# print("the name is ", obj.name, '\n')
					if(obj.name[0] == "p"):
						obj.name = shape_name + chr(alpha) + str(shape_num)
					# print("the name is now ", obj.name, '\n')
						shape_num += 1
					elif(obj.name[0] == "d"):
						obj.name = shape_name + chr(alpha) + obj.name

					elif(obj.name[0] == "c"):
						obj.name = shape_name + chr(alpha) + obj.name
						# print("new name is ", obj.name)
				alpha += 1

			finalList.append(temp_list)
		# print(finalList)

	sizes = ["small", "large"]
	relations = ["inside", "overlap"]
	relations_dict = defaultdict(list)
	size_dict = defaultdict(list)
	for file in os.listdir(output_dir):
		if(file.endswith(".txt")):
			# print("file is ", file)
			with open(os.path.join(output_dir, file)) as f:
				for line in f:
					if any(s in line for s in sizes):
						shape_name, garbage = file.split(".")
						# print("shape name is ", shape_name)
						# print("shape_name is ", shape_name)
						regex = re.compile(".*?\((.*?)\)")
						result = re.findall(regex, line)
						result = ''.join(result)
						size = line.split('(')[0]
						shape = str(result.split(",")[0])

						# print("shape is ", shape[0])

						if(shape[0] == "p"):
							shape = shape.replace(shape[0], shape_name)
						if(shape[0] == "c"):
							shape = shape_name + shape
						if(shape[0] == "d"):
							shape = shape_name + shape

						size_dict[(shape)].append(size)

					if any(s in line for s in relations):
				
						shape_name, garbage = file.split(".")
			
						regex = re.compile(".*?\((.*?)\)")
						result = re.findall(regex, line)
						result = ''.join(result)
						relation = line.split('(')[0]
						shape1, shape2 = result.split(",")
			
						if(shape1[0] == "p"):
							shape1 = shape1.replace(shape1[0], shape_name)
						if(shape2[0] == "p"):
							shape2 = shape2.replace(shape2[0], shape_name)
						if(shape1[0] == "c"):
							shape1 = shape_name + shape1
						if(shape2[0] == "c"):
							shape2 = shape_name + shape2
						if(shape1[0] == "d"):
							shape1 = shape_name + shape1
						if(shape2[0] == "d"):
							shape2 = shape_name + shape2

						relations_dict[(shape1, shape2)].append(relation) # what if multiple relations, change this to append?



	temp_relations_dict = deepcopy(relations_dict)

	# remove duplicates in the relations list
	for key, val in relations_dict.items():
		temp_relations_dict[key] = list(set(val))

	relations_dict = temp_relations_dict


	input_A = []
	input_B = []
	input_C = []
	options_k = []
	for item in finalList:
		numb_interpretations = 0
		for interpret in item:
		    # print("The length of the interpretations are ", len(interpret), '\n')
		    numb_interpretations += 1
		    temp = []
		    for obj in interpret:
		    	temp.append(obj)

		    if( temp[0].name.startswith("A") or temp[0].name.startswith("a")):
		    	input_A.append(temp)
		    elif( temp[0].name.startswith("B") or temp[0].name.startswith("b")):
		    	input_B.append(temp)
		    elif( temp[0].name.startswith("C") or temp[0].name.startswith("c")):
		    	input_C.append(temp)
		    elif( temp[0].name.startswith("k") or temp[0].name.startswith("K")):
		    	options_k.append(temp)
		
		# print("number of interpretations is ", numb_interpretations, '\n')

	cost_size =  7 
	cost_delete = 24  
	cost_add = 29.1
	cost_change_shape = 19 

	cost_area = 7.3

	hloc_left_cost = 3.0
	hloc_right_cost = 3.1
	hloc_center_cost = 3.2
	vloc_bottom_cost = 3.3
	vloc_top_cost = 3.4
	vloc_middle_cost = 3.5
	overlap_cost = 6.7
	inside_cost = 6.0 



	transform_AB = defaultdict()
	transform_Ck = defaultdict()
	transform_AC = defaultdict()
	transform_Bk = defaultdict()

	def Generate_Transformations(list1, list2, transform_dict):
		new_list = list(list(zip(r, p)) for (r, p) in zip(repeat(list1), permutations(list2)))
		# print("new interpretation")
		# print("new list is ", new_list)


		add_or_delete = -1
		if("*" in list1):
			# print("got here add")
			add_or_delete = 1
		if("*" in list2):
			# print("got into delete")
			add_or_delete = 0

		# print("list1_len is ", list1_len, '\n')
		# print("list2_len is ", list2_len, '\n')
		# for item in new_list:
		# 	for obj in item:
				# if(obj[0] != "*" and obj[1] != "*"):
					# print("the objects are ", obj[0].name, obj[1].name)
		for item in new_list:
			# print(item, '\n')
			# print("a transformation")
			running_cost = 0
			transform_key = ""
			relations_key = []
			# total_num_transformations = 0

			for obj in item:
				
				# print(obj[0].hloc)
				if obj[0] != "*" and obj[1] != "*":

					transform_key += ("(" + obj[0].name + "," + obj[1].name + ")")
					relations_key.append([obj[0].name, obj[1].name])


					if(obj[0].shape != obj[1].shape):
						# print("change(" + obj[0].shape + "(" + obj[0].name + 
						# ")" + "," + obj[1].shape + "(" + obj[1].name + ")" + ")" + '\n')
						# print("got here 7")
						running_cost += cost_change_shape

						# check big or small
						# both objects have a size attribute, check if they match
						if(obj[0].name in size_dict.keys() and obj[1].name in size_dict.keys()):
							if(size_dict[obj[0].name] != size_dict[obj[1].name]):
								# print("change(" + str(size_dict[obj[0].name]) + "(" + obj[0].name + ")," + str(size_dict[obj[1].name]) + "(" + obj[1].name + "))" + '\n')
								# print("got here 13")
								running_cost += cost_size

						elif(obj[0].name in size_dict.keys() and obj[1].name not in size_dict.keys()):
							# print("change(" + str(size_dict[obj[0].name]) + "(" + obj[0].name + "),(" + obj[1].name + "))" + '\n')
							# print("got here 14")
							running_cost += cost_size

						elif(obj[0].name not in size_dict.keys() and obj[1].name in size_dict.keys()):
							# print("change((" + obj[0].name + ")," + str(size_dict[obj[1].name]) + "(" + obj[1].name + "))" + '\n')
							# print("got here 15")
							running_cost += cost_size


					if(obj[0].shape == obj[1].shape):

						if(obj[0].shape == "scc" and obj[1].shape == "scc" and (len(obj[0].vertices) != len(obj[1].vertices))):
							# print("change(" + obj[0].shape + "(" + obj[0].name + 
							# ")" + "," + obj[1].shape + "(" + obj[1].name + ")" + ")" + '\n')
							# print("got here 16")
							running_cost += cost_change_shape



						elif(obj[0].shape != 'dot' and obj[1].shape != 'dot' and obj[0].area != obj[1].area):
							# print("got here 8")
							
							if(obj[0].area < obj[1].area):
								running_cost += cost_size
								# print("change(small(" + obj[0].name + "),large(" + obj[1].name + "))" + '\n')
							elif(obj[0].area > obj[1].area):
								running_cost += cost_size
								# print("change(large(" + obj[0].name + "),small(" + obj[1].name + "))" + '\n')


					if(obj[0].hloc != obj[1].hloc):
						if(obj[0].hloc == "left"):
							running_cost += hloc_left_cost
						if(obj[0].hloc == "right"):
							running_cost += hloc_right_cost
						if(obj[0].hloc == "center"):
							running_cost += hloc_center_cost
						if(obj[1].hloc == "left"):
							running_cost += hloc_left_cost
						if(obj[1].hloc == "right"):
							running_cost += hloc_right_cost
						if(obj[1].hloc == "center"):
							running_cost += hloc_center_cost
						# print("got here 9")
						# print("move(" + obj[0].hloc + "(" + obj[0].name + ")," + obj[1].hloc + "(" 
						# 	  + obj[1].name + "))" + '\n')
						# running_cost += h_loc_cost

					if(obj[0].vloc != obj[1].vloc):

						if(obj[0].vloc == "bottom"):
							running_cost += vloc_bottom_cost
						if(obj[0].vloc == "top"):
							running_cost += vloc_top_cost
						if(obj[0].vloc == "middle"):
							running_cost += vloc_middle_cost
						if(obj[1].vloc == "bottom"):
							running_cost += vloc_bottom_cost
						if(obj[1].vloc == "top"):
							running_cost += vloc_top_cost
						if(obj[1].vloc == "middle"):
							running_cost += vloc_middle_cost
						# print("got here 10")
						# print("move(" + obj[0].vloc + "(" + obj[0].name + ")," + obj[1].vloc + "(" 
						#       + obj[1].name + "))" + '\n')

				if(add_or_delete == 1 and (obj[0] != "*" and obj[1] == "*")):
				# if obj[0] != "*" and obj[1] == "*":
					transform_key += ("(" + obj[0].name + "," + obj[1] + ")")
					# print("got here 11")
					# print("add(" + obj[0].name + ")" + '\n')
					running_cost += cost_add
					# add_or_delete = -1

				# this is a delete
				# elif obj[0] != "*" and obj[1] == "*":
				elif(add_or_delete == 0 and (obj[0] != "*" and obj[1] == "*")):
				# if(obj[0] != "*" and obj[1] == "*"):
					transform_key += ("(" + obj[0].name + "," + obj[1] + ")")
					# print("got here 12")
					# print("delete(" + obj[0].name + ")" + '\n')
					running_cost += cost_delete
					# add_or_delete = -1






			# print("final relations key is ", relations_key)
			# print("relations dict is ", relations_dict)
			# print("temp relations dict is ", temp_relations_dict)
			if(len(relations_key) >= 2):
				# print("relations key is ", relations_key[0][0], relations_key[1][0], '\n')
				# print("relations key is "+ (relations_key[0][1]) + "," + relations_key[1][1] + '\n')
				for i in range(0, len(relations_key)):
					for j in range(i + 1, len(relations_key)):
						# print("the relations are ", relations_key[i], relations_key[j], '\n')
						# print("first set ", relations_key[i][0] + relations_key[j][0], "second set ", relations_key[i][1] + relations_key[j][1], '\n')
						# print("first set ", relations_key[j][0] + relations_key[i][0], "second set ",  relations_key[j][1] + relations_key[i][1], '\n')


				# RElATIONS
						if((relations_key[i][0], relations_key[j][0]) in relations_dict.keys() and
						   (relations_key[i][1], relations_key[j][1]) in relations_dict.keys()):
							if(relations_dict[(relations_key[i][0], relations_key[j][0])] != 
								relations_dict[(relations_key[i][1], relations_key[j][1])]):
								# print("got here 1")
								relations_set = (set(relations_dict[(relations_key[i][0], relations_key[j][0])]).symmetric_difference(set(relations_dict[(relations_key[i][1], relations_key[j][1])])))
								# print(relations_set)
								for relate in relations_set:
									if(relate == "overlap"):
										running_cost += overlap_cost
									if(relate == "inside"):
										running_cost += inside_cost


						if((relations_key[i][0], relations_key[j][0]) in relations_dict.keys() and
						   (relations_key[i][1], relations_key[j][1]) not in relations_dict.keys()):
							# print("got here 2")
							for relate in relations_dict[(relations_key[i][0], relations_key[j][0])]:
								if(relate == "overlap"):
									running_cost += overlap_cost
								if(relate == "inside"):
									running_cost += inside_cost


						if((relations_key[i][0], relations_key[j][0]) not in relations_dict.keys() and
						   (relations_key[i][1], relations_key[j][1]) in relations_dict.keys()):
							# print("got here 3")
							for relate in relations_dict[(relations_key[i][1], relations_key[j][1])]:
								if(relate == "overlap"):
									running_cost += overlap_cost
								if(relate == "inside"):
									running_cost += inside_cost


						if((relations_key[j][0], relations_key[i][0]) in relations_dict.keys() and
						   (relations_key[j][1], relations_key[i][1]) in relations_dict.keys()):
							if(relations_dict[(relations_key[j][0], relations_key[i][0])] != 
								relations_dict[(relations_key[j][1], relations_key[i][1])]):
								# print("got here 4")
								relations_set = (set(relations_dict[(relations_key[j][0], relations_key[i][0])]).symmetric_difference(set(relations_dict[(relations_key[j][1], relations_key[i][1])])))
								# print(relations_set)
								for relate in relations_set:
									if(relate == "overlap"):
										running_cost += overlap_cost
									if(relate == "inside"):
										running_cost += inside_cost


						if((relations_key[j][0], relations_key[i][0]) in relations_dict.keys() and
						   (relations_key[j][1], relations_key[i][1]) not in relations_dict.keys()):
							# print("got here 5")
							for relate in relations_dict[(relations_key[j][0], relations_key[i][0])]:
								if(relate == "overlap"):
									running_cost += overlap_cost
								if(relate == "inside"):
									running_cost += inside_cost

						if((relations_key[j][0], relations_key[i][0]) not in relations_dict.keys() and
						   (relations_key[j][1], relations_key[i][1]) in relations_dict.keys()):
							# print("got here 6")
							for relate in relations_dict[(relations_key[j][1], relations_key[i][1])]:
								if(relate == "overlap"):
									running_cost += overlap_cost
								if(relate == "inside"):
									running_cost += inside_cost


			elif(len(relations_key) == 1):
				# print("the transform key in this case is ", transform_key, '\n')
				new_key = re.findall('\(.*?\)', transform_key)
				# print("new key is ", new_key, '\n')
				for elem in new_key:
					elements = tuple(elem[1:-1].split(","))

					# print("elements are ", elements, '\n')

					if("*" in elements):
						# print("elements are ", elements[0], elements[1], '\n')
						for key in relations_dict.keys():
							if(elements[0] == key[0]):
								# print(relations_dict[key])
								# print("got here 17")
								if(relations_dict[key] == ['inside']):
									# print("We made it in here")
									running_cost += inside_cost
								if(relations_dict[key] == ['overlap']):
									running_cost += overlap_cost

			transform_dict[transform_key] = running_cost
		# print("transform_AB is ", transform_dict)
		return transform_dict

	

	for i, A_val in enumerate(input_A):
		for j, B_val in enumerate(input_B):
			# list1_len = len(A_val)
			# list2_len = len(B_val)
			if(len(A_val) > len(B_val)):
				B_val.append("*")
			elif(len(A_val) < len(B_val)):
				A_val.append("*")
			transform_AB = Generate_Transformations(A_val, B_val, transform_AB)


	for i, C_val in enumerate(input_C):
		for j, k_val in enumerate(options_k):
			# list1_len = len(C_val)
			# list2_len = len(k_val)
			if(len(C_val) > len(k_val)):
				k_val.append("*")
			elif(len(C_val) < len(k_val)):
				C_val.append("*")
			transform_Ck = Generate_Transformations(C_val, k_val,transform_Ck)



	# MAKE SURE TO DELETE OUTPUT FOLDER

	visited_AB = deepcopy(transform_AB)

	final_min = math.inf
	final_AB_key = ""
	final_Ck_key = ""
	final_Ck_val = math.inf



	while(len(visited_AB) != 0):
		visited_Ck = deepcopy(transform_Ck)
		# print("visited_Ck is ", visited_Ck, '\n')
		min_AB_key, min_AB_val = min(visited_AB.items(), key=lambda x:x[1])
		del visited_AB[min_AB_key]
		
		

		while(len(visited_Ck) != 0):
			update = True
			min_Ck_key, min_Ck_val = min(visited_Ck.items(), key=lambda x:x[1])
			del visited_Ck[min_Ck_key]


			# print("Ck up here is ", min_Ck_val, "Ck key is ", min_Ck_key ," AB key is ", min_AB_key,  " AB up here is ", min_AB_val)
			# 45.4
			if(min_Ck_val <= 67.1 and min_AB_val <= 67.1):
				curr_min = abs(min_Ck_val - min_AB_val)

				if(curr_min < final_min):

					final_min = curr_min
					final_AB_key = min_AB_key
					final_Ck_key = min_Ck_key
					final_Ck_val = min_Ck_val


		# print("final min is ", final_min, "final Ab is ", final_AB_key, "final ck is ", final_Ck_key)
	final_Ck_key = final_Ck_key.lower()
	k_index = final_Ck_key.find('k')

	print("k =", final_Ck_key[k_index + 1])
	
	


if __name__ == '__main__':
	main()