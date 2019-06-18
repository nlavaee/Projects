import sys
import os
import math
import string
import re
import numpy as np
from collections import defaultdict, OrderedDict, Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import path as mpath
from copy import deepcopy
import itertools 


def main():

	class Graph:

		def __init__(self):
			self.adjacency_list = defaultdict(list)


	file_name = sys.argv[1]
	output_directory = sys.argv[2]
	input_rep = list()
	edge_rep = list()
	coordinates = list()
	coordinate_pairs = list() 

	line_numb = 1
	with open(file_name) as data:
		for line in data:
			temp_key, value = line.split("(")
			temp_value = re.sub(r" ?\)", "", value)
			
			coordinates = [int(item) for item in temp_value.split(',')]

			garbage, key = temp_key.split("= ")

			key_edge = (line_numb)

			info = (key, coordinates)

			edge_info = (key_edge, coordinates)

			input_rep.append(info)

			if(garbage[0] != 'c' and garbage[0] != 'd'):

				edge_rep.append(edge_info)
				line_numb += 1

		

	g = Graph()

	for value in input_rep:
		if(value[0] != "circle" and value[0] != "dot"):
			it = iter(value[1])
			for item in it:
				coordinate_pairs.append((item, next(it)))


	edges = dict()
	edge_dictionary = dict()

	i = 1
	for item in coordinate_pairs:
		if item not in edges.keys():
			edges[item] = i
			i += 1

	index = 1
	for val in (coordinate_pairs[::2]):
		g.adjacency_list[edges[val]].append(edges[coordinate_pairs[index]])
		g.adjacency_list[edges[coordinate_pairs[index]]].append(edges[val])
		index += 2


	def DFS_util(pt, visited, parent, path, adjacency_list, curr, temp):
		visited[curr] = True
		temp.append(curr)


		if parent != pt and parent != curr and curr != pt:
			if pt in adjacency_list[curr]:
				path.append(deepcopy(temp))
				del temp[-1]
				visited[curr] = False 
				return

		for i in adjacency_list[curr]:

			if visited[i] == False:
				parent = curr
				DFS_util(pt, visited, parent, path, adjacency_list, i, temp)

		del temp[-1]
		visited[curr] = False


	path = []


	def DFS(edges, adjacency_list):
		for pt in edges.values():
			temp = list()
			visited = [False]*(len(edges) + 1)
			DFS_util(pt, visited, None, path, adjacency_list, pt, temp)


		final_path = set(tuple(x) for x in path)
		path_final = [ list(x) for x in final_path ]

		return path_final

	path_final = DFS(edges, g.adjacency_list)

	# find all duplicate paths
	test = deepcopy(path_final)

	for item in test:
		item.sort()

	keep_set = list()

	unique_data = [list(x) for x in set(tuple(x) for x in test)]

	for item in unique_data:
		path_idx = 0
		found_item = False
		while(not found_item):
			if item == test[path_idx]:
				keep_set.append(path_idx)
				found_item = True

			path_idx += 1

	correct_path = list()

	for i in keep_set:
		correct_path.append(path_final[i])

	path_final = correct_path

	# find all duplicate paths

	un_cycle_path = deepcopy(path_final)

	def Create_Cycle(path_final):
		for elem in path_final:
			elem.append(elem[0])

	Create_Cycle(path_final)


	edges_dictionary = dict((v,k) for k,v in edges.items())

	edge_rep_dict = dict(edge_rep)


	def pairwise(iterable):
		a = iter(iterable)
		return zip(a, a)

	def Graph_SCC(un_cycle_path, edges_dictionary):

		# create a structure that contains all the points in your graph
		point_paths = list()
		subList = list()
		for i, val1 in enumerate(un_cycle_path):
			points = list()
			for value in val1:
				points.append(edges_dictionary[value])

			point_paths.append(points)

		
		# create a subList containing the representation of lines
		for path in point_paths:
			temp = list()
			for i, pt in enumerate(path[:-1]):
				temp.append(pt)
				temp.append((path[i+1]))
			subList.append(temp)

		path_adjacency = defaultdict(list)

		# create the adjacency list for the possible paths
		for k, path1 in enumerate(subList):
			counter = k
			for h, path2 in enumerate(subList[k+1:]):
				share_edge = False
				counter += 1
				for item1, item1_next in pairwise(path1):
					for item2, item2_next in pairwise(path2):
						if(((item1 + item1_next) == (item2 + item2_next)) or ((item1_next + item1) == (item2 + item2_next))):
							share_edge = True
				if share_edge == False:
					path_adjacency[k].append(counter)
					path_adjacency[counter].append(k)
			
		interpretations = []

		# second search in order to find all possible interpretations
		for i, elem in enumerate(point_paths):

			visited_cycle = [False]*(len(point_paths))
			len_of_path = 0
			temp_path = []
			temp_path_adj_list = []

			temp_path.append(i)
			len_of_path += len(elem) - 1
			visited_cycle[i] = True
			idx = deepcopy(i)
			temp_path_adj_list.append(path_adjacency[i])
		

			for item in path_adjacency[idx]:

				test = deepcopy(temp_path_adj_list)
				test = list(set(test[0]).intersection(*test))

				if ((item not in temp_path) and (item in test) and (visited_cycle[item] == False)):
					temp_path.append(item)
					temp_path_adj_list.append(path_adjacency[item])
					idx = item
					visited_cycle[item] = True
					len_of_path += len(point_paths[item]) - 1

				if len_of_path == len(coordinate_pairs)/2:
					interpretations.append(temp_path)


		# Get rid of duplicates in the interpretations
		for item in interpretations:
			item.sort()

		interpretations.sort()

		interpretations = list(interpretations for interpretations,_ in itertools.groupby(interpretations))

		#if you only have one path possible total then just graph it
		if(len(subList) == 1):
			interpretations = [[0]]

		#if you only have a circles or dots then just graph those
		alpha1 = 97
		if(len(subList) == 0):
			for item in input_rep:
				if item[0] == 'circle':
					radius = item[1][2]
					timestep = np.arange(0, 2*np.pi, .01)
					x = radius*np.sin(timestep) + item[1][0]
					y = radius*np.cos(timestep) + item[1][1]
					plt.plot(x, y)
					plt.axis('equal')
				if item[0] == 'dot':
					plt.plot(item[1][0], item[1][1], 'o')

			path_name = os.path.basename(file_name)
			path_name = (os.path.splitext(path_name)[0])

			plt.savefig(os.getcwd() + '/' + output_directory + '/' + path_name + chr(alpha1))
			alpha1 += 1
			plt.close()

		# graph all the possible interpretations in unique colors
		alphabet = 97
		for inter in interpretations:
			for pt in inter:
				plt.axis([0, 100, 0, 100])
				plt.plot(*zip(*point_paths[pt]), 'o-')
			for item in input_rep:
				if item[0] == 'circle':
					radius = item[1][2]
					timestep = np.arange(0, 2*np.pi, .01)
					x = radius*np.sin(timestep) + item[1][0]
					y = radius*np.cos(timestep) + item[1][1]
					plt.plot(x, y)
					plt.axis('equal')
				if item[0] == 'dot':
					plt.plot(item[1][0], item[1][1], 'o')
			path_name = os.path.basename(file_name)
			path_name = (os.path.splitext(path_name)[0])

			plt.savefig(os.getcwd() + '/' + output_directory + '/' + path_name + chr(alphabet))
			alphabet += 1
			plt.close()
		
		return interpretations, point_paths, subList
	
	possible_interpretations, points_in_path, path_edges = Graph_SCC(path_final, edges_dictionary)



	edges_representation = deepcopy(edge_rep)

	for i, val in enumerate(edges_representation):
		new_item = []
		for item in val[1]:
			new_item = val[1][2:] + val[1][0:2] 
		edge_rep_dict[i+1] = [edge_rep_dict[i+1], new_item]


	def intersperse(lst, item):
		result = [item] * (len(lst) * 2 - 1)
		result[0::2] = lst
		return result


	def find_sub_list(sl,l):
		sll=len(sl)
		for ind in (i for i,e in enumerate(l) if e==sl[0]):
			if l[ind:ind+sll]==sl:
				return ind,ind+sll-1


	def getSlope(x1, y1, x2, y2):
		if(x1 == x2):
			return math.inf
		slope = (y2-y1)/(x2-x1)
		return slope

	def getYInt(x1, y1, x2, y2):
		s = getSlope(x1, y1, x2, y2)
		if s == math.inf:
			return math.inf
		y = -x1*s+y1
		return y

	def getXInt(x1, y1, x2, y2):
		s = getSlope(x1, y1, x2, y2)
		y_inter = getYInt(x1, y1, x2, y2)
		if s == 0:
			return math.inf
		if s == math.inf:
			return x1
		return -(y_inter)/(s)

	def gather_paths(all_matches, path, short_path_name):
		key1 = 0.0
		key2 = 0.0
		key3 = 0.0
		collinear_map = defaultdict(list)
		path = points_in_path[cycle]
		short_path = []
		all_final_paths = []
		new_path = len(input_rep)
		new_path += 1


		for i, val in enumerate(path[:-1]):
			next_point = path[i+1]
			key1 = getSlope(val[0], val[1], next_point[0], next_point[1])
			key2 = getYInt(val[0], val[1], next_point[0], next_point[1])
			key3 = getXInt(val[0], val[1], next_point[0], next_point[1])


			collinear_map[(key1, key2, key3)].append((val, next_point))

		
		for key, value in collinear_map.items():
			if len(value) > 1:
				for i, it in enumerate(value[:-1]):
					if it[1] == value[i+1][0]:
						all_matches.append(value)


		for itemer in all_matches:
			temp = []
			add = []
			for val in list(itemer):
				if val[0] not in temp:
					temp.append(val[0])
				if val[1] not in temp:
					temp.append(val[1])

			for item in temp:
				pt1, pt2 = item
				add.append(pt1 + pt2)
	
			Z = [x for _,x in sorted(zip(add,temp))]

			if Z not in all_final_paths:
				all_final_paths.append(Z)


		for item in all_final_paths:
			short_path_name[new_path] = item
			new_path += 1

	def distance_formula(x1, y1, x2, y2):
		return float(math.sqrt(abs((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))))

	def Calculate_Area_Shapes(path): # take in all paths and return list containing their areas
		data = np.array(path)
		data_transpose = data.T
		x, y = data_transpose
		return (0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1))))

	def returnShape(path, labels):
		# 0 is circle, 1 is triangle, 2 is square 3 is rectangle, 4 is scc, 5 is dot
		distance = []
		area = 0
		
		if(len(labels) == 3):
			return 1

		elif(len(path) == 3 and type(path[0]) == int):
			return 0

		elif(len(path) == 2 and type(path[0]) == int):
			return 5

		elif(len(labels) == 4):
			for i, elem in enumerate(path[:-1]):
				x1, y1 = elem
				x2, y2 = path[i+1]


				distances = distance_formula(x1, y1, x2, y2)
				distance.append(distances)


			area = Calculate_Area_Shapes(path)

			if(all(x == distance[0] for x in distance) and math.sqrt(area) == (distance[0])):
				return 2
			elif (distance[0]*distance[1] == distance[2]*distance[3] and all(x == distance[0] for x in distance) == False):
				return 3
			else:
				return 4

		else:
			return 4


	def Calculate_Centroid_Shapes(path): # take in all paths and return a list containing all the centroids
		
		data = np.array(path[:-1])
		length = data.shape[0]
		sum_x = np.sum(data[:, 0])
		sum_y = np.sum(data[:, 1])
		return sum_x/length, sum_y/length



	def Print_SCC(complete_path_all, path_labels_all):

		string_output_rep = []
		for item in path_labels_all:
			temp = []
			for pt in item:
				temp.append("S"+str(pt))
			string_output_rep.append(temp)


		for i, item in enumerate(complete_path_all):
			if type(item[0]) != int:
				f.write("P" + str(i+1) + " = scc" +  str(tuple(intersperse(item, 0))) +  " = " +  repr(intersperse(string_output_rep[i], '+')).replace(",", "") + '\n')


	def PrintShape(complete_path_all, path_labels_all):

		idx_shape = 1
		idx_circle = 1
		idx_dot = 1
		for i, value in enumerate(complete_path_all):
			if (returnShape(value, path_labels_all[i]) == 1):
				f.write("triangle(P"+str(idx_shape)+")" + '\n')
				idx_shape += 1
			if (returnShape(value, path_labels_all[i]) == 0):
				f.write("circle(C"+str(idx_circle)+")" + '\n')
				idx_circle += 1
			if (returnShape(value, path_labels_all[i]) == 2):
				f.write("square(P"+str(idx_shape)+")" + '\n')
				idx_shape += 1
			if (returnShape(value, path_labels_all[i]) == 3):
				f.write("rectangle(P"+str(idx_shape)+")" + '\n')
				idx_shape += 1
			if (returnShape(value, path_labels_all[i]) == 4):
				f.write("scc(P"+str(idx_shape)+")" + '\n')
				idx_shape += 1
			if (returnShape(value, path_labels_all[i]) == 5):
				f.write("dot(D"+str(idx_dot)+")" + '\n')
				idx_dot += 1

	def ObjectPosition(item, idx, tag):

		#horizontal location
		if item[0] == 50:
			f.write("hloc(" + tag+str(idx) + ', center)' + '\n')
		elif item[0] > 50:
			f.write("hloc(" + tag+str(idx) + ', right)' + '\n')
		elif item[0] < 50:
			f.write("hloc("+ tag+str(idx) + ', left)' + '\n')

		#vertical location
		if item[1] == 50:
			f.write("vloc(" + tag+str(idx) + ', middle)' + '\n')
		if item[1] >  50:
			f.write("vloc(" + tag+str(idx) + ', top)' + '\n')
		if item[1] < 50:
			f.write("vloc(" + tag+str(idx) + ', bottom)' + '\n')


	def PrintObjectPositions(path_centroids, circle_indices, dot_indices):
		shape_idx = 1
		circle_idx = 1
		dot_idx = 1

		for i, item in enumerate(path_centroids):

			if(i in circle_indices):
				ObjectPosition(item, circle_idx , "C")
				circle_idx += 1

			elif(i in dot_indices):
				ObjectPosition(item, dot_idx, "D")
				dot_idx += 1

			else:
				ObjectPosition(item, shape_idx, "P")
				shape_idx += 1

	def InternalRelations(path_centroids, tag1, tag2, i, j):


		if(path_centroids[i][0] > path_centroids[j][0]):
			f.write("right_of("+ tag1 + "," + tag2 + ")" + '\n')

		elif(path_centroids[i][0] < path_centroids[j][0]):
			f.write("left_of(" + tag1 + "," + tag2 + ")" + '\n')


		if(path_centroids[i][1] > path_centroids[j][1]):
			f.write("above(" + tag1 + "," + tag2 + ")" + '\n')

		elif(path_centroids[i][1] < path_centroids[j][1]):
			f.write("below("+ tag1 + "," + tag2 + ")" + '\n')




	def Print_Relative_Inter(multi_line_path, circle_indices, dot_indices, path_centroids, circle_dict, dot_dict, path_areas):
		# print("path centroids is ", path_centroids)

		# print("Multi line path is ", multi_line_path)
		for i in range(len(multi_line_path)):
				
			tag1 = "P" + str(i + 1)

			for j in range(i + 1, len(multi_line_path)):
				
				tag2 = "P" + str(j + 1)
		
				if i not in circle_indices:
					result = set(multi_line_path[i]).intersection(multi_line_path[j])

					if(len(result) != 0 and (i not in dot_indices and j not in dot_indices)): # dots can't overlap with anything

						# if i in dot_indices:
						# 	tag1 = (list(dot_dict.keys())[list(dot_dict.values()).index(multi_line_path[i])])
						# if j in dot_indices:
						# 	tag2 = (list(dot_dict.keys())[list(dot_dict.values()).index(multi_line_path[j])])

						# print("got here!")

						f.write("overlaps("+ tag1 + "," + tag2 + ")" + '\n')

						InternalRelations(path_centroids, tag1, tag2, i, j)

					# did not find overlappint shapes
					else: #if they do not overlap check to see if one shape is inside the other, if not then do the right, left, thing

						if i in dot_indices:
							tag1 = (list(dot_dict.keys())[list(dot_dict.values()).index(multi_line_path[i])])
						if j in dot_indices:
							tag2 = (list(dot_dict.keys())[list(dot_dict.values()).index(multi_line_path[j])])

						if j in circle_indices:
							tag2 = (list(circle_dict.keys())[list(circle_dict.values()).index(multi_line_path[j])])

						route1 = multi_line_path[i]
	
						if(len(multi_line_path[i]) == 2): #it's a dot
							route1 = np.array(multi_line_path[i]).reshape(1,2)

						p1 = mpath.Path(route1)

						point = np.array(path_centroids[j]).reshape(1, 2)
						contains = p1.contains_points(point)

						if len(contains) != 0:
							if(contains[0] == True):
								if(path_areas[i] > path_areas[j]):
									
									f.write("inside("+ tag2 + "," + tag1 + ")" + '\n')
								elif(path_areas[j] > path_areas[i]):
									f.write("inside("+ tag1 + "," + tag2 + ")" + '\n')

								InternalRelations(path_centroids,  tag1, tag2, i, j)

							else:
								InternalRelations(path_centroids, tag1, tag2, i, j)
						
						else: #don't know if I'll ever hit this
							InternalRelations(path_centroids, tag1, tag2, i, j)
				

				else:
					# circle and circle or dot and circle
					if i in dot_indices:
						tag1 = (list(dot_dict.keys())[list(dot_dict.values()).index(multi_line_path[i])])
					if j in dot_indices:
						tag2 = (list(dot_dict.keys())[list(dot_dict.values()).index(multi_line_path[j])])

					if j in circle_indices:
						tag2 = (list(circle_dict.keys())[list(circle_dict.values()).index(multi_line_path[j])])

					if i in circle_indices:
						tag1 = (list(circle_dict.keys())[list(circle_dict.values()).index(multi_line_path[i])])


					x1 = multi_line_path[i][0]
					y1 = multi_line_path[i][1]
					x2 = multi_line_path[j][0]
					y2 = multi_line_path[j][1]

					distance = distance_formula(x1, y1, x2, y2)

					if i in circle_indices and j in circle_indices:

						if(path_areas[i] > path_areas[j]):
							if(distance < multi_line_path[i][2]):
								f.write("inside("+tag2 + "," + tag1 + ")" + '\n')
						if(path_areas[j] > path_areas[i]):
							if(distance < multi_line_path[j][2]):
								f.write("inside("+tag1 + "," + tag2 + ")" + '\n')

					if i in circle_indices and j in dot_indices:

						if(distance < multi_line_path[i][2]):
							f.write("inside("+ tag2 + "," + tag1 + ")" + '\n')


					if i in dot_indices and j in circle_indices:

						if(distance < multi_line_path[j][2]):
							f.write("inside(" + tag1 + "," + tag2 + ")" + '\n')

					InternalRelations(path_centroids, tag1, tag2, i, j)



	alpha = 97
	# print("got here")
	no_scc = False

	if(len(possible_interpretations) == 0):
		no_scc = True

		for item in input_rep:
			# temp = []
			if item[0] == 'circle':
				possible_interpretations.append(item[1])
			if item[0] == 'dot':
				possible_interpretations.append(item[1])
			# possible_interpretations.append(temp)

	
	# print(possible_interpretations)
	#fix this to go through all the possible interpretations you have including just circles & dots
	for file, item in enumerate(possible_interpretations):
		path_name = os.path.basename(file_name)
		path_name = (os.path.splitext(path_name)[0])

		f = open(os.getcwd() + '/' + output_directory + '/' + path_name + chr(alpha) + ".txt", 'w')
		alpha += 1
	
		short_path_name = defaultdict(list)
		full_short_path = []
		all_matches = []

		if(no_scc == False):
			for cycle in item:
				gather_paths(all_matches, path, short_path_name)
	

		complete_path_all = []
		multi_line_path = []
		path_labels_all = []
		# print("got here")
		if(len(short_path_name) != 0):

			# Go through and find the actual path
			for cycle2 in item:
				path2 = points_in_path[cycle2]
				multi_line_path.append(path2)
				complete_path = deepcopy(path2)
				val1 = 0
				val2 = 0
				idx = 0
				for key, value in short_path_name.items():
				
					if (find_sub_list(value, complete_path)) is not None:

						val1, val2 = (find_sub_list(value, complete_path))
						idx = val1 + 1
						del complete_path[idx:val2]
					else:

						for i, pt in enumerate(complete_path):
							for point in value[1:-1]:
								if pt == point:
									complete_path[i] = 0
						complete_path = [y for y in complete_path if y != 0]

						if(complete_path[0] != complete_path[-1]):
							complete_path.append(complete_path[0])

				complete_path_all.append(complete_path)


			#Go through and find the actual paths

			#Go through and add in the names of the sides

				path_labels = np.zeros(len(complete_path)-1)

				for i, point in enumerate(complete_path[:-1]):
					for key, value in short_path_name.items():

						if (((point + complete_path[i + 1]) == (value[0] + value[-1])) or ((complete_path[i+1] + point) == (value[0] + value[-1]))):

							path_labels[i] = key

				for i, point in enumerate(path_labels):
					if point == 0.0:
						for key, value in edge_rep_dict.items():
							if (list(complete_path[i] + complete_path[i + 1]) == value[0] or list(complete_path[i] + complete_path[i + 1]) == value[1]):
								path_labels[i] = key

							elif(list(complete_path[i+1] + complete_path[i]) == value[0] or list(complete_path[i+1] + complete_path[i]) == value[1]):
								path_labels[i] = key


				path_labels_all.append(path_labels)


			#Go through and add in the names of the sides

			#Print out the shorter lines that make longer lines
			shorter_lines = defaultdict(list)
			for key, val in short_path_name.items():
				temp = []
				for i, pt in enumerate(val[:-1]):
					value1 = pt + val[i+1]
					value2 = val[i+1] + pt

					for k, v in edge_rep_dict.items():

						if v[0] == list(value1) or v[1] == list(value1) or v[0] == list(value2) or v[1] == list(value2):
							temp.append(k)
				shorter_lines[key] = temp

			
			for key, val in shorter_lines.items():
				f.write('S'+str(key) + "=" + "line"+ str(tuple(short_path_name[key][0]) + short_path_name[key][-1]) + "= ")
				for i, item in enumerate(val):
					if i != 0:
						f.write(" + ")#
					f.write('S'+str(item))
				f.write('\n')
				f.flush()


			path_areas = []
			path_centroids = []
			for loop in complete_path_all:
				path_areas.append(Calculate_Area_Shapes(loop))
				path_centroids.append(Calculate_Centroid_Shapes(loop))


			# put circles and dots in path

			
			circle_indices = []
			dot_indices = []
			circle_dict = dict()
			dot_dict = dict()
			circle_counter = 1
			dot_counter = 1

			for value in input_rep:
				idx = len(complete_path_all)
				if value[0] == "circle":
					complete_path_all.append(value[1])
					multi_line_path.append(value[1])
					circle_dict["C" + str(circle_counter)] = value[1]
					path_labels_all.append([]) # Just append an empty list because we want sizes to be consistent for complete path and labels
					circle_indices.append(idx)
					path_centroids.append((value[1][0], value[1][1]))
					path_areas.append(math.pi*value[1][2]*value[1][2])
					idx += 1
					circle_counter += 1
				if value[0] == "dot":
					complete_path_all.append(value[1])
					multi_line_path.append(value[1])
					path_labels_all.append([])
					dot_indices.append(idx)
					dot_dict["D" + str(dot_counter)] = value[1]
					path_centroids.append(value[1])
					path_areas.append(1)
					idx += 1
					dot_counter += 1



			# put circles and dots in path


			Print_SCC(complete_path_all, path_labels_all)

			PrintShape(complete_path_all, path_labels_all)

			PrintObjectPositions(path_centroids, circle_indices, dot_indices)

			Print_Relative_Inter(multi_line_path, circle_indices, dot_indices, path_centroids, circle_dict, dot_dict, path_areas)



		else:
			path_areas = []
			path_centroids = []
			if(no_scc == False):
				for cycle3 in item:
					path3 = points_in_path[cycle3]
					multi_line_path.append(path3)
					complete_path_all.append(path3)
					labeled_path = []

					labeled_path = np.zeros(len(path3) - 1)

					for i, point in enumerate(path3[:-1]):
						for key, value in edge_rep_dict.items():
							if (list(path3[i] + path3[i + 1]) == value[0] or list(path3[i] + path3[i + 1]) == value[1]):
								labeled_path[i] = key

							elif(list(path3[i+1] + path3[i]) == value[0] or list(path3[i+1] + path3[i]) == value[1]):
								labeled_path[i] = key

					path_labels_all.append(labeled_path)

				
				for loop in complete_path_all:
					path_areas.append(Calculate_Area_Shapes(loop))
					path_centroids.append(Calculate_Centroid_Shapes(loop))
			


			# put circles and dots in graph 

			circle_indices = []
			dot_indices = []
			circle_dict = dict()
			dot_dict = dict()
			circle_counter = 1
			dot_counter = 1

			for value in input_rep:
				idx = len(complete_path_all)
				if value[0] == "circle":
					complete_path_all.append(value[1])
					multi_line_path.append(value[1])
					circle_dict["C" + str(circle_counter)] = value[1]
					path_labels_all.append([]) # Just append an empty list because we want sizes to be consistenf for complete path and labels
					circle_indices.append(idx)
					path_centroids.append((value[1][0], value[1][1]))
					path_areas.append(math.pi*value[1][2]*value[1][2])
					idx += 1
					circle_counter += 1
				if value[0] == "dot":
					complete_path_all.append(value[1])
					multi_line_path.append(value[1])
					path_labels_all.append([])
					dot_indices.append(idx)
					dot_dict["D" + str(dot_counter)] = value[1]
					path_centroids.append(value[1])
					path_areas.append(1)
					idx += 1
					dot_counter += 1


			# put circles and dots in graph

			Print_SCC(complete_path_all, path_labels_all)

			# 0 is circle, 1 is triangle, 2 is square 3 is rectangle, 4 is scc, 5 is dot
			PrintShape(complete_path_all, path_labels_all)
			

			PrintObjectPositions(path_centroids, circle_indices, dot_indices)

			# print("got here")
			Print_Relative_Inter(multi_line_path, circle_indices, dot_indices, path_centroids, circle_dict, dot_dict, path_areas)


		f.close()


if __name__ == '__main__':
	main()