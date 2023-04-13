#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT6':
	from PyQt6.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))


import time
import numpy as np
from TSPClasses import *
import heapq
import itertools


class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution,
		time spent to find solution, number of mutations tried during search, the
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def greedy( self,time_allowance=60.0 ):
		pass



	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints:
		max queue size, total number of states created, and number of pruned states.</returns>
	'''

	def branchAndBound( self, time_allowance=60.0 ):
		pass



	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found during search, the
		best solution found.  You may use the other three field however you like.
		algorithm</returns>
	'''
	def fancy( self,time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		bestTime = 0.0
		start_time = time.time()
		# variables for ant colony algorithm
		numAnts = 20
		numBestAnts = 0
		decayRate = 0.9
		bestPathCost = 0
		worstPathCost = 0
		# paths tuple (path, cost, time)
		paths = []
		# iterations
		iterations = 0

		#initial bssf
		while not foundTour:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True

		# Build initial distanceMatrix
		distanceMatrix = np.full((len(cities), len(cities)), math.inf)
		for i in range(len(distanceMatrix)):
			for j in range(len(distanceMatrix)):
				if i != j:
					distanceMatrix[i][j] = cities[i].costTo(cities[j])

		# Build initial pheromoneMatrix.
		# FIXME: change this to zero
		pheromoneMatrix = np.full((len(cities), len(cities)), 1.0)

		while time.time() - start_time < time_allowance:
			iterations += 1
			# loop n ants going out
			for a in range(numAnts):
				antPath = []
				# for loop for number of cities
				currentNode = 0
				visitedNodes = [0]
				for c in range(ncities):
					# ant path selection algorithm
					# create probability array with tuple (probability, destination)
					probArray = []

					for i in range(len(distanceMatrix)):
						denominator = 0.0
						if distanceMatrix[currentNode][i] != math.inf and i not in visitedNodes:
							for j in range(len(distanceMatrix)):
								if distanceMatrix[currentNode][j] != math.inf and j not in visitedNodes:
									dist = 1 / distanceMatrix[currentNode][j]
									pher = pheromoneMatrix[currentNode][j] * dist
									denominator = (denominator + pher)
							numerator = pheromoneMatrix[currentNode][i] * (1 / distanceMatrix[currentNode][i])
							probability = numerator / denominator
							probArray.append((round(probability, 2), i))
						else:
							probArray.append((0, i))
					# sort probArray backwards for roulette wheel technique
					probArray.sort(key=lambda x: x[0], reverse=True)
					rouletteWheel = []
					for p in range(len(probArray)):
						probRouletteNum = 0
						for q in range(p, len(probArray)):
							probRouletteNum = probRouletteNum + probArray[q][0]
						rouletteWheel.append((round(probRouletteNum, 2), probArray[p][1]))

					# get random path
					probWinner = round(random.uniform(0, 1), 2)

					for r in range(len(rouletteWheel) - 1):
						if rouletteWheel[r][0] >= probWinner >= rouletteWheel[r + 1][0]:
							antPath.append((currentNode, rouletteWheel[r][1]))
							currentNode = rouletteWheel[r][1]
							visitedNodes.append(currentNode)
							break
				# find path distance and add to paths
				distance = 0
				for i in range(ncities - 1):
					distance = distance + cities[antPath[i][0]].costTo(cities[antPath[i][1]])
				distance = distance + cities[antPath[len(antPath)-1][1]].costTo(cities[antPath[0][0]])
				paths.append((antPath, distance, time.time() - start_time))

			# keep a sorted list of all the paths
			paths.sort(key=lambda x: x[1], reverse=False)

			# get the current bestPathCost and worstPathCost
			bestPathCost = paths[0][1]
			# if better than current bssf, change it to new bestPath
			if bestPathCost < bssf.cost:
				print(f'current bssf: {bssf.cost}')
				print(f' best path cost: {bestPathCost}')
				route = [cities[0]]
				for i in paths[0][0]:
					route.append(cities[i[1]])
				bssf = TSPSolution(route)
				bestTime = paths[0][2]
				count += 1
			worstPathCost = paths[len(paths) - 1][1]

			# Add pheromones to best trails (smallest = best)
			bestPath = paths[0][0]
			for i in range(ncities - 1):  # Num paths in one route is n - 1
				# Adds this percentage of itself back to itself. So somewhere below *2.
				percentageToAdd = 1 - (bestPathCost / worstPathCost)
				start_city = bestPath[i][0]
				end_city = bestPath[i][1]
				pheromoneMatrix[start_city, end_city] += 1  # Insures we get out of zero.
				pheromoneMatrix[start_city, end_city] += (pheromoneMatrix[start_city, end_city] * percentageToAdd)

			# Worst trail pheromones get decayed
			worstPath = paths[len(paths) - 1][0]
			for i in range(ncities - 1):  # Num paths in one route is n - 1
				start_city = worstPath[i][0]
				end_city = worstPath[i][1]
				pheromoneMatrix[start_city, end_city] *= decayRate

			# Decrement pheromoneMatrix by decayRate
			for i in range(len(pheromoneMatrix)):
				for j in range(len(pheromoneMatrix)):
					pheromoneMatrix[i][j] *= decayRate

		# TODO if best path for curr group is better than the current bssf replace it.
		results['cost'] = bssf.cost
		results['time'] = bestTime
		results['count'] = count
		results['soln'] = bssf
		print(iterations)
		return results
