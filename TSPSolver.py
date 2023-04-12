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
		time spent to find solution, number of permutations tried during search, the
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
	# TODO crashes lol
	def fancy( self,time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		count = 0
		bssf = None
		start_time = time.time()
		# variables for ant colony algorithm
		numAnts = 20
		numBestAnts = 0
		decayRate = 0.5
		bestPathCost = 0
		worstPathCost = 0
		paths = []

		# Build inital distanceMatrix
		distanceMatrix = np.full((len(cities), len(cities)), math.inf)
		for i in range(len(distanceMatrix)):
			for j in range(len(distanceMatrix)):
				if i != j:
					distanceMatrix[i][j] = cities[i].costTo(cities[j])

		# Build initial pheromoneMatrix
		pheromoneMatrix = np.full((len(cities), len(cities)), 1)


		# TODO for loop (timelimit)
		# TODO send out one group per loop

		while time.time() - start_time < time_allowance:
			# TODO for loop for decrementing pheromoneMatrix by decayRate
			#  loop n ants going out
			antPath = []
			for a in range(numAnts):
				# for loop for number of cities
				currentNode = 0
				for c in range(ncities):
					# TODO ant path selection algorithm (probably the toughest part of this whole project)
					# create probability array with tuple (probability, destination)
					probArray = []

					for i in range(len(distanceMatrix)):
						denominator = 0
						if distanceMatrix[currentNode][i] != math.inf:
							for j in range(len(distanceMatrix)):
								if distanceMatrix[currentNode][j] != math.inf:
									denominator = denominator + pheromoneMatrix[currentNode][j] * (1 // distanceMatrix[currentNode][j])
							numerator = pheromoneMatrix[currentNode][i] * (1 // distanceMatrix[currentNode][i])
							probArray.append(((numerator // denominator), i))
						else:
							probArray.append((0, i))
					# sort probArray backwards for roulette wheel technique
					probArray.sort(key=lambda x: x[0], reverse=True)
					rouletteWheel = []
					for p in range(len(probArray)):
						probRouletteNum = 0
						for q in range(p, len(probArray)):
							probRouletteNum = probRouletteNum + probArray[q][0]
						rouletteWheel.append(probRouletteNum)

					# get random path
					probWinner = round(random.uniform(0, 1), 2)

					for r in range(len(rouletteWheel)):
						if rouletteWheel[r] >= probWinner and rouletteWheel[r + 1] <= probWinner:
							antPath.append((currentNode, r))
							currentNode = r






		# TODO keep a sorted list of all the paths

		# TODO handle pheromones
		#  add pheromones to best r trails (smallest = best)
		#  decrement slightly pheromones on all trails
		#  decrement again k worst paths (from this group)

		end_time = time.time()
		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results
