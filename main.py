import random
import csv
import settings
import math
from geopy.distance import vincenty
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class Customer:
    def __init__(self, x, y, distance, duration, id):
        self.x = x
        self.y = y
        self.distance = distance
        self.duration = duration
        self.id = id

    def __repr__(self):
        return 'D:{} ({:.3f},{:.3f})'.format(int(self.distance), self.x, self.y)

    def __lt__(self, other):
        return self.distance < other.distance

    def __cmp__(self, other):
        if hasattr(other, 'distance'):
            return self.distance.__cmp__(other.distance)

class DroneHeuristic:
    depot = [50, 50]
    customers = []
    n_customers = 20
    truck_customers = []
    drone_customers = []
    unreachable_customers = []
    drone_distances = [0]
    drone_range = -1
    model = 2
    drone_speed = 45
    truck_speed = 30
    truck_threshold = 2000
    drone_autonomy = 30
    n_trucks = 1
    n_drones = 1

    randomString = ""

    def __init__(self):
        self.randomString = "INIT"

    def init_customers(self, file_path=None):
        if file_path is None:
            # Generates a set of N random customers
            self.customers = [random.sample(range(90), 2) for x in range(self.n_customers)]
        else:
            dr = []
            with open(file_path, 'r') as f:
                if "MA/" in file_path:
                    settings.euclidean_distance = False
                    reader = csv.reader(f, delimiter=';')
                    for i, row in enumerate(reader):
                        if i == 0: continue  # Skip column titles
                        if (row[2].lower()) == "depot":
                            self.depot[0] = float(row[0])
                            self.depot[1] = float(row[1])
                        else:
                            self.customers.append([float(row[0]), float(row[1])])
                            dr.append(row[2])
                else:
                    settings.euclidean_distance = True
                    reader = csv.reader(f, delimiter=' ')
                    for i, row in enumerate(reader):
                        if i < 6: continue  # Skip column titles
                        if row[0] != "EOF":
                            if i == 6:
                                self.depot[0] = float(row[1])
                                self.depot[1] = float(row[2])
                            else:
                                self.customers.append([float(row[1]), float(row[2])])

    def split_customers(self, customers):

        self.truck_customers.clear()
        self.drone_customers.clear()
        self.drone_distances.clear()
        self.unreachable_customers.clear()

        if self.drone_range < 0:
            self.drone_range = (0.5 * self.drone_autonomy) * (50 * self.drone_speed / 60)
            # Number 50 is a multiplier. Adjusted based on Eucledian problem.

        #
        # print("Drone range: " + str(self.drone_range))
        #

        if self.model == 3:
            # split customers into those reachable by drone and those out of reach
            for i, customer in reversed(list(enumerate(customers))):
                if customer.distance < self.truck_threshold and self.n_trucks > 0:
                    self.truck_customers.append(customer)
                elif customer.distance < self.drone_range and self.n_drones > 0:
                    self.drone_customers.append(customer)
                    self.drone_distances.append(customer.distance)
                else:
                    self.unreachable_customers.append(customer)
        else:
            # split customers into those reachable by drone and those out of reach
            for i, customer in reversed(list(enumerate(customers))):
                if customer.distance < self.drone_range and self.n_drones > 0:
                    self.drone_customers.append(customer)
                    self.drone_distances.append(customer.distance)
                else:
                    if self.model == 1 or self.n_trucks == 0:
                        self.unreachable_customers.append(customer)
                    else:
                        self.truck_customers.append(customer)

    def calculate_customer_distances(self, locations):
        customers = []
        global distance_matrix
        distance_matrix = [[-1 for x in range(len(locations) + 1)] for y in range(len(locations) + 1)]
        distance_matrix[0][0] = 0
        for i in range(len(locations)):
            dis = distance(locations[i], self.depot)
            c = Customer(locations[i][0], locations[i][1], dis, -1, i+1)
            distance_matrix[i+1][0] = dis
            distance_matrix[0][i+1] = dis
            distance_matrix[i+1][i+1] = 0
            customers.append(c)
        customers.sort()

        return customers


def distance(p1, p2, geo=None):
    if settings.euclidean_distance:
        # calculate basic euclidean distance
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    else:
        # calculate the distance in a spheroid in meters
        return vincenty(p1, p2).m


# MAIN CODE

nTruck = 3  # Number of trucks = number of clusters

FILE = 'problems/TSP/kroA100.tsp.txt'

heu = DroneHeuristic()

heu.init_customers(FILE)

print("")
print(" --- DEPOT ---")
print('Depot: ' + str(heu.depot))
print("")

# print('Customer number 1: ' + str(heu.customers[0]))

print(" --- CUSTOMERS ---")
print("Heu customers:")
print(heu.customers)
print("")

customers = heu.calculate_customer_distances(heu.customers)

print("Sorted customers based on distances:")
print(customers)
print("")

heu.split_customers(customers)

print(" --- TRUCK ---")
print("Number of customers served by trucks: " + str(len(heu.truck_customers)))
print('Truck customers: ' + str(heu.truck_customers))
print("")


print(" --- DRONE ---")
print("Drone range: " + str(heu.drone_range))
print("Number of customers served by drones: " + str(len(heu.drone_customers)))
print('Drone customers: ' + str(heu.drone_customers))


# BUILDING LIST OF TUPLE OF CUSTOMERS IN TRUCK

tuple_trucks = []

for i in range(len(heu.truck_customers)):
    #print("X: :" + str(heu.truck_customers[i].x))
    #print("Y: :" + str(heu.truck_customers[i].y))
    tuple_trucks.append((heu.truck_customers[i].x, heu.truck_customers[i].y))

# print('Truck customers 2: ' + str(tuple_trucks))


# KMEANS CLUSTERING

# N CLUSTERS WILL BE BASED ON NUMBER OF TRUCK
km = KMeans(n_clusters=nTruck)

km.fit(tuple_trucks)
x_val = [x[0] for x in tuple_trucks]
y_val = [x[1] for x in tuple_trucks]

plt.scatter(x_val, y_val, c = km.labels_, cmap="rainbow")
plt.show()


print("")
print(" --- CLUSTERS ---")

truckDF = pd.DataFrame(list(tuple_trucks))
truckDF.columns = ['X', 'Y']

truckDF['Cluster'] = km.labels_

print(truckDF)

