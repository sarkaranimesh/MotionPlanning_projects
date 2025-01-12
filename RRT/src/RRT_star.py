import random
import math
import pygame

class RRTMap:
    def __init__(self, start, goal, MapDimensions, obsdim, obsnum):
        self.start = start
        self.goal = goal
        self.MapDimensions = MapDimensions
        self.Maph, self.Mapw = self.MapDimensions
        
        # window settings
        self.MapWindowName = 'RRT Path Planning'
        pygame.display.set_caption(self.MapWindowName)
        self.map = pygame.display.set_mode((self.Mapw, self.Maph))
        self.map.fill((255, 255, 255))  # Fill the map with white color
        self.nodeRad = 2
        self.nodeThickness = 0
        self.edgeThickness = 1
        self.obstacles = []
        self.obsdim = obsdim
        self.obsNumber = obsnum
        # colors
        self.grey = (70, 70, 70)
        self.Blue = (0, 0, 255)
        self.Green = (0, 255, 0)
        self.Red = (255, 0, 0)
        self.White = (255, 255, 255)

    def drawMap(self, obstacles):
        # Draw the start and goal nodes
        pygame.draw.circle(self.map, self.Green, self.start, self.nodeRad + 5, 0)
        pygame.draw.circle(self.map, self.Red, self.goal, self.nodeRad + 20, 1)

        # Draw the obstacles
        self.drawObs(obstacles)

    def drawObs(self, obstacles):
        # Draw each obstacle as a rectangle
        obstaclesList = obstacles.copy()
        while len(obstaclesList) > 0:
            obstacle = obstaclesList.pop(0)
            pygame.draw.rect(self.map, self.grey, obstacle)

    def drawPath(self, path):
        # Draw the path as a series of red circles
        for node in path:
            pygame.draw.circle(self.map, self.Red, node, self.nodeRad + 3, 0)

class RRTGraph:
    def __init__(self, start, goal, MapDimensions, obsdim, obsnum):
        (x, y) = start
        self.start = start
        self.goal = goal
        self.goalFlag = False
        self.MapDimensions = MapDimensions
        self.Maph, self.Mapw = self.MapDimensions
        self.x = []
        self.y = []
        self.parent = []
        self.cost = []  # Add cost list

        # Initialize the tree with the start node
        self.x.append(x)
        self.y.append(y)
        self.parent.append(0)
        self.cost.append(0)  # Initialize cost for the start node

        # Initialize obstacles
        self.obstacles = []
        self.obsdim = obsdim
        self.obsNumber = obsnum

        # Path-related attributes
        self.goalstate = None
        self.path = []

    def makeRandomRect(self):
        # Generate random coordinates for the upper corner of a rectangle
        upperconrnerx = int(random.uniform(0, self.Mapw - self.obsdim))
        upperconrnery = int(random.uniform(0, self.Maph - self.obsdim))
        return (upperconrnerx, upperconrnery)

    def makeobs(self):
        # Create obstacles and ensure they do not overlap
        obs = []
        max_iter = 1000
        for i in range(0, self.obsNumber):
            rectang = None
            for j in range(max_iter):
                upper = self.makeRandomRect()
                rectang = pygame.Rect(upper, (self.obsdim, self.obsdim))

                # Check collision with existing obstacles only (not start/goal)
                if any(rectang.colliderect(existing_obstacle) for existing_obstacle in obs):
                    continue  # Skip to next iteration if collides with existing obstacles

                obs.append(rectang)  # Add valid obstacle to the list
                break
            if j == max_iter - 1:
                print("Unable to place all obstacles")
                break

        self.obstacles = obs.copy()
        return obs

    def add_node(self, n, x, y):
        self.x.append(x)
        self.y.append(y)
        self.cost.append(float('inf'))  # Initialize cost as infinity

    def remove_node(self, n):
        # Remove a node from the tree
        self.x.pop(n)
        self.y.pop(n)

    def add_edge(self, parent, child):
        # Add an edge between two nodes
        self.parent.append(parent)
        self.cost[child] = self.cost[parent] + self.distance(parent, child)  # Update cost

    def remove_edge(self, n):
        # Remove an edge from the tree
        self.parent.pop(n)
    
    def rewire(self, new_node):
        # Ensure new_node is valid
        if new_node >= len(self.x):
            return
            
        for i in range(len(self.x)):
            # Skip invalid connections
            if i == new_node or i >= len(self.cost) or new_node >= len(self.cost):
                continue
                
            dist = self.distance(i, new_node)
            if dist < 35:
                new_cost = self.cost[new_node] + dist
                if new_cost < self.cost[i]:
                    if not self.crossObstacles(self.x[new_node], self.x[i], 
                                             self.y[new_node], self.y[i]):
                        self.parent[i] = new_node
                        self.cost[i] = new_cost

    def number_of_nodes(self):
        # Return the number of nodes in the tree
        return len(self.x)

    def distance(self, n1, n2):
        # Add bounds checking
        if n1 >= len(self.x) or n2 >= len(self.x):
            return float('inf')
        
        point1 = (self.x[n1], self.y[n1])
        point2 = (self.x[n2], self.y[n2])
        sqr_diff = [(x1 - x2) ** 2 for x1, x2 in zip(point1, point2)]
        sum_of_sqediff = sum(sqr_diff)
        return math.sqrt(sum_of_sqediff)

    def sample_envir(self):
        # Sample a random point in the environment
        x = int(random.uniform(0, self.Mapw))
        y = int(random.uniform(0, self.Maph))
        return x, y

    def nearest(self, n):
        # Find the nearest node to the given node
        dmin = self.distance(0, n)
        nnear = 0
        for i in range(0, n):
            if self.distance(i, n) < dmin:
                dmin = self.distance(i, n)
                nnear = i
        return nnear

    def isFree(self):
        # Check if the last added node is free from obstacles
        n = self.number_of_nodes() - 1
        (x, y) = (self.x[n], self.y[n])
        obs = self.obstacles.copy()
        while len(obs) > 0:
            rectang = obs.pop(0)
            if rectang.collidepoint(x, y):
                self.remove_node(n)
                return False
        return True

    def crossObstacles(self, x1, x2, y1, y2):
        # Check if the line between two points crosses any obstacles
        obs = self.obstacles.copy()
        while len(obs) > 0:
            rectang = obs.pop(0)
            for i in range(0, 201):
                u = i / 200
                x = x1 * u + x2 * (1 - u)
                y = y1 * u + y2 * (1 - u)
                if rectang.collidepoint(x, y):
                    return True
        return False

    def connect(self, n1, n2):
        # Connect two nodes if the path between them is free from obstacles
        (x1, y1) = (self.x[n1], self.y[n1])
        (x2, y2) = (self.x[n2], self.y[n2])

        if self.crossObstacles(x1, x2, y1, y2):
            self.remove_node(n2)
            return False
        else:
            self.add_edge(n1, n2)
            return True

    def step(self, nnear, nrand, dmax=35):
        # Move from the nearest node towards the random node by a maximum distance
        d = self.distance(nnear, nrand)
        if d > dmax:
            u = dmax / d
            (xnear, ynear) = (self.x[nnear], self.y[nnear])
            (xrand, yrand) = (self.x[nrand], self.y[nrand])
            (px, py) = (xrand - xnear, yrand - ynear)
            theta = math.atan2(py, px)
            x, y = (int(xnear + dmax * math.cos(theta)), int(ynear + dmax * math.sin(theta)))
            self.remove_node(nrand)
            if abs(x - self.goal[0]) < dmax and abs(y - self.goal[1]) < dmax:
                self.add_node(nrand, self.goal[0], self.goal[1])
                self.goalstate = nrand
                self.goalFlag = True
            else:
                self.add_node(nrand, x, y)

    def expand(self):
        n = self.number_of_nodes()
        x, y = self.sample_envir()
        self.add_node(n, x, y)
        if self.isFree():
            xnearest = self.nearest(n)
            self.step(xnearest, n)
            self.connect(xnearest, n)
            self.rewire(n)  # Call rewire
        return self.x, self.y, self.parent

    def bias(self, ngoal):
        # Bias the tree expansion towards the goal
        n = self.number_of_nodes()
        self.add_node(n, ngoal[0], ngoal[1])
        nnear = self.nearest(n)
        self.step(nnear, n)
        self.connect(nnear, n)
        self.rewire(n)
        return self.x, self.y, self.parent

    def path_to_goal(self):
        # Find the path to the goal if it exists
        if self.goalFlag and self.goalstate is not None and self.goalstate < len(self.parent):
            self.path = []
            self.path.append(self.goalstate)
            newpos = self.parent[self.goalstate]
            while newpos != 0:
                self.path.append(newpos)
                newpos = self.parent[newpos]
            self.path.append(0)
            return True
        return False

    def getPathCord(self):
        # Get the coordinates of the path to the goal
        pathCoords = []
        for node in self.path:
            x, y = (self.x[node], self.y[node])
            pathCoords.append((x, y))
        return pathCoords

    def cost(self):
        # Placeholder for cost calculation method
        pass