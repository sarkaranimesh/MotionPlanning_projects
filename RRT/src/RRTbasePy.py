import random
import math
import pygame

class RRTMap:
    def __init__(self,start,goal,MapDimensions,obsdim,obsnum):
        self.start = start
        self.goal = goal
        self.MapDimensions = MapDimensions
        self.Maph , self.Mapw = self.MapDimensions
        
        # window settings
        self.MapWindowName = 'RRT Path Planning'
        pygame.display.set_caption(self.MapWindowName)
        self.map = pygame.display.set_mode((self.Mapw,self.Maph))
        self.map.fill((255,255,255))
        self.nodeRad = 2
        self.nodeThickness = 0
        self.edgeThickness = 1

        self.obstacles = []
        self.obsdim = obsdim
        self.obsNumber = obsnum

        # colors
        self.grey = (70,70,70)
        self.Blue = (0,0,255)
        self.Green = (0,255,0)
        self.Red = (255,0,0)
        self.White = (255,255,255)

    def drawMap(self,obstacles):
       # self.map.fill(self.White)  # White background

        pygame.draw.circle(self.map, self.Green,self.start,self.nodeRad+5,0)
        pygame.draw.circle(self.map, self.Red,self.goal,self.nodeRad+20,1)
        # pygame.draw.circle(self.map,self.grey,self.goal,self.nodeRad+5,0)

        self.drawObs(obstacles)

    def drawObs(self,obstacles):

        obstaclesList = obstacles.copy()
        while(len(obstaclesList)>0):
            obstacle = obstaclesList.pop(0)
            pygame.draw.rect(self.map,self.grey,obstacle)

    def drawPath(self, path):
        for node in path:
            pygame.draw.circle(self.map, self.Red, node, self.nodeRad+3,0)




class RRTGraph:
    def __init__(self,start,goal,MapDimensions,obsdim,obsnum):
        (x,y) = start
        self.start = start
        self.goal = goal
        self.goalFlag = False
        self.MapDimensions =MapDimensions
        self.Maph , self.Mapw = self.MapDimensions
        self.x =[]
        self.y = []
        self.parent = []

        # initialize the tree
        self.x.append(x)
        self.y.append(y)
        self.parent.append(0)

        # the obstacles
        self.obstacles = []
        self.obsdim = obsdim
        self.obsNumber = obsnum

        #path
        self.goalstate = None
        self.path = []

    def makeRandomRect(self):
        upperconrnerx = int(random.uniform(0,self.Mapw - self.obsdim))
        upperconrnery = int(random.uniform(0,self.Maph-self.obsdim))

        return(upperconrnerx,upperconrnery)
    
    def makeobs(self):
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
            if j == max_iter -1:
                print("unable to place all obstacles")
                break


        self.obstacles = obs.copy()
        return obs
    

    def add_node(self, n, x, y):
        if n not in range(len(self.x)):
            self.x.insert(n, x)
            self.y.insert(n, y)


    def remove_node(self, n):
        self.x.pop(n)
        self.y.pop(n)
        # Remove associated edge


    def add_edge(self, parent, child):
        self.parent.insert(child,parent)

    def remove_edge(self,n):
        self.parent.pop(n)

    def number_of_nodes(self):
        return len(self.x)
    
    def distance(self,n1,n2):
        point1 = (self.x[n1] , self.y[n1])
        point2 = (self.x[n2], self.y[n2])
        sqr_diff = [(x1-x2)**2 for x1,x2 in zip(point1,point2)]
        sum_of_sqediff = sum(sqr_diff)
        dist = math.sqrt(sum_of_sqediff)
        return dist
    
    def sample_envir(self):
        x = int(random.uniform(0,self.Mapw))
        y= int(random.uniform(0,self.Maph))
        return x,y  


    def nearest(self,n):
        dmin = self.distance(0,n)
        nnear = 0
        for i in range(0,n):
            if self.distance(i,n) < dmin:
                dmin = self.distance(i,n)
                nnear = i
        return nnear
            
    def isFree(self):
        n= self.number_of_nodes()-1
        (x,y) = (self.x[n],self.y[n])
        obs = self.obstacles.copy()
        while len(obs) > 0:
            rectang=  obs.pop(0)
            if rectang.collidepoint(x,y):
                self.remove_node(n)
                return False
        return True
    def crossObstacles(self,x1,x2,y1,y2):
        obs = self.obstacles.copy()
        while(len(obs)>0):
            rectang = obs.pop(0)
            for i in range(0,201):
                u = i/200
                x = x1*u + x2*(1-u)
                y = y1*u + y2*(1-u)
                if rectang.collidepoint(x,y):
                    return True
                
        return False
    
    def connect(self, n1,n2):
        (x1,y1) = (self.x[n1], self.y[n1])
        (x2,y2) = (self.x[n2],self.y[n2])

        if self.crossObstacles(x1,x2,y1,y2):
            self.remove_node(n2)
            return False
        else:
            self.add_edge(n1,n2)
            return True
        
    def step(self, nnear, nrand, dmax = 35):
        d= self.distance(nnear,nrand)
        if d> dmax:
            u = dmax /d
            (xnear,ynear) = (self.x[nnear], self.y[nnear])
            (xrand, yrand) = (self.x[nrand], self.y[nrand])
            (px,py) = (xrand-xnear , yrand-ynear)
            theta = math.atan2(py,px)
            x,y = (int(xnear + dmax*math.cos(theta)), int(ynear + dmax*math.sin(theta))
                   )
            self.remove_node(nrand)
            if abs(x - self.goal[0]) < dmax and abs(y - self.goal[1]) < dmax:

                self.add_node(nrand,self.goal[0], self.goal[1])
                self.goalstate = nrand
                self.goalFlag = True
            else:
                self.add_node(nrand,x,y)
    def expand(self):
        n = self.number_of_nodes()
        x,y = self.sample_envir()
        self.add_node(n,x,y)
        if self.isFree():
            xnearest = self.nearest(n)
            self.step(xnearest,n)
            self.connect(xnearest,n)
        return self.x, self.y, self.parent

    def bias(self, ngoal):
        n = self.number_of_nodes()
        self.add_node(n,ngoal[0], ngoal[1])
        nnear = self.nearest(n)
        self.step(nnear, n)
        self.connect(nnear,n)

        return self.x, self.y, self.parent
    
    # def path_to_goal(self):
    #     if self.goalFlag:
    #         self.path = []
    #         self.path.append(self.goalstate)
    #         newpos = self.parent[self.goalstate]
    #         while (newpos != 0):
    #             self.path.append(newpos)
    #             newpos = self.parent[newpos]
    #         self.path.append(0)
    #     return self.goalFlag
    
    def path_to_goal(self):
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
        pathCoords = []
        for node in self.path:
            x,y = (self.x[node], self.y[node])
            pathCoords.append((x,y))
        return pathCoords
    

    def cost(self):
        pass


            

        
    








        



