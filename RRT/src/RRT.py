import pygame
# from RRTbasePy import RRTGraph
# from RRTbasePy import RRTMap
from RRT_star import RRTGraph
from RRT_star import RRTMap
import time


def main():
    dimensions = (600,1000)
    start = (10,50)
    goal = (600,50)
    obsdim = 30
    obsnum = 50
    iteration = 0
    t1 =time.time()
    pygame.init()

    map = RRTMap(start,goal,dimensions,obsdim,obsnum)
    graph = RRTGraph(start,goal,dimensions,obsdim,obsnum)

    obstacles = graph.makeobs()


    map.drawMap(obstacles)

    while (not graph.path_to_goal()):
        elapsed = time.time() - t1
        if elapsed > 500:  # Timeout after 10 seconds
            print("Timeout: Unable to find path within 10 seconds.")
            return  # Exit the main() function

        if iteration %10 == 0:
            X,Y,parent = graph.bias(goal)
            print(f"Bias Node {iteration}: {X[-1]}, {Y[-1]}")  # Debugging

            pygame.draw.circle(map.map, map.grey,(X[-1],Y[-1]), map.nodeRad + 2,0)
            pygame.draw.line(map.map, map.Blue,(X[-1],Y[-1]), (X[parent[-1]], Y[parent[-1]]),map.edgeThickness)
        else:
            X,Y,parent = graph.expand()
            print(f"Expand Node {iteration}: {X[-1]}, {Y[-1]}")  # Debugging

            pygame.draw.circle(map.map, map.grey,(X[-1],Y[-1]), map.nodeRad + 2,0)
            pygame.draw.line(map.map, map.Blue,(X[-1],Y[-1]), (X[parent[-1]], Y[parent[-1]]),map.edgeThickness)

        if iteration%5 == 0:
            pygame.display.update()
        iteration+=1
    map.drawPath(graph.getPathCord())
    pygame.display.update()
    pygame.event.clear()
    pygame.event.wait(0)



        # pygame.display.update()
    # pygame.event.clear()
    # pygame.event.wait(0)

    # pygame.display.update()
    # pygame.time.delay(10)
    # pygame.quit()

if __name__ == '__main__':
    main()
    
