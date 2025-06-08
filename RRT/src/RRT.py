import pygame
from RRTbasePy import RRTGraph
from RRTbasePy import RRTMap
# from RRT_star import RRTGraph
# from RRT_star import RRTMap
import time
import math

def calculate_path_length(path_coords):
    length = 0
    for i in range(len(path_coords)-1):
        x1, y1 = path_coords[i]
        x2, y2 = path_coords[i+1]
        length += math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return length

def main():
    dimensions = (600,1000)
    start = (10,50)
    goal = (850,500)
    obsdim = 30
    obsnum = 50
    iteration = 0
    start_time = time.time()
    pygame.init()
    

    map = RRTMap(start,goal,dimensions,obsdim,obsnum)
    graph = RRTGraph(start,goal,dimensions,obsdim,obsnum)

    obstacles = graph.makeobs()


    map.drawMap(obstacles)

    while (not graph.path_to_goal()):
        elapsed = time.time() - start_time
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
 # When goal is reached
    if graph.goalFlag:
        path_coords = graph.getPathCord()
        map.drawPath(path_coords)
        end_time = time.time()
        
        # Calculate metrics
        execution_time = end_time - start_time
        path_length = calculate_path_length(path_coords)
        
        # Display metrics
        print(f"\nMetrics:")
        print(f"Execution Time: {execution_time:.2f} seconds")
        print(f"Path Length: {path_length:.2f} units")
        print(f"Total Iterations: {iteration}")

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
    
