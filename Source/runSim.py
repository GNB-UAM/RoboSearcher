#!/usr/bin/python
# Evaluates the characteristics of a Levy walk (https://www.nature.com/articles/s41598-021-03826-3)
# Compares its exploratory performance for distinct strategies to handle walls and obstacles
# Carlos Garcia-Saura, Grupo de Neurocomputacion, Universidad Autonoma de Madrid 2017-2022

# Begin modules
import psutil
from numpy import *
import matplotlib.pyplot as plt
import os
from math import sin, cos, hypot, atan2, sqrt, pow, floor
import time
#from eyediagram.core import grid_count
# End modules

import simulevy

twoPi = 2.*pi
halfPi = pi/2.
def angle_difference(a, b):
    arg = (a-b) % twoPi
    if arg < 0: arg += twoPi
    if arg > pi: arg -= twoPi
    return -arg


robot = simulevy.Robot()


# DEFINE THE MAPS

scenario = {}
scenario['Plain'] = []
scenario['Craters'] = [(robot.world_radius/2., 0., robot.world_radius/2.6), # (x, y, radius)
                          (0., robot.world_radius/2., robot.world_radius/3.5)]

big_circle_radius = 100000 # straight walls are implemented as a very large radius
triangle_size = 221.43
separation = sqrt(((triangle_size+big_circle_radius)**2)/5.)
scenario['Triangle'] = [(-big_circle_radius-triangle_size-80, 0., big_circle_radius), # (x, y, radius)
                        (separation, 2*separation, big_circle_radius),
                        (separation, -2*separation, big_circle_radius)]

scaling = (500./350.) # scale the map size while maintaining the proportions
passage_size = 50*scaling
scenario['Corridor'] = [(0, big_circle_radius+passage_size, big_circle_radius), # (x, y, radius)
                        (0, -big_circle_radius-passage_size, big_circle_radius),
                        (200*scaling, 2*robot.radius*3, passage_size),
                        (-100*scaling, -2*robot.radius*3, passage_size)]


passage_size = 3*robot.radius
scenario['narrow'] = [(0, robot.world_radius+passage_size/2., robot.world_radius), # (x, y, radius)
                      (0, -robot.world_radius-passage_size/2., robot.world_radius)]


random.seed(0) # optional, fix the seed if repeatability is desired

def get_random_obstacle(world_radius=robot.world_radius, size=2*robot.radius*scaling, center_margin=robot.radius*4):
    x = 0
    y = 0
    distance_from_center = 0
    while distance_from_center > robot.world_radius or distance_from_center < center_margin:
        x = random.uniform(-world_radius, world_radius)
        y = random.uniform(-world_radius, world_radius)
        distance_from_center = hypot(x,y)
    return (x, y, size)

scenario['Dense forest'] = []
for i in range(100):
    scenario['Dense forest'].append(get_random_obstacle())

scenario['Sparse forest'] = []
for i in range(50):
    scenario['Sparse forest'].append(get_random_obstacle())


# Set the desired scenario
scenario_name = 'Craters'

robot.setWorldParameters(obstacles=scenario[scenario_name])
print("simulating")


# DEFINE THE BOUNCE BEHAVIORS

def bounce_specularReflection(wall_angle):
    incidence_angle = angle_difference(robot.a,wall_angle-pi)
    incidence_angle_degrees = rad2deg(incidence_angle)
    if abs(incidence_angle_degrees) >= 90:
        return wall_angle
    return wall_angle + incidence_angle

random_bounce_range = deg2rad(180)
def bounce_random(wall_angle):
    return wall_angle + random.uniform(-random_bounce_range/2., random_bounce_range/2.)

def bounce_wallFollow(wall_angle):
    incidence_angle = angle_difference(robot.a,wall_angle-pi)
    if incidence_angle > 0: return wall_angle + halfPi - 0.01
    return wall_angle - halfPi + 0.01

def random_rotation():
    return random.random()*twoPi




# DEFINE THE SEARCH STRATEGIES
robot_speed = 9.36 # [cm/s]
max_search_distance = 80*60*robot_speed # [cm] (80 min battery duration at the given velocity)
max_levy_step = max_search_distance*0.1 # 10% of max battery duration
min_step_size = 21.


def pureBrownian(wall_detected, wall_dist, wall_angle):
    robot.remaining_distance -= robot.sim_step
    if wall_detected: return bounce_random(wall_angle)
    if robot.remaining_distance <= 0:
        robot.remaining_distance = min_step_size
        return robot.a + random_rotation()
    return robot.a

def brownianMemory(wall_detected, wall_dist, wall_angle):
    robot.remaining_distance -= robot.sim_step
    if wall_detected: return bounce_random(wall_angle)
    if robot.remaining_distance <= 0:
        robot.remaining_distance = min_step_size
        
        new_angle = 0
        trials = 0
        while True:
            new_angle = robot.a + random_rotation()
            if robot.checkVisited(new_angle,robot.remaining_distance) and random.random() < 0.95:
                trials += 1
                continue
            else:
                break
        return new_angle
    return robot.a


def pureSpecular(wall_detected, wall_dist, wall_angle):
    if wall_detected: return bounce_specularReflection(wall_angle)
    return robot.a

def pureRandomBounce(wall_detected, wall_dist, wall_angle):
    if wall_detected: return bounce_random(wall_angle)
    return robot.a

levy_alpha = 1.5
levy_exponent = -1./levy_alpha

def levySpecular(wall_detected, wall_dist, wall_angle):
    robot.remaining_distance -= robot.sim_step
    if wall_detected:
        return bounce_specularReflection(wall_angle)
    if robot.remaining_distance <= 0:
        robot.remaining_distance = pow(random.random(),levy_exponent) * min_step_size
        if robot.remaining_distance > max_levy_step:
            robot.remaining_distance = max_levy_step
        return robot.a + random_rotation()
    return robot.a

def levySpecularMemory(wall_detected, wall_dist, wall_angle):
    robot.remaining_distance -= robot.sim_step
    if wall_detected:
        return bounce_specularReflection(wall_angle)
    if robot.remaining_distance <= 0:
        robot.remaining_distance = pow(random.random(),levy_exponent) * min_step_size
        if robot.remaining_distance > max_levy_step:
            robot.remaining_distance = max_levy_step
        new_angle = 0
        trials = 0
        while True:
            new_angle = robot.a + random_rotation()
            if robot.checkVisited(new_angle,robot.remaining_distance) and random.random() < 0.95:
                trials += 1
                continue
            else:
                break
        return new_angle
    return robot.a

def levyRandomBounce(wall_detected, wall_dist, wall_angle):
    robot.remaining_distance -= robot.sim_step
    if wall_detected: return bounce_random(wall_angle)
    if robot.remaining_distance <= 0:
        robot.remaining_distance = pow(random.random(),levy_exponent) * min_step_size
        if robot.remaining_distance > max_levy_step:
            robot.remaining_distance = max_levy_step
        return robot.a + random.random()*twoPi
    return robot.a

def levyWallFollow(wall_detected, wall_dist, wall_angle):
    robot.remaining_distance -= robot.sim_step
    if wall_detected: return bounce_wallFollow(wall_angle)
    if robot.remaining_distance <= 0:
        robot.remaining_distance = pow(random.random(),levy_exponent) * min_step_size
        if robot.remaining_distance > max_levy_step:
            robot.remaining_distance = max_levy_step
        return robot.a + random.random()*twoPi
    return robot.a

def levyRethrow(wall_detected, wall_dist, wall_angle):
    robot.remaining_distance -= robot.sim_step
    if wall_detected or robot.remaining_distance <= 0:
        robot.remaining_distance = pow(random.random(),levy_exponent) * min_step_size
        if robot.remaining_distance > max_levy_step:
            robot.remaining_distance = max_levy_step
        if wall_detected: return bounce_random(wall_angle)
        return robot.a + random.random()*twoPi
    return robot.a


strategies = {}
strategies['Brownian'] = pureBrownian
strategies['Brownian, memory'] = brownianMemory
strategies['Ballistic, mirror bounce'] = pureSpecular
strategies['Ballistic, random bounce'] = pureRandomBounce
strategies['Lévy, mirror bounce'] = levySpecular
strategies['Lévy, mirror & memory'] = levySpecularMemory
strategies['Lévy, random bounce'] = levyRandomBounce
strategies['Lévy, wall follow'] = levyWallFollow
strategies['Lévy, recast bounce'] = levyRethrow


# Initial conditions
random_angle = random.random()*twoPi
robot.setPos(a=random_angle)

# Set the desired strategy:
strategy_name = 'Lévy, mirror bounce'
#strategy_name = 'Brownian'
#strategy_name = 'Brownian, memory'
#strategy_name = 'Ballistic, random bounce'
#strategy_name = 'Ballistic, mirror bounce'
robot.setStrategy(strategies[strategy_name])

robot.plot_title = strategy_name

robot.animate()
exit(0)

# Example to export multiple simulations
# It is optimized for our cluster, you may need to adapt it to your setup
tasks = [('Plain','Lévy, recast bounce'),
         ('Plain','Lévy, mirror bounce'),
         ('Plain','Lévy, random bounce'),
         ('Plain','Lévy, recast bounce'),
         ('Craters','Ballistic, mirror bounce'),
         ('Craters','Lévy, random bounce'),
         ('Dense forest','Lévy, recast bounce'),
         ('Dense forest','Ballistic, mirror bounce')]

def cumuProb(rt):
    #rt = np.diff(rt) # implicit
    nt = len(rt)
    if nt <= 0: return
    rt_sort = sort(rt)
    p = repeat(1./nt, nt)
    p_cumsum = cumsum(p) #reverse(cumsum(p)) # optional

    # Resample to reduce number of points
    x = [rt_sort[0]]
    y = [p_cumsum[0]]
    prevY = p_cumsum[0]
    for i in range(nt):
        if prevY-p_cumsum[i] < 0.5*p_cumsum[i]: # reduce the 0.5 if more points are needed
            x.append(rt_sort[i])
            y.append(p_cumsum[i])
            prevY = p_cumsum[i]
    return array([x,y])


# Visit sequence simulation
for (scenario_name, strategy_name) in tasks:
    pid = os.fork()
    if pid == 0: # child
        random.seed(0)
        print("Simulating "+scenario_name+"-"+strategy_name)
        robot = simulevy.Robot()
        a = random.random()*twoPi
        x = 200*random.random() - 100
        y = 200*random.random() - 100
        robot.setPos(x,y,a)
        robot.setWorldParameters(obstacles=scenario[scenario_name])
        robot.setStrategy(strategies[strategy_name])
        process = psutil.Process()
        while robot.total_steps < 1e8: # in this case the terminating condition is the number of steps
            robot.steps(100000)
            ram_usage_mb = int(process.memory_info().rss/1000000)
            print("Simulating "+scenario_name+"-"+strategy_name+" Steps: "+str(robot.total_steps)+" Ram: "+str(ram_usage_mb))
            #if ram_usage_mb > 40*1000: break # the simulations are intensive on RAM, remember to keep an eye on it
        output_folder = "summarized_data"
        os.makedirs(output_folder, exist_ok=True)
        fname = output_folder+"/"+scenario_name+"-"+strategy_name
        
        print("Processing...")
        data = {}
        matsize = len(robot.cell_visits[0,:])
        data['matsize'] = matsize
        data['total_steps'] = robot.total_steps
        data['sequences_sampled'] = robot.cell_visits[::10,::10]
        
        # Examples of possible exports
        data['n_visits'] = zeros((matsize,matsize), dtype=uint32)
        data['data_valid_mask'] = zeros((matsize,matsize), dtype=bool)
        #data['std'] = zeros((matsize,matsize), dtype=float)
        #data['var'] = zeros((matsize,matsize), dtype=float)
        #data['mean'] = zeros((matsize,matsize), dtype=float)
        #data['median'] = zeros((matsize,matsize), dtype=float)
        #data['mode'] = zeros((matsize,matsize), dtype=uint32)
        #data['last_timestamp'] = zeros((matsize,matsize), dtype=uint32)
        #data['percentiles'] = zeros((matsize,matsize,4), dtype=float)
        
        resolution = 500
        seq = cumuProb(diff(robot.cell_visits[500,500]))
        equispaced = interp(logspace(1,6), seq[0], seq[1])
        h = grid_count(equispaced, len(equispaced)-1, size=(resolution,resolution), fuzz=True, bounds=(0,1))
        h = 0
        
        for i in range(matsize):
            print(scenario_name +"-"+strategy_name + str(int((i*100)/matsize)))
            for j in range(matsize):
                cells = robot.cell_visits[i,j]
                if len(cells) <= 1000:
                    continue
                data['data_valid_mask'][i,j] = True
                time_differences = diff(cells)
                data['n_visits'][i,j] = len(time_differences)
                
                seq = cumuProb(time_differences)
                equispaced = interp(logspace(1,6), seq[0], seq[1])
                h += grid_count(equispaced, len(equispaced)-1, size=(resolution,resolution), fuzz=True, bounds=(0,1))
                
                # Examples of possible exports. Update accordingly.
                #data['last_timestamp'][i,j] = cells[i,j][-1]
                #data['std'][i,j] = std(time_differences)
                #data['var'][i,j] = var(time_differences)
                #data['mean'][i,j] = mean(time_differences)
                #data['median'][i,j] = median(time_differences)
                #data['mode'][i,j] = stats.mode(time_differences).mode[0]
                #data['percentiles'][i,j,:] = percentile(time_differences,[5,25,75,95])        
        data['cumulative_probabilities_histogram'] = h
        
        save(fname+"_summary.npy", data)
        # This next step takes a lot of space; we recommend to only export the needed results and/or a subset of map cells.
        #for i in range(robot.matsize):
        #    for j in range(robot.matsize):
        #        robot.cell_visits[i,j] = array(robot.cell_visits[i,j], dtype=uint)
        #save(fname+"_cells.npy", robot.cell_visits)
        break

exit(0)



# Example to aggregate many short runs

simlen = 2000

for scenario_name in scenario.keys():
    pid = 0 #os.fork() # adapt to your multicore cluster
    if pid == 0: # child
        for strategy_name in strategies.keys():
            pid = 0 #os.fork() # adapt to your multicore cluster
            if pid == 0: # child
                random.seed(0) # set seed so all simulations start in the same conditions
                res = []
                for i in range(1000):
                    print("Simulating "+scenario_name+"-"+strategy_name+" "+str(i))
                    robot = simulevy.Robot()
                    # Initial conditions
                    a = random.random()*twoPi
                    x = 200*random.random() - 100
                    y = 200*random.random() - 100
                    robot.setPos(x,y,a)
                    robot.setWorldParameters(obstacles=scenario[scenario_name])
                    
                    # Useful to export the map shape
                    os.makedirs("obstacle_mask", exist_ok=True)
                    save("obstacle_mask/"+scenario_name+"_mask", robot.pos_histogram_mask)
                    
                    robot.setStrategy(strategies[strategy_name])
                    series = zeros(simlen,dtype=int)
                    for j in range(simlen):
                        robot.steps(100)
                        series[j] = count_nonzero(robot.pos_histogram > 0)
                    res.append(series)
                os.makedirs("resultsequences", exist_ok=True)
                save("resultsequences/"+scenario_name+"-"+strategy_name, res)
                #break # uncomment if using fork() above
        #break # uncomment if using fork() above
