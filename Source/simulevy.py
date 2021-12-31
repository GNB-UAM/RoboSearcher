#!/usr/bin/python
# Evaluates the characteristics of a Levy walk (https://www.nature.com/articles/s41598-021-03826-3)
# Compares its exploratory performance for distinct strategies to handle walls and obstacles
# Carlos Garcia-Saura, Grupo de Neurocomputacion, Universidad Autonoma de Madrid 2017-2022

# Begin modules
import array as arr
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
import time
import hashlib
import os
import subprocess
#import pywt # haar wavelet transform
# End modules

from math import sin, cos, hypot, atan2, sqrt, pow, floor

vectorized_append = vectorize(list.append)

def exponent_fmt(x,pos=None):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

def human_format_precision(num,pos=None):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

def human_format(num,pos=None):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    if magnitude == 0:
        return '%d%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
    return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

class Robot():
    def __init__(self):
        self.setPos()
        self.setParameters()
        self.setWorldParameters()
        self.plot = False
        self.steps_per_frame = 500
        self.wall_detection_distance = 1.
        self.total_steps = 0
        self.plot_title = "2D robot exploration"
        self.ani = []
        self.fig = []
        self.end = False
        self.remaining_distance = 0.
        self.use_memory = False
        
        self.variability_evolution = []
    
    
    def setPos(self, x=0, y=0, a=deg2rad(-50)):
        self.x = float(x)
        self.y = float(y)
        self.a = float(a)
    
    
    def setParameters(self, sim_step=1, radius=5, sensor_radius=20, drift_degPerStep=0):
        self.sim_step = float(sim_step)
        self.radius = float(radius)
        self.sensor_radius = float(sensor_radius) # radius, distance from center of robot
        self.drift = float(drift_degPerStep) * pi / 180.
        
        self.sensor_mask = zeros((int(self.sensor_radius*2+1),int(self.sensor_radius*2+1)), dtype=bool) # the +1 margin allows to set zero the previous visits overlap at each step
        for px in range(self.sensor_mask.shape[0]):
            for py in range(self.sensor_mask.shape[1]):
                dist = hypot(px-self.sensor_radius,py-self.sensor_radius)
                if dist < self.sensor_radius:
                    self.sensor_mask[py,px] = True
        self.sensor_mask_inverted = logical_not(self.sensor_mask)
    
    
    def setWorldParameters(self, world_radius=500, obstacles=[]):
        self.world_radius = float(world_radius)
        self.obstacles = obstacles
        r = self.world_radius
        self.wold_extent = [-r,r, -r,r]
        matsize = int(2*r+1)
        self.matsize = matsize
        self.pos_histogram = zeros((matsize,matsize), dtype=uint32)
        self.pos_histogram_max_val = 0
        self.last_visited = zeros((matsize,matsize), dtype=uint32)
        self.one_last_visited = zeros((matsize,matsize), dtype=uint32)
        self.one_last_visited_timestamp = zeros((matsize,matsize), dtype=uint32)
        self.last_visited_time_counter = zeros((matsize,matsize), dtype=uint32)
        self.last_visit_mask = zeros((matsize,matsize), dtype=bool)
        self.last_visit_mask_range = []
        self.cell_visits = empty((matsize,matsize),dtype=object)
        for i in range(matsize):
            for j in range(matsize):
                self.cell_visits[i,j] = arr.array('L', [])
        obstacles_hash = hashlib.md5(str(obstacles).encode('utf-8')).hexdigest()
        cache_filename = "cache/"+obstacles_hash+".npy"
        try:
            self.pos_histogram_mask = load(cache_filename)
            print("Loading obstacle mask from cached file...")
        except:
            print("Creating obstacle mask...")
            if True: # Set to false in order to disable mask generation
                self.pos_histogram_mask = zeros(self.pos_histogram.shape, dtype=bool)
                for px in range(self.pos_histogram.shape[0]):
                    for py in range(self.pos_histogram.shape[1]):
                        (wall_dist, wall_angle) = self.checkObstacle(px-r,py-r)
                        if wall_dist > 0:
                            self.pos_histogram_mask[py,px] = True
                os.makedirs("cache", exist_ok=True)
                save(cache_filename, self.pos_histogram_mask)
            else:
                self.pos_histogram_mask = ones(self.pos_histogram.shape, dtype=bool)
        self.pos_histogram_mask_inverted = logical_not(self.pos_histogram_mask)
        
        b1 = self.pos_histogram
        b2 = self.sensor_mask
        self.slice_shapes = (b1.shape[0],b1.shape[1], b2.shape[0],b2.shape[1])
    
    def setStrategy(self, strategy):
        self.strategy = strategy
    
    def checkObstacle(self, x,y):
        distance_from_center = hypot(x,y) # check world limits
        wall_dist = self.world_radius-distance_from_center
        wall_angle = atan2(y,x)+pi
        
        for (obs_x,obs_y,obs_radius) in self.obstacles:
            distance_to_obstacle = hypot(x-obs_x,y-obs_y)
            other_wall_dist = distance_to_obstacle - obs_radius
            if other_wall_dist < wall_dist:
                wall_dist = other_wall_dist
                wall_angle = atan2(y-obs_y,x-obs_x)
        
        return (wall_dist, wall_angle)
    
    def checkVisited(self, angle, lenght):
        self.use_memory = True
        newx = self.x + lenght*cos(angle)
        newy = self.y + lenght*sin(angle)
        
        
        if self.plot and False: # Set to True in order to plot the target position, useful to debug the memory implementation
            self.plt_robot_dest.center = (newx,newy)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            time.sleep(0.1)
        
        (rx,ry) = int(floor(newx+self.world_radius)), int(floor(newy+self.world_radius))
        
        max_pos = int(2*self.world_radius+1)
        if rx < 0 or ry < 0: return False
        if rx >= max_pos or ry >= max_pos: return False
        
        return self.last_visited[ry,rx] > max(self.total_steps-100,0)
    
    def plotMap(self,ax=None):
        plt.title(self.plot_title)
        search_area_color = 'white'
        edge_color = 'green'
        obstacle_color = 'gray'
        
        if ax==None: ax = plt.gca()
        ax.set_facecolor(obstacle_color)
        
        cmap = plt.get_cmap("coolwarm")
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # force the first color entry to be grey
        cmaplist[0] = (0,0,0,1.0)
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
        
        self.plt_pos_histogram = ax.imshow(self.pos_histogram, extent=self.wold_extent, origin='low', cmap='jet')#cmap)
        cbar = plt.colorbar(self.plt_pos_histogram, format=ticker.FuncFormatter(human_format))#format='%d')
        cbar.set_label("Cell visit count")
        #cbar.set_label("Time from last visit [steps]")
        
        
        #ax.imshow(self.pos_histogram_mask, extent=self.wold_extent, cmap='gray', origin='low', alpha=.9)
        
        circle = plt.Circle((0,0), self.world_radius, edgecolor=edge_color, facecolor=search_area_color, fill=False)
        ax.add_patch(circle)
        
        for (obs_x,obs_y,obs_radius) in self.obstacles: # Obstacles
            circle = plt.Circle((obs_x,obs_y), obs_radius, edgecolor=edge_color, facecolor=obstacle_color, fill=False)
            ax.add_patch(circle)
        plt.xlim((-self.world_radius, self.world_radius))
        plt.ylim((-self.world_radius, self.world_radius))
        ax.set_aspect('equal')
        
        # Draw sensor range
        self.plt_robot_sensor_circle = plt.Circle((self.x, self.y), self.sensor_radius, facecolor='green', alpha=0.1)
        ax.add_patch(self.plt_robot_sensor_circle)
        
        # Draw robot
        self.plt_robot_circle = plt.Circle((self.x, self.y), self.radius, edgecolor='black', facecolor='cyan')
        ax.add_patch(self.plt_robot_circle)
        self.plt_robot_arrow = plt.Polygon([(0,0),(0,0),(0,0)], facecolor='green', edgecolor='black')
        ax.add_patch(self.plt_robot_arrow)
        
        #self.plt_robot_dest = plt.Circle((self.x, self.y), self.radius, edgecolor='black', facecolor='red')
        #ax.add_patch(self.plt_robot_dest)
        
        self.redrawRobot()

        plt.xticks([])
        plt.yticks([])
        
        self.plot = True
    
    def redrawRobot(self):
        self.plt_robot_circle.center = (self.x, self.y)
        self.plt_robot_sensor_circle.center = (self.x, self.y)
        self.plt_robot_arrow.set_xy([(self.x+self.radius*cos(self.a-pi/3.)*0.5,self.y+self.radius*sin(self.a-pi/3.)*0.5),
                               (self.x+self.radius*cos(self.a),self.y+self.radius*sin(self.a)),
                               (self.x+self.radius*cos(self.a+pi/3.)*0.5,self.y+self.radius*sin(self.a+pi/3.)*0.5)])
    
    def steps(self, N, ignoreBounds=False):
        N = int(N)
        if ignoreBounds:
            for i in range(N):
                self.a = self.strategy(False, 0, 0)
                
                self.x += self.sim_step*cos(self.a)
                self.y += self.sim_step*sin(self.a)
                self.a += self.drift
                
                self.total_steps += 1
            if self.plot:
                self.redrawRobot()
        
        else: # Normal simulation with obstacles
            for i in range(N):
                wall_dist, wall_angle = self.checkObstacle(self.x, self.y)
                if wall_dist >= self.radius:
                    wall_detected = wall_dist - self.radius <= self.wall_detection_distance
                    self.a = self.strategy(wall_detected, wall_dist, wall_angle)
                else: # Robot collides with obstacle
                    self.a = wall_angle # bounce away from wall
                self.x += self.sim_step*cos(self.a)
                self.y += self.sim_step*sin(self.a)
                self.a += self.drift
                
                # Rounded robot position, projected into the grid
                (pos_h,pos_v) = int(floor(self.x+self.world_radius-self.sensor_radius)), int(floor(self.y+self.world_radius-self.sensor_radius))
                
                b1 = self.pos_histogram
                b2 = self.sensor_mask
                self.slice_shapes = (b1.shape[0],b1.shape[1], b2.shape[0],b2.shape[1])
                (b1shape0,b1shape1,b2shape0,b2shape1) = self.slice_shapes
                v_range1 = slice(max(0, pos_v), max(min(pos_v + b2shape0, b1shape0), 0))
                h_range1 = slice(max(0, pos_h), max(min(pos_h + b2shape1, b1shape1), 0))

                v_range2 = slice(max(0, -pos_v), min(-pos_v + b1shape0, b2shape0))
                h_range2 = slice(max(0, -pos_h), min(-pos_h + b1shape1, b2shape1))
                
                last_visited_slice = self.last_visited[v_range1, h_range1]
                sensor_mask_slice = self.sensor_mask[v_range2, h_range2]
                
                old_cells = self.total_steps-5 > last_visited_slice
                new_revisits = logical_and(old_cells,sensor_mask_slice)
                
                last_visited_slice[sensor_mask_slice] = self.total_steps
                
                # UNCOMMENT WHEN STORING CELL VISIT TIMES
                #cell_visits_view = self.cell_visits[v_range1, h_range1]
                #for cell in cell_visits_view[new_revisits]:
                #    cell.append(self.total_steps)
                
                # UNCOMMENT FOR POSITION HISTOGRAM
                histogram_slice = self.pos_histogram[v_range1, h_range1]
                histogram_slice[new_revisits] += 1
                
                
                # UNCOMMENT FOR INCREMENTAL FULL MAP COUNTER
                #self.last_visited_time_counter += 1
                #last_visited_time_counter_slice = self.last_visited_time_counter[v_range1, h_range1]
                #last_visited_time_counter_slice[sensor_mask_slice] = 0
                
                # UNCOMMENT TO REGISTER EXITED CELLS
                #exited_cells = logical_and(self.last_visit_mask[v_range1, h_range1], self.sensor_mask_inverted[v_range2, h_range2])
                '''
                exited_cells = nonzero(logical_and(self.last_visit_mask[v_range1, h_range1], self.sensor_mask_inverted[v_range2, h_range2]))
                histogram_slice = b1[v_range1, h_range1]
                last_visited_slice = self.last_visited[v_range1, h_range1]
                for index in range(len(exited_cells[0])):
                    ii = exited_cells[0][index]
                    jj = exited_cells[1][index]
                    if self.total_steps > last_visited_slice[ii,jj]+3:
                        histogram_slice[ii,jj] += 1
                        last_visited_slice[ii,jj] = self.total_steps
                '''
                
                # UNCOMMENT FOR EITHER SIMPLE POSITION HISTOGRAM or TIME SIMULATION
                #b1[v_range1, h_range1] += b2[v_range2, h_range2]
                
                # UNCOMMENT WHEN STORING ONE LAST VISIT
                #one_last_visited_timestamp_view = self.one_last_visited_timestamp[v_range1, h_range1]
                #one_last_visited_view = self.one_last_visited[v_range1, h_range1]
                #exited_cells = nonzero(logical_and(self.last_visit_mask[v_range1, h_range1], self.sensor_mask_inverted[v_range2, h_range2]))
                #for index in range(len(exited_cells[0])):
                #    ii = exited_cells[0][index]
                #    jj = exited_cells[1][index]
                #    one_last_visited_view[ii,jj] = self.total_steps - one_last_visited_timestamp_view[ii,jj]
                #    one_last_visited_timestamp_view[ii,jj] = self.total_steps
                
                # UNCOMMENT WHEN STORING CELLS
                #cell_visits_view = self.cell_visits[v_range1, h_range1]
                #exited_cells = cell_visits_view[logical_and(self.last_visit_mask[v_range1, h_range1], self.sensor_mask_inverted[v_range2, h_range2])]
                #for cell in exited_cells:
                #    cell.append(self.total_steps)
                
                # ENABLE LAST_VISITED WHEN USING MEMORY
                #putmask(self.last_visited[v_range1, h_range1], b2[v_range2, h_range2], self.total_steps)
                
                # VISUALIZATION OF LAST VISITED TIME
                #self.pos_histogram = self.last_visited_time_counter
                
                # VISUALIZATION OF RECORDED CELLS
                #self.pos_histogram = logical_and(self.last_visit_mask[v_range1, h_range1], self.sensor_mask_inverted[v_range2, h_range2])
                
                # Create last mask (array of 0s with mask -1s-)
                #self.last_visit_mask_range = (v_range1, h_range1) # This is NOT needed ONLY BECAUSE the sensor mask has 0s in all its perimeter
                #self.last_visit_mask[self.last_visit_mask_range] = 0 # So the value is already 0 and we can skip this step. Remember to check if updating the masks.
                
                # also re-set to 0 the perimeter (last explored cells)
                #self.last_visit_mask[v_range1, h_range1] = b2[v_range2, h_range2]
                
                self.total_steps += 1
            
            # UNCOMMENT FOR EITHER TIME SIMULATION or POSITION HISTOGRAM
            self.pos_histogram = multiply(self.pos_histogram, self.pos_histogram_mask) # Set 0 to the obstacle areas
            
            # UNCOMMENT FOR POSITION HISTOGRAM DENSITY TERMINATION
            '''
            self.pos_histogram_max_val = self.pos_histogram.max()
            if self.pos_histogram_max_val > 7500000:
                self.end = True
            '''
            
            if self.plot:
                text_paused = ""
                #text_paused = "[PAUSED]" if self.total_steps in [1e3,1e4,1e5,1e6,1e7] else ""
                plt.xlabel("Step: "+human_format(self.total_steps)+" "+text_paused, horizontalalignment='left')
                
                # Sef plot limits, defaults to autorange
                #self.plt_pos_histogram.set_clim([0,max(9,8500*self.total_steps/100000000.)])
                #self.plt_pos_histogram.set_clim([0,3e5])
                
                # UNCOMMENT FOR variability evolution
                '''
                coeffs = pywt.dwt2(self.pos_histogram, 'haar')
                cA, (cH, cV, cD) = coeffs
                thr = 1
                
                stdCalc = int(std(self.pos_histogram[self.pos_histogram_mask]))
                
                variability = (self.total_steps,
                               (cA > thr).sum(),
                               (cH > thr).sum(),
                               (cV > thr).sum(),
                               (cD > thr).sum(),
                               stdCalc)
                               
                               
                
                print(variability)
                
                self.variability_evolution.append(variability)
                '''
                
                # Uncomment to visualize N visits
                '''
                for i in range(self.matsize):
                    for j in range(self.matsize):
                        self.pos_histogram[i,j] = len(self.cell_visits[i,j])
                '''
                
                '''
                nonZero = ma.masked_equal(self.pos_histogram, 0, copy=False)
                minVal = 0 #nonZero.min() # optional if 0 isn't the minimum
                maxVal = nonZero.max()
                self.plt_pos_histogram.set_clim([minVal-1,maxVal])
                self.plt_pos_histogram.set_data(self.pos_histogram)
                '''
                
                # Auto.range the color scale
                self.plt_pos_histogram.set_clim([0,max(1,self.pos_histogram.max())])
                
                # Important: Specify the desired magnitude to represent
                pos_hist_masked = ma.masked_where(self.pos_histogram_mask_inverted, self.pos_histogram) # self.last_visited_time_counter)
                self.plt_pos_histogram.set_data(pos_hist_masked)
                #self.plt_pos_histogram.set_data(image_data)
                self.redrawRobot()
                
                fig = plt.gcf()
                fig.canvas.draw() # update frame for the video
                
                '''
                try:
                    # extract the image as an ARGB string
                    string = fig.canvas.tostring_argb()
                    # write to pipe
                    self.video_p.stdin.write(string)
                except:
                    fps = 30
                    canvas_width, canvas_height = fig.canvas.get_width_height()
                    # Open an ffmpeg process
                    outf = self.plot_title+'.mp4'
                    cmdstring = ('ffmpeg', # https://stackoverflow.com/a/31315362
                                '-y', '-r', str(fps),
                                '-s', '%dx%d' % (canvas_width, canvas_height), # size of image string
                                '-pix_fmt', 'argb', # format
                                '-f', 'rawvideo',  '-i', '-', # tell ffmpeg to expect raw video from the pipe
                                '-q:v', '0',
                                '-vcodec', 'mpeg4', outf) # output encoding
                    self.video_p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)
                
                def appendSeconds(sec):
                    string = fig.canvas.tostring_argb()
                    for i in range(sec*30):
                        self.video_p.stdin.write(string)
                
                if self.total_steps == 1e6:
                    appendSeconds(3)
                    self.steps_per_frame = 1e5
                elif self.total_steps == 1e5:
                    appendSeconds(3)
                    self.steps_per_frame = 1e4
                elif self.total_steps == 1e4:
                    appendSeconds(3)
                    self.steps_per_frame = 1e3
                elif self.total_steps == 1e3:
                    appendSeconds(3)
                    self.steps_per_frame = 100
                
                if self.total_steps == 3e5:
                    save(self.plot_title+"_last_visited_time_counter_300k", self.last_visited_time_counter)
                    exit(0)
                '''
                
                if self.end or self.total_steps >= 1e8:
                    print("Simulation "+self.plot_title+" completed in "+str(self.total_steps))
                    
                    
                    '''
                    # Extend the last frame
                    appendSeconds(5)
                    
                    # Finish up video
                    self.video_p.communicate()
                    '''
                    
                    self.pause = 1
                    self.ani.event_source.stop()
                    self.steps_per_frame = 1
                    self.end = True
                    exit(0)
            
            print(self.total_steps)
        '''
        explored_area = count_nonzero(self.pos_histogram)
        if explored_area >= 295575:
            print(self.plot_title+" FINISHED in " + str(self.total_steps) + " steps")
            time.sleep(1)
            exit(0)
        '''
    
    def animation_frame(self, frame=0):
        self.steps(self.steps_per_frame)
    
    def animate(self):
        self.fig = plt.figure()
        self.ani = animation.FuncAnimation(self.fig, init_func=self.plotMap, func=self.animation_frame, interval=20)#, frames=10000000, repeat=False)#, frames=int(800000/500), repeat=False) # uncomment the last part ("frames="..) to set video duration

        self.pause = False
        def onKey(event):
            if event.key == 'left':
                self.a = deg2rad(180)
            if event.key == 'right':
                self.a = deg2rad(0)
            if event.key == 'up':
                self.a = deg2rad(90)
            if event.key == 'down':
                self.a = deg2rad(-90)
            if event.key == ' ':
                if self.pause: self.ani.event_source.start()
                else: self.ani.event_source.stop()
                self.pause ^= -1
                self.steps_per_frame = 1
            if event.key == '+':
                self.steps_per_frame *= 10
                if self.steps_per_frame > 100000: self.steps_per_frame = 100000
                print(self.steps_per_frame)
            if event.key == '-':
                self.steps_per_frame /= 10
                if self.steps_per_frame < 1: self.steps_per_frame = 1
            self.steps_per_frame = int(self.steps_per_frame)
            if event.key == 'escape':
                exit(0)

        self.fig.canvas.mpl_connect('key_press_event', onKey)
        
        #save(self.plot_title+"_variability_evolution", self.variability_evolution)
        #exit(0)
        plt.show()

# To stack the videos:
# ffmpeg -i "Lévy, mirror bounce.mp4" -i "Ballistic, random bounce.mp4" -i "Brownian, memory.mp4" -i "Brownian.mp4" -filter_complex "[0:v][1:v]hstack[top]; [2:v][3:v]hstack[bottom]; [top][bottom]vstack,format=yuv420p[v]" -map "[v]" merged.mp4
