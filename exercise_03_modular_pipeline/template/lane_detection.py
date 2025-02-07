import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.spatial import distance
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time


class LaneDetection:
    '''
    Lane detection module using edge detection and b-spline fitting

    args: 
        cut_size (cut_size=68) cut the image at the front of the car
        spline_smoothness (default=10)
        gradient_threshold (default=14)
        distance_maxima_gradient (default=3)

    '''

    def __init__(self, cut_size=66, spline_smoothness=10, gradient_threshold=14, distance_maxima_gradient=3):
        self.car_position = np.array([48,0])
        self.spline_smoothness = spline_smoothness
        self.cut_size = cut_size
        self.gradient_threshold = gradient_threshold
        self.distance_maxima_gradient = distance_maxima_gradient
        self.lane_boundary1_old = 0
        self.lane_boundary2_old = 0


    def cut_gray(self, state_image_full):
        '''
        ##### TODO--> DONE #####
        This function should cut the image at the front end of the car (e.g. pixel row 68)
        and translate to grey scale

        input:
            state_image_full 96x96x3

        output:
            gray_state_image 68x96x1

        '''
        state = np.dot(state_image_full[..., :3], [0.299, 0.587, 0.144])
        gray_state_image = state[:self.cut_size, :]
        plt.imshow(gray_state_image[::-1])
        return gray_state_image[::-1]


    def edge_detection(self, gray_image):
        '''
        ##### TODO--> Done #####
        In order to find edges in the gray state image, 
        this function should derive the absolute gradients of the gray state image.
        Derive the absolute gradients using numpy for each pixel. 
        To ignore small gradients, set all gradients below a threshold (self.gradient_threshold) to zero. 

        input:
            gray_state_image 68x96x1

        output:
            gradient_sum 68x96x1

        '''
        gradients = np.gradient(gray_image)
        h_gradient = np.absolute(gradients[0])
        h_gradient[h_gradient < self.gradient_threshold] = 0

        v_gradient = np.absolute(gradients[1])
        v_gradient[v_gradient < self.gradient_threshold] = 0

        gradient_sum = h_gradient + v_gradient
        plt.imshow(gradient_sum)
        return gradient_sum


    def find_maxima_gradient_rowwise(self, gradient_sum):
        '''
        ##### TODO #####
        This function should output arguments of local maxima for each row of the gradient image.
        You can use scipy.signal.find_peaks to detect maxima. 
        Hint: Use distance argument for a better robustness.

        input:
            gradient_sum 68x96x1

        output:
            maxima (np.array) 2x Number_maxima

        '''
        argmaxima = []

        for row in gradient_sum:
            peaks, _ = find_peaks(row, distance=self.distance_maxima_gradient)
            argmaxima.append(peaks)
        argmaxima = np.asarray(argmaxima)
        #print("PRINT argmaxima: ", argmaxima)
        return argmaxima


    def find_first_lane_point(self, gradient_sum):
        '''
        Find the first lane_boundaries points above the car.
        Special cases like just detecting one lane_boundary or more than two are considered. 
        Even though there is space for improvement ;) 

        input:
            gradient_sum 68x96x1

        output: 
            lane_boundary1_startpoint
            lane_boundary2_startpoint
            lanes_found  true if lane_boundaries were found
        '''
        
        # Variable if lanes were found or not
        lanes_found = False
        row = 0

        # loop through the rows
        while not lanes_found:
            
            # Find peaks with min distance of at least 3 pixel 
            argmaxima = find_peaks(gradient_sum[row], distance=self.distance_maxima_gradient)[0]

            # if one lane_boundary is found
            if argmaxima.shape[0] == 1:
                lane_boundary1_startpoint = np.array([[argmaxima[0],  row]])

                if argmaxima[0] < 48:
                    lane_boundary2_startpoint = np.array([[0,  row]])
                else: 
                    lane_boundary2_startpoint = np.array([[96,  row]])

                lanes_found = True
            
            # if 2 lane_boundaries are found
            elif argmaxima.shape[0] == 2:
                lane_boundary1_startpoint = np.array([[argmaxima[0],  row]])
                lane_boundary2_startpoint = np.array([[argmaxima[1],  row]])
                lanes_found = True

            # if more than 2 lane_boundaries are found
            elif argmaxima.shape[0] > 2:
                # if more than two maxima then take the two lanes next to the car, regarding least square
                A = np.argsort((argmaxima - self.car_position[0])**2)
                lane_boundary1_startpoint = np.array([[argmaxima[A[0]],  0]])
                lane_boundary2_startpoint = np.array([[argmaxima[A[1]],  0]])
                lanes_found = True

            row += 1
            
            # if no lane_boundaries are found
            if row == self.cut_size:
                lane_boundary1_startpoint = np.array([[0,  0]])
                lane_boundary2_startpoint = np.array([[0,  0]])
                break
        #plt.plot(lane_boundary1_startpoint)
        #plt.show()
        return lane_boundary1_startpoint, lane_boundary2_startpoint, lanes_found


    def lane_detection(self, state_image_full):
        '''
        This function should perform the road detection

        args:
            state_image_full [96, 96, 3]

        out:
            lane_boundary1 spline
            lane_boundary2 spline
        '''

        # to gray
        gray_state = self.cut_gray(state_image_full)

        # edge detection via gradient sum and thresholding
        gradient_sum = self.edge_detection(gray_state)
        maxima = self.find_maxima_gradient_rowwise(gradient_sum)
        maxima = maxima
        print("maxima.ndim: ", maxima.ndim)
        print("maxima.size: ", maxima.size)
        print("maxima.shape: ", maxima.shape)
        print("maxima.dtype: ", maxima[0].dtype)
        print("maxima values: ", maxima)
        #array = np.fromiter(maxima.items(), count=len(maxima))

        # first lane_boundary points
        lane_boundary1_points, lane_boundary2_points, lane_found = self.find_first_lane_point(gradient_sum)
        print("lane boundary points1: ", lane_boundary1_points)
        print("lane boundary points2: ", lane_boundary2_points)
        # if no lane was found,use lane_boundaries of the preceding step
        if lane_found:
            #  in every iteration: 
            # 1- find maximum/edge with the lowest distance to the last lane boundary point 
            # 2- append maxium to lane_boundary1_points or lane_boundary2_points
            # 3- delete maximum from maxima
            # 4- stop loop if there is no maximum left 
            #    or if the distance to the next one is too big (>=100)
            last_point_lane_1 = 0
            last_point_lane_2 = 0
            print("MÖP: ", lane_boundary1_points[last_point_lane_1])
            print("MÖP: ", lane_boundary1_points.dtype)
            print("MÖP: ", lane_boundary1_points.shape)
            print("MÖP: ", lane_boundary1_points.ndim)
            print("MÖP: ", lane_boundary1_points.reshape(1, -1).ndim)
            print("MÖP: ", lane_boundary1_points.reshape(-1, 1).shape)

            while maxima.size != 0:
                # lane_boundary 1 np.array([1,2,3,4]).reshape(-1,1).shape
                print("TEST_np.linalg.norm: ", distance.cdist(maxima, lane_boundary1_points[last_point_lane_1].reshape(-1, 1)))
                nearest_1 = np.argmin(np.linalg.norm(maxima - lane_boundary1_points[last_point_lane_1]))
                lowest_dist_1 = np.amin(np.linalg.norm(maxima - lane_boundary1_points[last_point_lane_1]))
                # nearest_1 = np.argmin(distance.cdist(maxima, lane_boundary1_points))
                # lowest_dist_1 = np.amin(distance.cdist(maxima, lane_boundary1_points[last_point_lane_1]))
                if lowest_dist_1 >= 100:
                    pass
                else:
                    np.append(lane_boundary1_points, maxima[nearest_1])
                    np.delete(maxima, nearest_1)
                    last_point_lane_1 += 1
                
                # lane_boundary 2
                nearest_2 = np.argmin(distance.cdist(maxima, lane_boundary2_points[last_point_lane_2]))
                lowest_dist_2 = np.amin(distance.cdist(maxima, lane_boundary2_points[last_point_lane_2]))
                if lowest_dist_2 >= 100:
                    pass
                else:
                    np.append(lane_boundary2_points, maxima[nearest_2])
                    np.delete(maxima, nearest_2)
                    last_point_lane_2 += 1
                ################

            ##### TODO #####
            # spline fitting using scipy.interpolate.splprep 
            # and the arguments self.spline_smoothness
            # 
            # if there are more lane_boundary points points than spline parameters 
            # else use perceding spline
            if lane_boundary1_points.shape[0] > 4 and lane_boundary2_points.shape[0] > 4:

# Pay attention: the first lane_boundary point might occur twice
# lane_boundary 1
                (_, lane_boundary1) = splprep(lane_boundary1_points, s=self.spline_smoothness)
# lane_boundary 2
                (_, lane_boundary2) = splprep(lane_boundary2_points, s=self.spline_smoothness)

            else:
                lane_boundary1 = self.lane_boundary1_old
                lane_boundary2 = self.lane_boundary2_old
            ################

        else:
            lane_boundary1 = self.lane_boundary1_old
            lane_boundary2 = self.lane_boundary2_old

        self.lane_boundary1_old = lane_boundary1
        self.lane_boundary2_old = lane_boundary2

        # output the spline
        return lane_boundary1, lane_boundary2


    def plot_state_lane(self, state_image_full, steps, fig, waypoints=[]):
        '''
        Plot lanes and way points
        '''
        # evaluate spline for 6 different spline parameters.
        t = np.linspace(0, 1, 6)
        lane_boundary1_points_points = np.array(splev(t, self.lane_boundary1_old))
        lane_boundary2_points_points = np.array(splev(t, self.lane_boundary2_old))
        
        plt.gcf().clear()
        plt.imshow(state_image_full[::-1])
        plt.plot(lane_boundary1_points_points[0], lane_boundary1_points_points[1]+96-self.cut_size, linewidth=5, color='orange')
        plt.plot(lane_boundary2_points_points[0], lane_boundary2_points_points[1]+96-self.cut_size, linewidth=5, color='orange')
        if len(waypoints):
            plt.scatter(waypoints[0], waypoints[1]+96-self.cut_size, color='white')

        plt.axis('off')
        plt.xlim((-0.5,95.5))
        plt.ylim((-0.5,95.5))
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        fig.canvas.flush_events()
