import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import math
from matplotlib.colors import LinearSegmentedColormap

def value(x, y, x_sensor, y_sensor):
    # Placeholder for the 'value' function. Define it according to your requirements.
    # Example implementation could be a function of the distance between (x, y) and (x_sensor, y_sensor)
    return np.sqrt((x - x_sensor) ** 2 + (y - y_sensor) ** 2)

def earthquake_exercise_setup():
    # explosion detector (using spiral coordinate system)

    # define the coordinate system:
    S = 2000  # number of points on the spiral
    rate = 25  # angular rate of spiral
    sd = 0.2  # standard deviation of the sensor Gaussian noise

    x = np.zeros(S)
    y = np.zeros(S)
    for s in range(S):
        theta = rate * 2 * np.pi * s / S
        r = s / S
        x[s] = r * np.cos(theta)
        y[s] = r * np.sin(theta)

    # define the locations of the detection stations on the surface
    N = 30  # number of stations
    x_sensor = np.zeros(N)
    y_sensor = np.zeros(N)
    v = np.zeros((S, N))
    for sensor in range(N):
        theta_sensor = 2 * np.pi * sensor / N
        x_sensor[sensor] = np.cos(theta_sensor)
        y_sensor[sensor] = np.sin(theta_sensor)
        for s in range(S):
            v[s, sensor] = value(x[s], y[s], x_sensor[sensor], y_sensor[sensor])  # explosion value for some value function

    return x, y, x_sensor, y_sensor, v

# Call the function and get the result
# x, y, x_sensor, y_sensor, v = earthquake_exercise_setup()

# define read data
def read_data(file_path: str):
    with open(file_path, 'r') as f:
        res = [float(line.strip()) for line in f.readlines()]
    
    return res

# def plot_outermost_loop(S, rate):
#     # Define the start and end angle for the last loop of the spiral
#     theta_start = 2 * np.pi * (rate - 1)
#     theta_end = 2 * np.pi * rate

#     # Generate the points for the outermost loop
#     theta = np.linspace(theta_start, theta_end, S)
#     r = theta / (2 * np.pi * rate)

#     # Convert polar coordinates to Cartesian
#     x = r * np.cos(theta)
#     y = r * np.sin(theta)
    
#     return x, y

def visualize(estimate_x, estimate_y, x_sensor, y_sensor, vs, s1, s2, all_x, all_y):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=100)
    # cmap = LinearSegmentedColormap.from_list('mycmap', ['white', 'black'])
    # cmap2 = LinearSegmentedColormap.from_list('mycmap', ['white', 'blue'])

    # Plot the spiral using a line plot
    x, y = np.cos(np.arange(0,2*np.pi,0.01)), np.sin(np.arange(0,2*np.pi,0.01))
    ax.plot(x, y, 'k-', label='Spiral Path')


    # Scatter plot for the locations of the detection stations
    ax.scatter(x_sensor, y_sensor, color='red', marker='o', label='Sensor Locations')
    

     # Plot the observed measurements as lines extending from the sensor locations
    for sensor_x, sensor_y, measurement in zip(x_sensor, y_sensor, np.array(vs) / 10):
        # Calculate the direction of the measurement vector
        direction = np.array([sensor_x, sensor_y])
        direction = direction / np.linalg.norm(direction)  # Normalize the direction vector

        # Calculate the end point of the measurement vector
        end_point = np.array([sensor_x, sensor_y]) + direction * measurement

        # Plot the measurement vector
        ax.plot([sensor_x, end_point[0]], [sensor_y, end_point[1]], 'm-')
        
    # ax.scatter(all_x, all_y, s = 50 * s2)
    ax.scatter(all_x, all_y, s = 50 * s1, c = 'green', label = 'p(h1|v)')
    ax.scatter(all_x, all_y, s = 50 * s2, c = 'grey', label = 'p(h2|v)')
    
    # plot the esitamted explosion
    ax.scatter(estimate_x, estimate_y, color='cyan', marker='x', label='Estimated Locations')
    
    # Setting labels and title
    ax.set_aspect('equal')
    ax.set_title('Spiral Coordinate System and Sensor Locations')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')

    # Adding a legend
    ax.legend(loc = 'best', fontsize = 6)

    # Display the plot
    plt.savefig('../../figure/question_11.png')
    plt.show()


        
def compute_joint_posterior_vectorized(v, distances, epsilon, sd):
    v_funcs = 1/(distances[:, 0]**2 + 0.1) + 1/(distances[:, 1]**2 + 0.1)
    v_means = v_funcs + sd * epsilon
    return np.sum((v - v_means)**2, axis=1)

def map_optimized(v, distance, sd):
    np.random.seed(7)
    epsilon = np.random.normal(0, 1, size=30)
    
    combos = np.array(list(combinations(np.arange(2000), 2)))
    distances = distance[combos]  # Assuming distance can be indexed like this

    posteriors = compute_joint_posterior_vectorized(v, distances, epsilon, sd)
    min_index = np.argmin(posteriors) # negavtive sign in normal pdf thus max -xxxx -> min xxxx  

    return combos[min_index], posteriors

def compute_joint_normalized_posterior(posterior, sd):
    unormalized = (1/(np.sqrt(2 * math.pi) * sd))**30*np.exp(-(posterior/(2*sd**2)))
    return unormalized / np.sum(unormalized)

def compute_single_normalized_posterior(joint_p):
    combo = np.array(list(combinations(range(2000), 2)))
    s1_posteriors = np.zeros(2000)
    s2_posteriors = np.zeros(2000)
    for i, (s1, s2) in enumerate(combo):
        s1_posteriors[s1] += joint_p[i]
        s2_posteriors[s2] += joint_p[i] 
        
    return s1_posteriors / np.sum(s1_posteriors), s2_posteriors / np.sum(s2_posteriors)

# if __name__ == '__main__':
    
# x, y, x_sensor, y_sensor, distance = earthquake_exercise_setup()
# vs = read_data('../../data/EarthquakeExerciseData.txt')
# (s1, s2), posterior = map_optimized(vs, distance, 0.2)
# estimate_x, estimate_y = x[[s1, s2]], y[[s1, s2]]
# posterior_distribution = compute_joint_normalized_posterior(posterior, 0.2)
# h1, h2 = compute_single_normalized_posterior(posterior_distribution)
# visualize(estimate_x, estimate_y, x_sensor, y_sensor, vs, h1, h2, x,y)
