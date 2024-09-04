import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, LiteralString, Self

from transforms import get_corners

def read(file: LiteralString):
    case_params = {}
    with open(file, 'r') as f:
        reader = csv.reader(f)
        tmp = list(reader)
        v = [float(i) for i in tmp[0]]
        case_params["x0"], case_params["y0"], case_params["yaw0"] = v[0:3]
        case_params["xf"], case_params["yf"], case_params["yawf"] = v[3:6]
        case_params["xmin"] = min(case_params["x0"], case_params["xf"]) - 8
        case_params["xmax"] = max(case_params["x0"], case_params["xf"]) + 8
        case_params["ymin"] = min(case_params["y0"], case_params["yf"]) - 8
        case_params["ymax"] = max(case_params["y0"], case_params["yf"]) + 8

        case_params["obs_num"] = int(v[6])
        num_vertexes = np.array(v[7:7 + case_params["obs_num"]], dtype=int)
        vertex_start = 7 + case_params["obs_num"] + (np.cumsum(num_vertexes, dtype=int) - num_vertexes) * 2
        case_params["obs"] = []
        for vs, nv in zip(vertex_start, num_vertexes):
            case_params["obs"].append(np.array(v[vs:vs + nv * 2]).reshape((nv, 2), order='A'))
    return case_params

def plot_case(case_params, car_params, filename=None, show=False, save=True, bare=False):
    if filename is None:
        filename = 1
    plt.xlim(case_params["xmin"], case_params["xmax"])
    plt.ylim(case_params["ymin"], case_params["ymax"])
    plt.gca().set_aspect('equal', adjustable = 'box')
    plt.gca().set_axisbelow(True)

    for j in range(0, case_params["obs_num"]):
        plt.fill(case_params["obs"][j][:, 0], case_params["obs"][j][:, 1], facecolor = 'k', alpha = 0.5)

    if bare is False:
        plt.arrow(case_params["x0"], case_params["y0"], np.cos(case_params["yaw0"]), np.sin(case_params["yaw0"]), width=0.2, color = "gold")
        plt.arrow(case_params["xf"], case_params["yf"], np.cos(case_params["yawf"]), np.sin(case_params["yawf"]), width=0.2, color = "gold")
        temp = get_corners(car_params, case_params["x0"], case_params["y0"], case_params["yaw0"])
        plt.plot(temp[:, 0], temp[:, 1], linestyle='--', linewidth = 0.4, color = 'green')
        temp = get_corners(car_params, case_params["xf"], case_params["yf"], case_params["yawf"])
        plt.plot(temp[:, 0], temp[:, 1], linestyle='--', linewidth = 0.4, color = 'red')
        plt.grid(linewidth = 0.2)
        plt.title(filename)
        plt.xlabel('X / m', fontsize = 14)
        plt.ylabel('Y / m', fontsize = 14)
    else:
        plt.axis('off')


    if save is True:
        plt.savefig(f"{filename}.png", dpi=500)
    if show is True:
        plt.show()

def write_case_csv(file_name, start_pose, end_pose, obstacles):
    """
    Write a CSV file that represents a scenario with a start pose, end pose, and obstacles.
    
    :param file_name: Name of the output CSV file.
    :param start_pose: Tuple or list of (x0, y0, theta0) for the start pose.
    :param end_pose: Tuple or list of (xf, yf, thetaf) for the end pose.
    :param obstacles: List of numpy arrays, where each array has shape [n, 2] representing obstacle vertices.
    """
    # Combine start and end poses into a single list
    start_end_pose = list(start_pose) + list(end_pose)
    
    # Number of obstacles
    obs_num = len(obstacles)
    
    # Number of vertices for each obstacle
    num_vertices = [obs.shape[0] for obs in obstacles]
    
    # Flatten the list of obstacle vertices
    flattened_obstacles = []
    for obs in obstacles:
        flattened_obstacles.extend(obs.flatten())
    
    # Combine everything into a single list
    csv_data = start_end_pose + [obs_num] + num_vertices + flattened_obstacles
    
    # Write to CSV file
    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_data)

if __name__ == "__main__":

    # case_num = 1
    # case_params = read(f"TPCAP_demo/BenchmarkCases/Case{case_num}.csv")

    # lets test create a case of reversing into a parking space
    obstacles = [
        np.array([ # left obstacle
            [-3.8-2.5, 2.5],
            [-3.8+2.5, 2.5],
            [-3.8+2.5, -2.5],
            [-3.8-2.5, -2.5],
        ]),
        np.array([ # right obstacle
            [3.8-2.5, 2.5],
            [3.8+2.5, 2.5],
            [3.8+2.5, -2.5],
            [3.8-2.5, -2.5]
        ]),
        np.array([
            [-6, 10-0.4],
            [6, 10-0.4],
            [6, 10+3.4],
            [-6, 10+3.4]
        ])
    ]

    start_pose = (-5.0, 4.35, 0.0)
    goal_pose = [0.0, 0.0, np.deg2rad(90)]

    write_case_csv('test_case.csv', start_pose, goal_pose, obstacles)

    
