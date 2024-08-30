import numpy as np
import scipy.spatial
from matplotlib.path import Path

from car import car_params

#### Define base parameters that everything else is derived from 

planner_params = {
    "xy_resolution": 0.5,
    "yaw_resolution": np.deg2rad(5.0),
    "max_iter": 100
}

#### calculate parameters that depend on these params and the case we are solving ####

def calculate_obstacle_grid_and_kdtree(planner_params, case_params):
    grid_width = int((planner_params["xmax"] - planner_params["xmin"]) // planner_params["xy_resolution"] + 1)
    grid_height = int((planner_params["ymax"] - planner_params["ymin"]) // planner_params["xy_resolution"] + 1)

    grid = np.zeros((grid_width, grid_height))
    obstacle_x_idx = []
    obstacle_y_idx = []
    for obs in case_params["obs"]:
        path = Path(obs)
        
        # Create a meshgrid for the bounding box
        x_range = np.arange(case_params['xmin'], case_params["xmax"], planner_params["xy_resolution"])
        y_range = np.arange(case_params["ymin"], case_params["ymax"], planner_params["xy_resolution"])
        xv, yv = np.meshgrid(x_range, y_range)
        points = np.vstack((xv.flatten(), yv.flatten())).T
        
        # Check which points are inside the obstacle
        inside = path.contains_points(points)
        
        # Get grid coordinates of the inside points
        for point in points[inside]:
            grid_x_idx = int((point[0] - case_params["xmin"]) // planner_params["xy_resolution"])
            grid_y_idx = int((point[1] - case_params["ymin"]) // planner_params["xy_resolution"])
            grid[grid_x_idx, grid_y_idx] = 1
            obstacle_x_idx.append(grid_x_idx)
            obstacle_y_idx.append(grid_y_idx)
    obstacle_x_idx = np.array(obstacle_x_idx)
    obstacle_y_idx = np.array(obstacle_y_idx)

    # calculate the map bounds in terms of the grid
    grid_xmin = round(case_params["xmin"] / planner_params["xy_resolution"])
    grid_ymin = round(case_params["ymin"] / planner_params["xy_resolution"])

    obstacle_x = (obstacle_x_idx + 0.5) * planner_params["xy_resolution"] + grid_xmin * planner_params["xy_resolution"]
    obstacle_y = (obstacle_y_idx + 0.5) * planner_params["xy_resolution"] + grid_ymin * planner_params["xy_resolution"]
    obstacle_kdtree = scipy.spatial.KDTree([[x, y] for x, y in zip(obstacle_x, obstacle_y)])

    return grid, obstacle_kdtree

if __name__ == "__main__":
    