import numpy as np
from matplotlib.path import Path
import scipy.spatial


def calculate_obstacle_grid_and_kdtree(planner_params, case_params):
    """
    Here we calculate the grid and where the obstacles lie within it, according to the resolutions of
    the planner. We also create a scipy.spatial.KDTree data structure for the discretized obstacle positions
    in the grid, this allows fast calculation of ball distances to all obstacles at runtime as a first 
    check for collision. If a collision is then detected according to this we calculate the true collisions.
    This ends up saving a lot of time when we are not close to obstacles.

    Args:
        planner_params (Dict): defining parameters of the hybrid A* planner
        case_params (Dict): defining parameters of the environment we are in

    Returns:
        grid (): _description_
        grid_bounds (): _description_
        obstacle_kdtree (): _description_
    """
    grid_width = int((case_params["xmax"] - case_params["xmin"]) // planner_params["xy_resolution"] + 1)
    grid_height = int((case_params["ymax"] - case_params["ymin"]) // planner_params["xy_resolution"] + 1)

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
    grid_bounds = {}
    grid_bounds["xmax"] = round(case_params["xmax"] / planner_params["xy_resolution"])
    grid_bounds["xmin"] = round(case_params["xmin"] / planner_params["xy_resolution"])
    grid_bounds["ymax"] = round(case_params["ymax"] / planner_params["xy_resolution"])
    grid_bounds["ymin"] = round(case_params["ymin"] / planner_params["xy_resolution"])
    grid_bounds["yawmax"] = round(2*np.pi / planner_params["yaw_resolution"])
    grid_bounds["yawmin"] = round(0.0 / planner_params["yaw_resolution"])

    obstacle_x = (obstacle_x_idx + 0.5) * planner_params["xy_resolution"] + grid_bounds["xmin"] * planner_params["xy_resolution"]
    obstacle_y = (obstacle_y_idx + 0.5) * planner_params["xy_resolution"] + grid_bounds["ymin"] * planner_params["xy_resolution"]
    obstacle_kdtree = scipy.spatial.KDTree([[x, y] for x, y in zip(obstacle_x, obstacle_y)])

    return grid, grid_bounds, obstacle_kdtree