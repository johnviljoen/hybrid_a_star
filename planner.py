import heapq
import numpy as np
import scipy.spatial
from matplotlib.path import Path

from holonomic_cost_map import calculate_holonomic_cost_map

#### Define base parameters that everything else is derived from 

planner_params = {
    "xy_resolution": 0.5,
    "yaw_resolution": np.deg2rad(5.0),
    "max_iter": 100
}

#### calculate parameters that depend on these params and the case we are solving ####

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


def run(planner_params, case_params, car_params):

    start_grid_index = (round(case_params["x0"] /  planner_params["xy_resolution"]), \
                        round(case_params["y0"] /  planner_params["xy_resolution"]), \
                        round(case_params["yaw0"]/ planner_params["yaw_resolution"]))

    goal_grid_index = (round(case_params["xf"] /  planner_params["xy_resolution"]), \
                       round(case_params["yf"] /  planner_params["xy_resolution"]), \
                       round(case_params["yawf"]/ planner_params["yaw_resolution"]))

    start_node = {
        "grid_index": start_grid_index,
        "traj": [[case_params["x0"], case_params["y0"], case_params["yaw0"]]],
        "cost": 0.0,
        "parent_index": start_grid_index,
    }

    goal_node = {
        "grid_index": goal_grid_index,
        "traj": [[case_params["xf"], case_params["y0"], case_params["yaw0"]]],
        "cost": 0.0,
        "parent_index": goal_grid_index,
    }

    # calculate the costmap for the A* solution to the environment, which we guide the non-holonomic search with
    holonomic_cost_map = calculate_holonomic_cost_map(planner_params, goal_node, grid, grid_bounds)





#### Example usage

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from tpcap_utils import read, plot_case
    from car import car_params

    case_num = 1
    case_params = read(f"tpcap_cases/Case{case_num}.csv")

    # plot to be sure
    plot_case(case_params, car_params, show=False, save=False, bare=False)

    # check out the data yourself!
    grid, grid_bounds, obstacle_kdtree = calculate_obstacle_grid_and_kdtree(planner_params, case_params)

    plt.imshow(grid.T, cmap='cividis_r', origin='lower', extent=(case_params["xmin"], case_params["xmax"], case_params["ymin"], case_params["ymax"]))

    # lets plot the holonomic cost
    start_grid_index = (round(case_params["x0"] /  planner_params["xy_resolution"]) - grid_bounds["xmin"], \
                        round(case_params["y0"] /  planner_params["xy_resolution"]) - grid_bounds["ymin"], \
                        round(case_params["yaw0"] / planner_params["yaw_resolution"]) - grid_bounds["yawmin"])

    goal_grid_index = (round(case_params["xf"] /  planner_params["xy_resolution"]) - grid_bounds["xmin"], \
                       round(case_params["yf"] /  planner_params["xy_resolution"]) - grid_bounds["ymin"], \
                       round(case_params["yawf"] / planner_params["yaw_resolution"]) - grid_bounds["yawmin"])

    start_node = {
        "grid_index": start_grid_index,
        "traj": [[case_params["x0"], case_params["y0"], case_params["yaw0"]]],
        "cost": 0.0,
        "parent_index": start_grid_index,
    }

    goal_node = {
        "grid_index": goal_grid_index,
        "traj": [[case_params["xf"], case_params["yf"], case_params["yawf"]]],
        "cost": 0.0,
        "parent_index": goal_grid_index,
    }

    # double check that the holonomic heuristic starts at end and goes from there.
    holonomic_cost_map = calculate_holonomic_cost_map(planner_params, goal_node, grid, grid_bounds)
    assert holonomic_cost_map[goal_grid_index[0], goal_grid_index[1]] == 0.0

    plt.imshow(holonomic_cost_map.T, cmap='gray', origin='lower', extent=(case_params["xmin"], case_params["xmax"], case_params["ymin"], case_params["ymax"]))
    cbar = plt.colorbar()
    cbar.set_label('Cost Value', rotation=270, labelpad=15)
    plt.scatter([(goal_grid_index[0] + grid_bounds["xmin"]) * planner_params["xy_resolution"]],
                [(goal_grid_index[1] + grid_bounds["ymin"]) * planner_params["xy_resolution"]], linewidths=1.0, color="red")
    plt.scatter([(start_grid_index[0] + grid_bounds["xmin"]) * planner_params["xy_resolution"]],
                [(start_grid_index[1] + grid_bounds["ymin"]) * planner_params["xy_resolution"]], linewidths=1.0, color="green")
    plt.savefig('obstacle_grid_overlayed_on_case.png')

    run(planner_params, case_params, car_params)

    print('fin')
