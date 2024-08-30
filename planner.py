import numpy as np
import scipy.spatial
from matplotlib.path import Path

#### Define base parameters that everything else is derived from 

planner_params = {
    "xy_resolution": 0.5,
    "yaw_resolution": np.deg2rad(5.0),
    "max_iter": 100
}

#### calculate parameters that depend on these params and the case we are solving ####

def calculate_obstacle_grid_and_kdtree(planner_params, case_params):
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
    grid_xmin = round(case_params["xmin"] / planner_params["xy_resolution"])
    grid_ymin = round(case_params["ymin"] / planner_params["xy_resolution"])

    obstacle_x = (obstacle_x_idx + 0.5) * planner_params["xy_resolution"] + grid_xmin * planner_params["xy_resolution"]
    obstacle_y = (obstacle_y_idx + 0.5) * planner_params["xy_resolution"] + grid_ymin * planner_params["xy_resolution"]
    obstacle_kdtree = scipy.spatial.KDTree([[x, y] for x, y in zip(obstacle_x, obstacle_y)])

    return grid, obstacle_kdtree

def run(planner_params, case_params, car_params):

    start_grid_index = [round(case_params["x0"] /  planner_params["xy_resolution"]), \
                        round(case_params["y0"] /  planner_params["xy_resolution"]), \
                        round(case_params["yaw0"]/ planner_params["yaw_resolution"])]

    goal_grid_index = [round(case_params["xf"] /  planner_params["xy_resolution"]), \
                       round(case_params["yf"] /  planner_params["xy_resolution"]), \
                       round(case_params["yawf"]/ planner_params["yaw_resolution"])]

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

def holonomic_cost(planner_params, goal_non_holonomic_node, grid, ):

    grid_index = [round(goal_non_holonomic_node["traj"][-1][0]/planner_params["xy_resolution"]), round(goal_non_holonomic_node["traj"][-1][1]/planner_params["xy_resolution"])]
    
    goal_holonomic_node = {
        "grid_index": grid_index,
        "cost": 0,
        "parent_index": grid_index
    }

    holonomic_motion_commands = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]

    def is_holonomic_node_valid(neighbour_node, grid)



#### Example usage

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from tpcap_utils import read, plot_case
    from car import car_params

    case_num = 1
    case_params = read(f"tpcap_cases/Case{case_num}.csv")

    # plot to be sure
    plot_case(case_params, car_params, show=False, save=False)

    # check out the data yourself!
    grid, obstacle_kdtree = calculate_obstacle_grid_and_kdtree(planner_params, case_params)

    plt.imshow(grid.T, cmap='gray', origin='lower', extent=(case_params["xmin"], case_params["xmax"], case_params["ymin"], case_params["ymax"]))
    plt.savefig('obstacle_grid_overlayed_on_case.png')

    print('fin')
