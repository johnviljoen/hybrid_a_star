import heapq
import numpy as np

from holonomic_cost_map import calculate_holonomic_cost_map
from grid import calculate_obstacle_grid_and_kdtree

#### Example usage

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    from tpcap_utils import read, plot_case
    from params import car_params, planner_params
    from non_holonomic_search import plan

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

    plan(planner_params, case_params, car_params)

    print('fin')
