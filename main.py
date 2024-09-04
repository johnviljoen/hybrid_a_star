from holonomic_cost_map import calculate_holonomic_cost_map
from grid import calculate_obstacle_grid_and_kdtree
from non_holonomic_search import plan
from tpcap_utils import plot_case
from transforms import get_corners

def main(case_params, save_name=None):

    # plots the true case so that we can see where the non discretized obstacles and start/end spots are
    plot_case(case_params, car_params, show=False, save=False, bare=False)

    # actually plan the trajectory
    traj = plan(planner_params, case_params, car_params)

    #### A lot of plotting to demonstrate whats happening ####

    # check out the data yourself!
    grid, grid_bounds, obstacle_kdtree = calculate_obstacle_grid_and_kdtree(planner_params, case_params)

    # this plots the grid with its obstacles
    plt.imshow(grid.T, cmap='cividis_r', origin='lower', extent=(case_params["xmin"], case_params["xmax"], case_params["ymin"], case_params["ymax"]))

    # some data to plot the holonomic cost map
    start_grid_index = (round(case_params["x0"] /  planner_params["xy_resolution"]) - grid_bounds["xmin"], \
                        round(case_params["y0"] /  planner_params["xy_resolution"]) - grid_bounds["ymin"], \
                        round(case_params["yaw0"] / planner_params["yaw_resolution"]) - grid_bounds["yawmin"])
    goal_grid_index = (round(case_params["xf"] /  planner_params["xy_resolution"]) - grid_bounds["xmin"], \
                       round(case_params["yf"] /  planner_params["xy_resolution"]) - grid_bounds["ymin"], \
                       round(case_params["yawf"] / planner_params["yaw_resolution"]) - grid_bounds["yawmin"])
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
    cbar.set_label('Holonomic Cost', rotation=270, labelpad=15)

    # plot the actual corners of the car throughout the trajectory
    plt.plot(traj[:,0], traj[:,1])
    for state in traj:
        corners = get_corners(car_params, state[0], state[1], state[2])
        plt.plot(corners[:,0], corners[:,1], 'magenta')

    # plot the start and end positions of center of mass of car in grid space
    plt.scatter([(goal_grid_index[0] + grid_bounds["xmin"]) * planner_params["xy_resolution"]],
                [(goal_grid_index[1] + grid_bounds["ymin"]) * planner_params["xy_resolution"]], linewidths=1.0, color="red")
    plt.scatter([(start_grid_index[0] + grid_bounds["xmin"]) * planner_params["xy_resolution"]],
                [(start_grid_index[1] + grid_bounds["ymin"]) * planner_params["xy_resolution"]], linewidths=1.0, color="green")
    
    # saving and closing...
    if save_name is None:
        plt.savefig('output.png', dpi=500)
    else:
        plt.savefig(save_name, dpi=500)
    plt.close()

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    from tpcap_utils import read
    from params import car_params, planner_params

    # this will run the TPCAP benchmark itself
    case_num = 1
    for case_num in range(1,21):
        scenario_name = f"Case{case_num}"
        case_params = read(f"tpcap_cases/{scenario_name}.csv")
        main(case_params, save_name=f"output/{scenario_name}")

    # do the manual reverse park example
    scenario_name = "reverse_park"
    case_params = read(f"manual_cases/{scenario_name}.csv")
    main(case_params, save_name=f"output/{scenario_name}")

    print('fin')
