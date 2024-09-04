import numpy as np
import matplotlib.pyplot as plt
from heapdict import heapdict

from grid import calculate_obstacle_grid_and_kdtree
from reeds_shepp import reeds_shepp_node
from holonomic_cost_map import calculate_holonomic_cost_map
from kinematic_simulation import kinematic_simulation_node

def plan(planner_params, case_params, car_params):

    # gets the grid which represents obstacles, the grid bounds which tells us the x,y,yaw limits of the grid in
    # cartesian space which lets us map the grid to real x,y positions. The obstacle_kdtree is a scipy.spatial.KDTree
    # formed of the obstacle true x,y positions in the space, which lets us quickly query ball collisions which
    # lets us accelerate the algorithms collision detection in sparsely constrained areas.
    grid, grid_bounds, obstacle_kdtree = calculate_obstacle_grid_and_kdtree(planner_params, case_params)

    start_grid_index = (round(case_params["x0"] /  planner_params["xy_resolution"]) - grid_bounds["xmin"], \
                        round(case_params["y0"] /  planner_params["xy_resolution"]) - grid_bounds["ymin"], \
                        round(case_params["yaw0"]/ planner_params["yaw_resolution"]) - grid_bounds["yawmin"])

    goal_grid_index = (round(case_params["xf"] /  planner_params["xy_resolution"]) - grid_bounds["xmin"], \
                       round(case_params["yf"] /  planner_params["xy_resolution"]) - grid_bounds["ymin"], \
                       round(case_params["yawf"]/ planner_params["yaw_resolution"]) - grid_bounds["yawmin"])

    start_node = {
        "grid_index": start_grid_index,
        "traj": [[case_params["x0"], case_params["y0"], case_params["yaw0"]]],
        "cost": 0.0,
        "direction": 0.0,
        "steering_angle": 0.0,
        "parent_index": start_grid_index,
    }

    goal_node = {
        "grid_index": goal_grid_index,
        "traj": [[case_params["xf"], case_params["yf"], case_params["yawf"]]],
        "cost": 0.0,
        "direction": 0.0,
        "steering_angle": 0.0,
        "parent_index": goal_grid_index,
    }

    # Motion commands for a Non-Holonomic Robot like a Car or Bicycle (Trajectories using Steer Angle and Direction)
    action_space = []
    for i in np.linspace(-car_params["max_steer"], car_params["max_steer"], planner_params["steer_options"]):
        for j in np.linspace(-planner_params["max_movement"], planner_params["max_movement"], planner_params["movement_options"]):
            action_space.append([i,j])
    action_space = np.vstack(action_space)

    # calculate the costmap for the A* solution to the environment, which we guide the non-holonomic search with
    holonomic_cost_map = calculate_holonomic_cost_map(planner_params, goal_node, grid, grid_bounds)

    # Add start node to open Set
    open_set = {start_node["grid_index"]: start_node}
    closed_set = {}

    # Create a priority queue for acquiring nodes based on their cost's
    cost_queue = heapdict()

    # Add start mode into priority queue
    cost_queue[start_node["grid_index"]] = max(
        start_node["cost"], 
        planner_params["hybrid_cost"] * holonomic_cost_map[start_node["grid_index"][0]][start_node["grid_index"][1]]
    )
    counter = 0

    # Run loop while path is found or open set is empty
    while True:

        counter += 1
        print(counter)
        
        # if empty open set then no solution available
        if not open_set: 
            return None

        # bookkeeping
        current_node_index = cost_queue.popitem()[0]
        current_node = open_set[current_node_index]
        open_set.pop(current_node_index)
        closed_set[current_node_index] = current_node

        # is the reeds shepp solution collision free?
        rs_node = reeds_shepp_node(planner_params, car_params, current_node, goal_node, obstacle_kdtree)

        # if reeds shepp trajectory exists then we store solution and break the loop
        if rs_node: closed_set[rs_node["grid_index"]] = rs_node; break

        # edge case of directly finding the solution without reeds shepp, break loop
        if current_node_index == goal_node["grid_index"]: print("path found"); break

        # get all simulated nodes from the current node
        for action in action_space:
            simulated_node = kinematic_simulation_node(planner_params, case_params, car_params, current_node, action, obstacle_kdtree, grid_bounds)

            # check if path is valid
            if not simulated_node: 
                continue

            # Draw Simulated Node
            x,y,z =zip(*simulated_node["traj"])
            plt.plot(x, y, linewidth=0.3, color='g')

            # Check if simulated node is already in closed set
            if simulated_node["grid_index"] not in closed_set: 

                # Check if simulated node is already in open set, if not add it open set as well as in priority queue
                if simulated_node["grid_index"] not in open_set:
                    open_set[simulated_node["grid_index"]] = simulated_node
                    cost_queue[simulated_node["grid_index"]] = max(simulated_node["cost"], planner_params["hybrid_cost"] * holonomic_cost_map[simulated_node["grid_index"][0]][simulated_node["grid_index"][1]])
                else:
                    if simulated_node["cost"] < open_set[simulated_node["grid_index"]]["cost"]:
                        open_set[simulated_node["grid_index"]] = simulated_node
                        cost_queue[simulated_node["grid_index"]] = max(simulated_node["cost"], planner_params["hybrid_cost"] * holonomic_cost_map[simulated_node["grid_index"][0]][simulated_node["grid_index"][1]])


    # extract trajectory
    def backtrack(start_node, goal_node, closed_set):

        # Goal Node data
        current_node_index = goal_node["parent_index"]
        current_node = closed_set[current_node_index]
        x=[]
        y=[]
        yaw=[]

        # Iterate till we reach start node from goal node
        while current_node_index != start_node["grid_index"]:
            a, b, c = zip(*current_node["traj"])
            x += a[::-1] 
            y += b[::-1] 
            yaw += c[::-1]
            current_node_index = current_node["parent_index"]
            current_node = closed_set[current_node_index]

        traj = np.array([x[::-1], y[::-1], yaw[::-1]]).T
        return traj

    traj = backtrack(start_node, goal_node, closed_set)

    return traj


if __name__ == "__main__":

    from params import planner_params, car_params
    from tpcap_utils import read

    case_num = 1
    case_params = read(f"tpcap_cases/Case{case_num}.csv")

    traj = plan(planner_params, case_params, car_params)

    print('fin')

