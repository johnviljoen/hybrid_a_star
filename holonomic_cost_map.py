import heapq
import numpy as np

def calculate_holonomic_cost_map(planner_params, goal_non_holonomic_node, grid, grid_bounds):

    """A* planner

    This function reduces the hybrid A* problem of searching across {x,y,yaw} space, to {x,y} space. This
    is *almost* equivalent to the traditional A* algorithm and its result is used as a heuristic to guide the 
    hybrid A* algorithm. The difference is that we search across the entire grid rather than finding a single path.
    This lets us compare our cost in the non holonomic search with the holonomic result on the fly using the
    precomputed holonomic cost map.

    Returns:
        np.ndarray: grid cost values according to A*
    """
    
    grid_index = (round(goal_non_holonomic_node["traj"][-1][0]/planner_params["xy_resolution"]) - grid_bounds["xmin"], 
                  round(goal_non_holonomic_node["traj"][-1][1]/planner_params["xy_resolution"]) - grid_bounds["ymin"])
    
    goal_holonomic_node = {
        "grid_index": grid_index,
        "cost": 0,
        "parent_index": grid_index
    }

    holonomic_motion_commands = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]

    def is_holonomic_node_valid(neighbour_node, grid):

        # check environment bounds
        if neighbour_node["grid_index"][0] < 0 or \
           neighbour_node["grid_index"][0] >= grid.shape[0] or \
           neighbour_node["grid_index"][1] < 0 or \
           neighbour_node["grid_index"][1] >= grid.shape[1]:
            return False
        
        # check no obstacle collisions - grid is 1 where obstacles are
        if grid[neighbour_node["grid_index"][0]][neighbour_node["grid_index"][1]]:
            return False
        
        return True
    
    # only tuples are hashable apparently - no lists 4 me >:(
    open_set = {goal_holonomic_node["grid_index"]: goal_holonomic_node}
    closed_set = {}

    priority_queue = []
    heapq.heappush(priority_queue, (goal_holonomic_node["cost"], goal_holonomic_node["grid_index"]))

    while True:
        if not open_set: break

        _, current_node_index = heapq.heappop(priority_queue)
        current_node = open_set[current_node_index]
        open_set.pop(current_node_index)
        closed_set[current_node_index] = current_node

        for action in holonomic_motion_commands:

            neighbour_holonomic_node = {
                "grid_index": (current_node["grid_index"][0] + action[0], \
                               current_node["grid_index"][1] + action[1]),
                "cost": current_node["cost"] + np.hypot(action[0], action[1]), # euclidean cost added
                "parent_index": current_node_index
            }

            if not is_holonomic_node_valid(neighbour_holonomic_node, grid): continue

            if neighbour_holonomic_node["grid_index"] not in closed_set:
                if neighbour_holonomic_node["grid_index"] in open_set:
                    if neighbour_holonomic_node["cost"] < open_set[neighbour_holonomic_node["grid_index"]]["cost"]:
                        open_set[neighbour_holonomic_node["grid_index"]]['cost'] = neighbour_holonomic_node["cost"]
                        open_set[neighbour_holonomic_node["grid_index"]]["parent_index"] = neighbour_holonomic_node["parent_index"]
                else:
                    open_set[neighbour_holonomic_node["grid_index"]] = neighbour_holonomic_node
                    heapq.heappush(priority_queue, (neighbour_holonomic_node['cost'], neighbour_holonomic_node["grid_index"]))
    
    holonomic_cost = np.ones_like(grid) * np.inf
    for nodes in closed_set.values():
        holonomic_cost[nodes["grid_index"][0]][nodes["grid_index"][1]] = nodes["cost"]

    return holonomic_cost