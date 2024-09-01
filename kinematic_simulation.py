import numpy as np

from collision import is_traj_valid
from transforms import get_corners
from reeds_shepp import pi_2_pi

def _simulated_path_cost(planner_params, current_node, action):
    
    # prior node cost
    cost = current_node["cost"]

    # distance cost
    if action[1] > 0:
        cost += planner_params["kinematic_simulation_length"] * action[1]
    else:
        cost += planner_params["kinematic_simulation_length"] * action[1] * planner_params["reverse_cost"]

    # direction change cost
    if np.sign(current_node["direction"]) != np.sign(action[1]):
        cost += planner_params["direction_change_cost"]

    # steering angle cost
    cost += action[0] * planner_params["steer_angle_cost"]

    # steering angle change cost
    cost += np.abs(action[0] - current_node["steering_angle"]) * planner_params["steer_angle_change_cost"]

    return cost

def _is_valid(case_params, car_params, traj, obstacle_kdtree):

    # check if node is out of map bounds
    np_traj = np.array(traj)
    for state in np_traj:
        points = get_corners(car_params, state[0], state[1], state[2])
        for point in points:
            if point[0]<=case_params["xmin"] or point[0]>=case_params["xmax"] or \
            point[1]<=case_params["ymin"] or point[1]>=case_params["ymax"]:
                return False

    # Check if Node is colliding with an obstacle
    if not is_traj_valid(car_params, traj, case_params["obs"], obstacle_kdtree):
        return False
    return True

def kinematic_simulation_node(planner_params, case_params, car_params, current_node, action, obstacle_kdtree, grid_bounds):

    # Simulate node using given current Node and Motion Commands
    traj = []
    angle = pi_2_pi(current_node["traj"][-1][2] + action[1] * planner_params["kinematic_simulation_step"] / car_params["wheel_base"] * np.tan(action[0]))
    traj.append([current_node["traj"][-1][0] + action[1] * planner_params["kinematic_simulation_step"] * np.cos(angle),
                current_node["traj"][-1][1] + action[1] * planner_params["kinematic_simulation_step"] * np.sin(angle),
                pi_2_pi(angle + action[1] * planner_params["kinematic_simulation_step"] / car_params["wheel_base"] * np.tan(action[0]))])
    for i in range(int((planner_params["kinematic_simulation_length"]/planner_params["kinematic_simulation_step"]))-1):
        traj.append([traj[i][0] + action[1] * planner_params["kinematic_simulation_step"] * np.cos(traj[i][2]),
                    traj[i][1] + action[1] * planner_params["kinematic_simulation_step"] * np.sin(traj[i][2]),
                    pi_2_pi(traj[i][2] + action[1] * planner_params["kinematic_simulation_step"] / car_params["wheel_base"] * np.tan(action[0]))])
    
    # Find grid index
    grid_index = (round(traj[-1][0]/planner_params["xy_resolution"] - grid_bounds["xmin"]), \
                  round(traj[-1][1]/planner_params["xy_resolution"] - grid_bounds["ymin"]), \
                  round(traj[-1][2]/planner_params["yaw_resolution"] - grid_bounds["yawmin"]))

    if not _is_valid(case_params, car_params, traj, obstacle_kdtree):
        return None

    # Calculate Cost of the node
    cost = _simulated_path_cost(planner_params, current_node, action)

    return {
        "grid_index": grid_index,
        "traj": traj,
        "cost": cost,
        "direction": action[1],
        "steering_angle": action[0],
        "parent_index": current_node["grid_index"],
    }
    