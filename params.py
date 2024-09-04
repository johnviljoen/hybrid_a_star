import numpy as np

#### Car ####

car_params = {
    "wheel_base": 2.8,
    "width": 1.942,
    "front_hang": 0.96,
    "rear_hang": 0.929,
    "max_steer": 0.5,
}

#### Testing equivalent setup to other hybrid a* ####

# LB = 2.3
# LF = 2.3
# max_steer = np.deg2rad(40)
# total_length = LB + LF
# wheel_base = 2.7
# width = 1.85
# front_hang = LF - wheel_base/2
# rear_hang = LB - wheel_base/2

# car_params = {
#     "wheel_base": wheel_base,
#     "width": width,
#     "front_hang": front_hang,
#     "rear_hang": rear_hang,
#     "max_steer": max_steer,
# }

#### Testing end ####

# bubble for fast detection of potential collisions later on
car_params["total_length"] = car_params["rear_hang"] + car_params["wheel_base"] + car_params["front_hang"]
car_params["bubble_radius"] = np.hypot(car_params["total_length"] / 2, car_params["width"] / 2)

# origin is defined around the rear axle, default orientiation is facing east
car_params["corners"] = np.array([
    [car_params["wheel_base"] + car_params["front_hang"], car_params["width"] / 2], # front left
    [- car_params["rear_hang"], car_params["width"] / 2], # back left
    [- car_params["rear_hang"], - car_params["width"] / 2], # back right
    [car_params["wheel_base"] + car_params["front_hang"], - car_params["width"] / 2] # front right
])

car_params["center_to_front"] = car_params["wheel_base"]/2 + car_params["front_hang"]
car_params["center_to_back"] = car_params["wheel_base"]/2 + car_params["rear_hang"]

#### Planner ####

planner_params = {
    "xy_resolution": 0.5,
    "yaw_resolution": np.deg2rad(5.0),
    "steer_options": 5,
    "movement_options": 2,   # resolution of action space at every timestep of the search
    "max_movement": 1.0,          # max movement forwards or backwards at every timestep of the search
    "max_iter": 100,
    "reverse_cost": 1.0,         # used in reeds shepp cost in reeds_shepp.py
    "direction_change_cost": 1.0, # used in reeds shepp cost in reeds_shepp.py
    "steer_angle_cost": 0.5,      # used in reeds shepp cost in reeds_shepp.py
    "steer_angle_change_cost": 0.5, # ^ same
    "rs_step_size": 1.0,          # the rs step size used in calculating the reeds shepp trajectory at the leaves
    "hybrid_cost": 1.0,            # used in holonomic_cost_map weighting in non_holonomic_search.py
    "kinematic_simulation_length": 1,   # for the kinematic simulation node in non_holonomic_search.py
    "kinematic_simulation_step": 1.0
}

#### Tests ####

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from transforms import get_corners, get_bubble

    x = 0.1
    y = -3.0
    yaw = np.deg2rad(45)

    #### Test Car Params ####

    corners = get_corners(car_params, x, y, yaw)
    bubble = get_bubble(car_params, x, y, yaw, num_points=100)

    plt.plot(corners[:,0], corners[:,1])
    plt.plot(bubble[:,0], bubble[:,1])
    plt.savefig('test_params_transforms_corners_bubble.png')

    #### Test Planner Params ####

    # no need lmao

