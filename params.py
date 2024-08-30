import numpy as np

#### Car ####

car_params = {
    "wheel_base": 2.8,
    "width": 1.942,
    "front_hang": 0.96,
    "rear_hang": 0.929,
    "max_steer": 0.5,
}

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

#### Planner ####

planner_params = {
    "xy_resolution": 0.5,
    "yaw_resolution": np.deg2rad(5.0),
    "max_iter": 100,
    "reverse_cost": 10,         # used in reeds shepp cost in reeds_shepp.py
    "direction_change_cost": 1, # used in reeds shepp cost in reeds_shepp.py
    "steer_angle_cost": 1,      # used in reeds shepp cost in reeds_shepp.py
    "steer_angle_change_cost": 1 # ^ same
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

