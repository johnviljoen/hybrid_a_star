import numpy as np

#### base parameters from which everything else is derived ####

car_params = {
    "wheel_base": 2.8,
    "width": 1.942,
    "front_hang": 0.96,
    "rear_hang": 0.929,
    "max_steer": 0.5,
}

#### parameters that are functions of other parameters ####

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

#### Example usage / testing ####

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from transforms import get_corners, get_bubble

    x = 0.1
    y = -3.0
    yaw = np.deg2rad(45)

    corners = get_corners(car_params, x, y, yaw)
    bubble = get_bubble(car_params, x, y, yaw, num_points=100)

    plt.plot(corners[:,0], corners[:,1])
    plt.plot(bubble[:,0], bubble[:,1])
    plt.savefig('test_params_transforms_corners_bubble.png')


