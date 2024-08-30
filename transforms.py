import numpy as np

def get_corners(car_params, x, y, theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    points = np.array([
        [-car_params["rear_hang"], -car_params["width"] / 2, 1],
        [ car_params["front_hang"] + car_params["wheel_base"], -car_params["width"] / 2, 1],
        [ car_params["front_hang"] + car_params["wheel_base"], car_params["width"] / 2, 1],
        [-car_params["rear_hang"],  car_params["width"] / 2, 1],
        [-car_params["rear_hang"], -car_params["width"] / 2, 1],
    ]).dot(np.array([
        [cos_theta, -sin_theta, x],
        [sin_theta, cos_theta, y],
        [0, 0, 1]
    ]).transpose())
    return points[:, 0:2]

def get_bubble(car_params, x, y, yaw, num_points=100):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=True)
    x_cg = x + (- car_params["rear_hang"] + car_params["total_length"] / 2) * np.cos(yaw) # get center of mass for bubble center
    y_cg = y + (- car_params["rear_hang"] + car_params["total_length"] / 2) * np.sin(yaw) # get center of mass for bubble center
    x_points = x_cg + car_params["bubble_radius"] * np.cos(angles)
    y_points = y_cg + car_params["bubble_radius"] * np.sin(angles)
    return np.vstack([x_points, y_points]).T
