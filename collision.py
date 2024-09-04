import numpy as np
import scipy.spatial
from transforms import get_corners

def is_traj_valid(car_params, traj, obs_kdtree):

    for state in traj:
        x, y, yaw = state

        # check bubble first
        cx = x + car_params["wheel_base"]/2 * np.cos(yaw)
        cy = y + car_params["wheel_base"]/2 * np.sin(yaw)
        points_in_obstacle = obs_kdtree.query_ball_point([cx, cy], car_params["bubble_radius"])

        # skip past points not close to obstacles by just checking bubble
        if not points_in_obstacle:
            continue

        # check corners based on grid
        if rectangle_check(car_params, x, y, yaw,
                               [obs_kdtree.data[i][0] for i in points_in_obstacle], [obs_kdtree.data[i][1] for i in points_in_obstacle]):
            return False  # collision

    return True

def rectangle_check(car_params, x, y, yaw, ox, oy, eps=1e-8):
    # transform obstacles to base link frame
    rot = scipy.spatial.transform.Rotation.from_euler('z', yaw).as_matrix()[0:2, 0:2]
    for iox, ioy in zip(ox, oy):
        tx = iox - x
        ty = ioy - y
        converted_xy = np.stack([tx, ty]).T @ rot
        rx, ry = converted_xy[0], converted_xy[1]
        corners = get_corners(car_params, 0, 0, 0)[:-1]
        rx_max = corners[1,0]
        rx_min = corners[0,0]
        ry_max = corners[2,1]
        ry_min = corners[0,1]
        crit = (rx > rx_max or rx < rx_min or ry > ry_max or ry < ry_min)
        if not crit:
            return True # collision
        
    return False  # no collision
