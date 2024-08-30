import numpy as np
from transforms import get_corners

def is_traj_valid(car_params, traj, polygons, obs_kdtree):

    collision = False

    for state in traj:
        x, y, yaw = state

        # check bubble first
        cx = x + car_params["wheel_base"]/2 * np.cos(yaw)
        cy = y + car_params["wheel_base"]/2 * np.sin(yaw)
        pointsInObstacle = obs_kdtree.query_ball_point([cx, cy], car_params["bubble_radius"])

        # skip past points not close to obstacles by just checking bubble
        if not pointsInObstacle:
            continue 
        
        #### check true collision ####

        # real corners of car at this state in the trajectory
        points = get_corners(car_params, x, y, yaw)[:-1]

        # check if any point of car is in any polygon
        violation = np.array([0])
        for point in points:
            violation += is_point_in_polygons(point, polygons)
        vehicle_in_obstacle = np.clip(violation, a_min=0, a_max=1)

        # check if any point of polygon is in car
        violation = np.array([0])
        vec_polygons = np.vstack(polygons)
        for point in vec_polygons:
            violation += is_point_in_polygons(point, [points])
        obstacle_in_vehicle = np.clip(violation, a_min=0, a_max=1)

        # assert both criteria
        collision = np.logical_or(obstacle_in_vehicle, vehicle_in_obstacle)

        if collision == True:
            return False

    return True


def is_point_in_polygons(point, polygons):

    point_shifted_right = np.copy(point)
    point_shifted_right[0] += 1e2
    line = np.vstack([point, point_shifted_right])

    for _polygon in polygons:

        # Calculate ai, bi, ci for each edge of the polygon
        ai = _polygon[:-1, 1:2] - _polygon[1:, 1:2]
        bi = _polygon[1:, 0:1] - _polygon[:-1, 0:1]
        ci = _polygon[:-1, 0:1] * _polygon[1:, 1:2] - _polygon[1:, 0:1] * _polygon[:-1, 1:2]
        aibi = np.hstack([ai, bi])

        # Calculate a, b, c for the line
        a = line[0:1, 1:2] - line[1:2, 1:2]
        b = line[1:2, 0:1] - line[0:1, 0:1]
        c = line[0:1, 0:1] * line[1:2, 1:2] - line[1:2, 0:1] * line[0:1, 1:2]
        ab = np.hstack([a, b])

        # Create AB matrices and check determinants for parallel lines and filter
        AB = np.concatenate((np.tile(ab, (aibi.shape[0], 1, 1)), aibi[:, np.newaxis]), axis=1)
        det_AB = np.linalg.det(AB)
        valid_indices = np.where(det_AB != 0)[0]
        valid_AB = AB[valid_indices]
        valid_ci = ci[valid_indices]
        if valid_indices[0] == 0:
            polygon_valid_indices = np.append(valid_indices, 0)
        else:
            polygon_valid_indices = valid_indices
        _polygon = _polygon[polygon_valid_indices]

        # Form C matrices
        C = np.concatenate((np.tile(c, (valid_ci.shape[0], 1, 1)), valid_ci[:, np.newaxis]), axis=1)

        # Compute intersections for valid AB and C
        intersections = np.linalg.inv(valid_AB) @ C
        intersections = - intersections[...,0] # flatten last dimension and invert
        # check that intersections are ON the line segments or not: 0 < dot(b>a, c>a) < (b>a)**2 + eps

        distances_squared = np.sum(np.square(_polygon[:-1] - _polygon[1:]), axis=1)

        # (x - x1)(x2 - x1) + (y - y1)(y2 - y1)
        dot_product = (intersections[:,0] - _polygon[:-1,0]) * (_polygon[1:,0] - _polygon[:-1,0]) + (intersections[:,1] - _polygon[:-1,1]) * (_polygon[1:,1] - _polygon[:-1,1])

        # for the intersection to be on the line segment of the obstacle edge the dot product < squared distance between the points defining edge and > 0
        valid_indices = np.logical_and(dot_product < distances_squared, dot_product > 0)
        valid_intersections = intersections[valid_indices]

        # filter out intersections not along the beam segment
        distance = lambda x, y: np.linalg.norm(x-y, axis=1, ord=2) # np.sqrt(np.sum((x-y)**2, axis=1))
        valid_indices = np.abs(distance(line[0], valid_intersections) + distance(line[1], valid_intersections) - distance(line[0:1], line[1:2])) <= 1e-5
        intersections = valid_intersections[valid_indices]     

        if len(intersections) % 2 != 0:
            return True

    return False

def grid_traj_collision_check(car_params, traj, obstacles, obstacle_kdtree):

    collision = False
    for state in traj:
        x, y, yaw = state

        # check bubble fast
        cx = x + car_params["wheel_base"]/2 * np.cos(yaw)
        cy = y + car_params["wheel_base"]/2 * np.sin(yaw)
        pointsInObstacle = obstacle_kdtree.query_ball_point([cx, cy], car_params["bubble_radius"])

        if not pointsInObstacle:
            continue 

        # check true collision
        points = get_corners(x, y, yaw)[:-1]

        # check if any point of car is in any polygon
        violation = np.array([0])
        for point in points:
            violation += is_point_in_polygons(point, obstacles)
        vehicle_in_obstacle = np.clip(violation, a_min=0, a_max=1)

        # check if any point of polygon is in car
        violation = np.array([0])
        vec_obstacles = np.vstack(obstacles)
        for point in vec_obstacles:
            violation += is_point_in_polygons(point, [points])
        obstacle_in_vehicle = np.clip(violation, a_min=0, a_max=1)

        # check both criteria
        collision = np.logical_or(obstacle_in_vehicle, vehicle_in_obstacle)

        if collision == True:
            return True

    return False