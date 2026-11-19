"""Independent camera-space instance scoring for the fixed two-table simulator."""
from itertools import permutations
import numpy as np


def score_choice(initial, names, camera, boxes, selected, direction):
    # Fixed world-to-odom spawn offset; camera extrinsics and intrinsics belong to
    # the captured RGB-D frame. No data from this function goes to the controller.
    transform = np.array(camera['odom_from_camera'])
    k = np.array(camera['k']).reshape(3, 3)
    projections = {}
    for name in names:
        point = np.array(initial[name]) + [.8, 0., 0.]
        optical = transform[:3, :3].T @ (point - transform[:3, 3])
        if optical[2] <= 0:
            raise ValueError("Cube is behind the source camera")
        uvw = k @ optical
        projections[name] = (uvw[:2] / uvw[2]).tolist()
    axis = int(direction in ('topmost', 'bottommost'))
    ordered = sorted(names, key=lambda name: projections[name][axis])
    if direction == 'middle' and len(names) % 2 == 0:
        raise ValueError('Middle requires an odd number of objects')
    rank = len(ordered)//2 if direction == 'middle' else -1 if direction in ('rightmost', 'bottommost') else 0
    expected = ordered[rank]
    result = dict(target_model=expected, projected_centers_px=projections, selection_correct=False)
    if len(boxes) != len(names) or selected not in boxes:
        return result
    # Account for common image translation from planar odometry drift. Match all
    # centers one-to-one and reject ambiguous or poorly localized assignments.
    centers = np.array([[(x1+x2)/2, (y1+y2)/2] for x1,y1,x2,y2 in boxes])
    points = np.array([projections[name] for name in names])
    aligned = points - points.mean(axis=0) + centers.mean(axis=0)
    distances = np.linalg.norm(centers[:, None] - aligned[None, :], axis=2)
    order = min(permutations(range(len(names))), key=lambda perm: sum(distances[i,j]**2 for i,j in enumerate(perm)))
    matched = [names[j] for j in order]
    unambiguous = all(distances[i,j] < min(distances[i,k] for k in range(len(names)) if k!=j)
                      and distances[i,j] <= max(boxes[i][2]-boxes[i][0], boxes[i][3]-boxes[i][1])/2
                      for i,j in enumerate(order))
    chosen = matched[boxes.index(selected)]
    result.update(matched_instances=matched, match_residuals_px=[float(distances[i,j]) for i,j in enumerate(order)],
                  unique_match=bool(unambiguous), chosen_model=chosen, selection_correct=bool(unambiguous and chosen==expected))
    return result
