# Spatial clarification in ten live Gazebo transfers

The robot resolved the requested position in **10/10** trials. Gazebo verified transfer of the requested cube in **7/10**.

[MoveIt diagnosis and paired reruns of the three failed scenes](#pickup-approach-investigation-and-paired-reruns)

Each trial began with three same-color cubes on Table A and an empty Table B. Ten different layouts covered red, green, and blue cubes and five camera-relative positions. The scripted operator supplied a reply only after the graph paused. The camera views and prompts used spatial descriptions.

| Measure | Result |
|---|---:|
| Returned three candidate boxes | 10/10 |
| Asked the operator | 10/10 |
| Recorded the prescribed dialogue with three candidates | 10/10 |
| Asked again after an underspecified reply, then accepted a position | 2/2 |
| Selected the intended physical instance | 10/10 |
| Confirmed a held object using wrist RGB-D after lift | 7/10 |
| Independently verified transfer and stationary bystanders | 7/10 |

| Trial | Cube | Operator replies, in order | Detections | Correct selection | Verified transfer |
|---|---|---|---:|---|---|
| [spatial_01](spatial_01/selection.png) | red | “left most” | 3 | Yes | Yes |
| [spatial_02](spatial_02/selection.png) | blue | “the blue cube on the right” | 3 | Yes | Yes |
| [spatial_03](spatial_03/selection.png) | green | “pick up the middle green cube” | 3 | Yes | Yes |
| [spatial_04](spatial_04/selection.png) | red | “the cube at the top” | 3 | Yes | No |
| [spatial_05](spatial_05/selection.png) | blue | “the bottom-most blue cube” | 3 | Yes | Yes |
| [spatial_06](spatial_06/selection.png) | green | “the one on the left” | 3 | Yes | No |
| [spatial_07](spatial_07/selection.png) | red | “pick up the right-most red cube” | 3 | Yes | Yes |
| [spatial_08](spatial_08/selection.png) | blue | “the blue cube” → “the one in the middle” | 3 | Yes | Yes |
| [spatial_09](spatial_09/selection.png) | green | “the green cube” → “top most” | 3 | Yes | No |
| [spatial_10](spatial_10/selection.png) | red | “pick up the bottom red cube” | 3 | Yes | Yes |

10/10 trials used a single bounding-box model call across the instruction and follow-up replies.

Median wall time from verifier start to verdict was 5.5 minutes across 10 trials (range 2.3–5.6 minutes). This includes model calls and simulated robot motion, with up to two trials running concurrently.

## Trials without a verified transfer

- **spatial_04** (topmost). MoveIt could not find a valid arm configuration for the approach pose. The cube was not lifted.
- **spatial_06** (leftmost). MoveIt could not find a valid arm configuration for the approach pose. The cube was not lifted.
- **spatial_09** (topmost). MoveIt could not find a valid arm configuration for the approach pose. The cube was not lifted.

## Scoring

The verifier projects independently measured Gazebo cube centers into the source camera frame. It derives the requested position from those projections and pairs all detections with the projected centers, allowing a common translation for odometry drift. Selection requires a unique geometric match to the expected cube. Physical success additionally requires a lift over 10 cm, a verified release on Table B, and at most 1.5 cm of movement by each bystander.

Each case was measured once. These results describe the ten scripted scenes in simulation. The clarification and transfer rates both use all ten trials as their denominator.

## Pickup approach investigation and paired reruns

In each of the three failed baseline trials, the requested cube was correctly identified. RRTConnect could not sample a valid goal state for the downward-facing flange at the hover waypoint, about 17.3 cm above the grasp. The arm stopped before descending.

A motion-free MoveIt IK replay reconstructed the approximate parked base pose from each recorded camera transform, the rigid wrist mount and the commanded overview pose. At the original dock the solver returned NO_IK_SOLUTION for all three hovers, and again with the hover lowered by 5.3 cm. The hover from successful control scene spatial_03 solved. This isolates kinematic reach from obstacle checks. The live transfers below used normal collision checking. [IK measurements](pickup-reach/ik.json)

Two pickup positions were tested. First, moving the dock from (1.10, 0.00) m to (1.22, −0.14) m in the map frame gave IK solutions for all three hovers and grasps. In the red topmost and green leftmost trials, the wrist camera confirmed the cube after lifting, but the independent pose trace showed that it dropped during the drive to Table B.

The **centerline pickup dock** at **(1.36, 0.00) m** advances the base 26 cm along the original travel line. All three hover and grasp endpoints solved in the IK replay at that dock. Nav2 parks before the camera captures the source image, so the operator chooses from the view used to compute the grasp. The wrist orientation, grasp height and collision checks are the same in all three conditions.

MoveIt executed the approach and proceeded to descent in **3/3** centerline trials. Verified transfers in the matched scenes were **0/3 at the original dock**, **1/3 at the offset dock**, and **2/3 at the centerline dock**.

| Scene and choice | Original dock | Offset dock | Centerline dock | Peak rise at centerline |
|---|---|---|---|---:|
| spatial_04 (topmost red) | Stopped at approach | [Lifted, then lost in transit](pickup-reach/offset/spatial_04/verification.json) | [Verified transfer](pickup-reach/centerline/spatial_04/verification.json) | 24.9 cm |
| spatial_06 (leftmost green) | Stopped at approach | [Lifted, then lost in transit](pickup-reach/offset/spatial_06/verification.json) | [Verified transfer](pickup-reach/centerline/spatial_06/verification.json) | 23.9 cm |
| spatial_09 (topmost green) | Stopped at approach | [Verified transfer](pickup-reach/offset/spatial_09/verification.json) | [Reached cube, pickup unverified](pickup-reach/centerline/spatial_09/verification.json) | 0.2 cm |

Each condition used fresh simulations with matching world checksums, instructions, and scripted replies. Peak rise is the maximum measured change in cube height over the whole trial. Each of the three previously failed scenes was repeated once per revised dock. The original ten-scene rate remains **7/10**. The seven original successes were not rerun with the revised dock. [Paired measurements](pickup-reach/comparison.json)

In spatial_09, the centerline failure occurred after the gripper closed. The topmost cube rose 1.8 mm, below the 10 cm lift criterion. The wrist check stopped the task before transport. The pose trace establishes that the grasp failed to lift the cube. Jaw contact and grasp alignment would need separate measurements to identify the cause.
