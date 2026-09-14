# One detected shape.

int32   id            # tracker ID; 0 when tracking is disabled
string  label         # "circle", "triangle", "quad", "pentagon", ...
bool    occluded      # true when the pose is predicted, not observed

geometry_msgs/Point  center_px   # centroid in pixels (z unused)
geometry_msgs/Polygon outline    # traced outline, pixels (z unused)

bool    has_3d        # false if no scale reference has been seen yet
geometry_msgs/Point  position    # X, Y, Z in INCHES, camera frame
bool    depth_from_circle        # false = depth held from an earlier frame
float32 size_in                  # equivalent-circle radius, inches

float32 area_px
int32   vertices      # 0 for a circle
