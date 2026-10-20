# One detected object inside the train clearance box.
uint32 track_id            # temporal track identifier (0 = none)
bool confirmed             # confirmed by the temporal filter (confirm_hits within confirm_window)
float32 distance_m         # Euclidean range from the lidar origin to the nearest reliable surface of the object
float32 forward_m          # along-track distance of the nearest surface
float32 lateral_offset_m   # lateral offset of the object centre from the estimated track centre-line (+ = +lateral axis)
float32 height_above_rail_m
float32 size_forward_m
float32 size_lateral_m
float32 size_height_m
uint32 num_points          # returns (voxelised) supporting the object
float32 confidence         # 0..1, point support relative to the expectation at that range
geometry_msgs/Point center # object centre in the cloud's sensor frame (header.frame_id of ObstacleStatus)
