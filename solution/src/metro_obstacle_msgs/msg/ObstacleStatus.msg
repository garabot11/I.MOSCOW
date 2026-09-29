# Result of the obstacle detector for one input cloud.
std_msgs/Header header     # stamp = input cloud header.stamp (sensor clock), frame_id = input cloud frame
uint8 STATUS_UNKNOWN=0     # no / invalid / stale input, or the track surface is not observable
uint8 STATUS_CLEAR=1       # no confirmed obstacle inside the clearance box
uint8 STATUS_DETECTED=2    # at least one confirmed obstacle
uint8 status
float32 distance_m         # nearest confirmed obstacle: Euclidean range from the lidar origin (NaN if none)
float32 forward_m          # nearest confirmed obstacle: along-track distance (NaN if none)
uint32 frame_index         # running index of processed clouds in this node instance
uint32 num_candidates      # single-frame candidates before temporal confirmation
Obstacle[] obstacles       # all candidates of this frame (confirmed flag inside)
float32 processing_ms      # wall-clock time spent in the detector for this cloud
float32 rail_level_m       # estimated rail level at the sensor position (sensor frame height), NaN if unknown
float32 track_offset_m     # estimated sensor offset from the track centre-line, NaN if unknown
string diagnostics         # short human-readable note (reason for UNKNOWN, resets, drops)
