from pathlib import Path
import subprocess, os, signal, time, json
import rclpy
from metro_obstacle_msgs.msg import ObstacleStatus
out=Path('/results'); out.mkdir(exist_ok=True)
records=[]
for bag,topic in [('roundT_doubleT','/lidar_points'),('doubleT_obstacle','/sensing/lidar/hesai128/pointcloud')]:
    rclpy.init()
    observer=rclpy.create_node('audit_status_observer')
    statuses=[]
    sub=observer.create_subscription(ObstacleStatus,'/obstacles/status',lambda m: statuses.append({'status':int(m.status),'stamp_ns':m.header.stamp.sec*10**9+m.header.stamp.nanosec,'frame_index':int(m.frame_index),'distance_m':float(m.distance_m),'monotonic':time.monotonic()}),10)
    nf=(out/(bag+'_node.log')).open('w'); pf=(out/(bag+'_play.log')).open('w')
    node=subprocess.Popen(['ros2','launch','metro_obstacle_detector','detector.launch.py','input_topic:='+topic,'log_path:=/results/'+bag+'_stream.jsonl'],stdout=nf,stderr=subprocess.STDOUT,start_new_session=True)
    def spin(seconds):
        stop=time.monotonic()+seconds
        while time.monotonic()<stop:
            rclpy.spin_once(observer,timeout_sec=0.05)
    spin(4)
    initial_count=len(statuses)
    player=subprocess.Popen(['ros2','bag','play','/data/'+bag,'--rate','1.0','--delay','2','--read-ahead-queue-size','10'],stdout=pf,stderr=subprocess.STDOUT)
    start=time.monotonic()
    while player.poll() is None:
        rclpy.spin_once(observer,timeout_sec=0.05)
        if time.monotonic()-start>90:
            player.terminate(); raise RuntimeError('playback timeout')
    spin(3)
    os.killpg(node.pid,signal.SIGINT)
    try: node.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(node.pid,signal.SIGTERM); node.wait(timeout=5)
    nf.close();pf.close()
    rows=[json.loads(line) for line in (out/(bag+'_stream.jsonl')).read_text().splitlines()]
    rec={'bag':bag,'player_exit':player.returncode,'node_exit':node.returncode,'statuses_before_any_input':initial_count,'status_messages_total':len(statuses),'stale_unknown_statuses':sum(x['status']==0 for x in statuses),'logged_frames':len(rows),'first_input_stamp':rows[0]['header_time_ns'] if rows else None,'last_input_stamp':rows[-1]['header_time_ns'] if rows else None}
    records.append(rec);print(json.dumps(rec),flush=True)
    (out/(bag+'_observed_status.json')).write_text(json.dumps(statuses,indent=2))
    observer.destroy_node();rclpy.shutdown()
(out/'stream_summary.json').write_text(json.dumps(records,indent=2))
