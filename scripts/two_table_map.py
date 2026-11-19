"""Known static table/floor navigation map; no object poses or simulator API."""
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

rclpy.init()
node=Node('two_table_known_map',parameter_overrides=[Parameter('use_sim_time',value=True)])
qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
publisher=node.create_publisher(OccupancyGrid,'/map',qos)
static=StaticTransformBroadcaster(node)
transform=TransformStamped();transform.header.frame_id='map';transform.child_frame_id='odom';transform.transform.rotation.w=1.
static.sendTransform(transform)
msg=OccupancyGrid();msg.header.frame_id='map';msg.info.resolution=.05
msg.info.width=240;msg.info.height=240;msg.info.origin.position.x=-5.0;msg.info.origin.position.y=-6.0;msg.info.origin.orientation.w=1.
# Table geometry is the same explicitly configured geometry used by MoveIt.
data=[]
for row in range(240):
 y=-6+(row+.5)*.05
 for col in range(240):
  x=-5+(col+.5)*.05
  occupied=any(abs(x-1.58)<=.25 and abs(y-ty)<=.325 for ty in (-1.,1.))
  data.append(100 if occupied else 0)
msg.data=data

def publish():
 msg.header.stamp=node.get_clock().now().to_msg();publisher.publish(msg)
node.create_timer(1.,publish);publish()
try:rclpy.spin(node)
except KeyboardInterrupt:pass
finally:node.destroy_node();rclpy.shutdown()
