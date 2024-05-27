import rosbag
from cv_bridge import CvBridge
from PIL import Image as PILImage
from sensor_msgs.msg import CompressedImage
import io

 
# 初始化cv_bridge
bridge = CvBridge()
 
# 指定bag文件和话题名称
bag_file = "/home/wzm/data/CTIP_data/SACSoN/Feb-13-2023-bww8-intloss/00000006.bag"
image_topic = "/fisheye_image/compressed"
 
# 打开bag文件
bag = rosbag.Bag(bag_file, "r")
 
# 创建消息过滤器
image_messages = []
 
# 遍历bag文件中的所有消息
for topic, msg, t in bag.read_messages():
    if topic == image_topic:
        image_messages.append((msg, t))
        img = PILImage.open(io.BytesIO(msg.data))
        print((img.size))
 
# 关闭bag文件
bag.close()