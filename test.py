import rosbag
from cv_bridge import CvBridge
from PIL import Image as PILImage
from sensor_msgs.msg import CompressedImage
import io

 
# 初始化cv_bridge
bridge = CvBridge()
 
# 指定bag文件和话题名称
bag_file = "/home/wzm/data/real_casia/rgb/rgb_test/2024-05-23-15-10-11.bag"
image_topic = "/camera/color/image_raw/compressed"
 
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