import rosbag
from cv_bridge import CvBridge
from PIL import Image as PILImage
from sensor_msgs.msg import CompressedImage
import io
import cv2
import numpy as np
 
# 初始化cv_bridge
bridge = CvBridge()
 
# 指定bag文件和话题名称
bag_file = "/home/wzm/data/real_casia/red_test/indoor.bag"
image_topic = "/galaxy_camera/galaxy_camera/image_raw/compressed"
 
# 打开bag文件
bag = rosbag.Bag(bag_file, "r")
 
# 创建消息过滤器
image_messages = []
 
# 遍历bag文件中的所有消息
for topic, msg, t in bag.read_messages():
    if topic == image_topic:
        image_messages.append((msg, t))
        img = PILImage.open(io.BytesIO(msg.data))
        img = PILImage.Image.crop(img, (200, 200, 2448-200, 2048-300))
        img.show()
        cv2.waitKey(0)
        break
 
# 关闭bag文件
bag.close()