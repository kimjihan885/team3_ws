from rosbags.rosbag2 import Reader
from rosbags.typesys import get_typestore, Stores
import cv2
import numpy as np
import os

bag_path = '/home/wego/team3_ws/my_bag'
output_dir = '/home/wego/team3_ws/raw_image/output_images'
os.makedirs(output_dir, exist_ok=True)

typestore = get_typestore(Stores.ROS2_HUMBLE)

with Reader(bag_path) as reader:
    total = 0
    saved = 0
    for connection, timestamp, rawdata in reader.messages():
        if connection.topic == '/camera/color/image_raw':
            if total % 10 == 0:  # 15장 중 1장만 저장
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                img = np.array(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, -1)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'{output_dir}/frame_{saved:04d}.jpg', img_bgr)
                saved += 1
                print(f'저장: frame_{saved:04d}.jpg')
            total += 1

print(f'완료: 총 {saved}장 저장 ({total}장 중)')