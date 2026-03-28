import os
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm

def crop_shiprsimagenet(
    img_dir,
    ann_dir,
    save_dir,
    min_size=16
):
    os.makedirs(save_dir, exist_ok=True)

    for xml_file in tqdm(os.listdir(ann_dir)):
        if not xml_file.endswith('.xml'):
            continue

        xml_path = os.path.join(ann_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        img_name = root.find('filename').text
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert('RGB')

        for i, obj in enumerate(root.findall('object')):
            cls_name = obj.find('name').text

            bbox = obj.find('bndbox')
            xmin = int(float(bbox.find('xmin').text))
            ymin = int(float(bbox.find('ymin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymax = int(float(bbox.find('ymax').text))

            # 过滤太小目标
            if (xmax - xmin) < min_size or (ymax - ymin) < min_size:
                continue

            crop = image.crop((xmin, ymin, xmax, ymax))

            # 保存到 ImageFolder 结构
            class_dir = os.path.join(save_dir, cls_name)
            os.makedirs(class_dir, exist_ok=True)

            save_name = f"{xml_file.replace('.xml','')}_{i}.jpg"
            crop.save(os.path.join(class_dir, save_name))