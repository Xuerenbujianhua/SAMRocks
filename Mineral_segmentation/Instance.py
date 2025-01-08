import cv2
import os


class Instance:
    def __init__(self,instances_id, image_path, mask_path, bbox, original_image_index,measuring_scale=1 ,label_id=None,label_path=None,image_floder_id=0):
        self.instances = None
        self.instances_id = instances_id
        self.image_path = image_path
        self.mask_path = mask_path
        self.label_path = label_path
        self.bbox = bbox
        self.cluster_id = None
        self._image = None  # Lazy loading of image
        self._mask = None
        self._label = None  # Lazy loading of mask
        self._label_id = label_id  #
        self._color_mask = None  # Lazy loading of mask
        self.colored_mask_path = None  # 新增的属性，用于存储彩色掩码路径
        self.feature_path = None
        self.measuring_scale = measuring_scale #比例尺
        self.original_image_name = os.path.basename(image_path)  # 存储原始图片名称
        self.original_image_index = original_image_index
        self.image_floder_id = image_floder_id
    def __len__(self):
        """返回实例的数量"""
        return len(self.instances)

    def get_image(self):
        if self._image is None:
            self._image = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)  # 从磁盘加载图像
        return self._image

    def get_mask(self):

        if self._mask is None:
            self._mask = cv2.imread(self.mask_path, cv2.IMREAD_UNCHANGED)  # 从磁盘加载掩码
        return self._mask

    def get_colored_mask(self):
        if self._color_mask is None:
            self._color_mask = cv2.imread(self.colored_mask_path, cv2.IMREAD_UNCHANGED)  # 从磁盘加载掩码
        return self._color_mask

    def get_label(self):

        if self._label is None:
            self._label = cv2.imread(self.label_path, cv2.IMREAD_GRAYSCALE)  # 从磁盘加载掩码
        return self._label


    def set_feature_path(self, path):
        """
        设置该实例对应的特征路径。

        参数：
        - path: 特征文件的路径。
        """
        self.feature_path = path

    def clear_memory(self):
        """手动清理已加载的图像和掩码，以释放内存。"""
        self._image = None
        self._mask = None

    def add_dynamic_attribute(self, attr_name, value):
        """
        动态为实例添加新属性。

        参数：
        - attr_name: 字符串，新属性的名称。
        - value: 新属性的值。
        """
        setattr(self, attr_name, value)

    def get_dynamic_attribute(self, attr_name):
        """
        获取动态添加的属性值。

        参数：
        - attr_name: 字符串，属性名称。

        返回：
        - 动态属性的值，如果不存在返回 None。
        """
        return getattr(self, attr_name, None)


class InstanceManager:
    def __init__(self, instance_info):
        self.instances = instance_info

    def get_instance(self, index):
        """获取指定索引处的实例"""
        if index < 0 or index >= len(self.instances):
            raise IndexError("Index out of bounds")
        return self.instances[index]

    def get_all_images(self):
        """获取所有实例的图像（按需加载）"""
        return [instance.get_image() for instance in self.instances]

    def get_all_masks(self):
        """获取所有实例的掩码（按需加载）"""
        return [instance.get_mask() for instance in self.instances]

    def clear_all_memory(self):
        """清理所有实例的图像和掩码内存"""
        for instance in self.instances:
            instance.clear_memory()

    def add_attribute_to_all(self, attr_name, value):
        """
        为所有实例添加相同的新属性。

        参数：
        - attr_name: 新属性的名称。
        - value: 新属性的值。
        """
        for instance in self.instances:
            instance.add_dynamic_attribute(attr_name, value)

    def get_all_original_image_names(self):
        """
        获取所有实例的原始图片名称。

        返回：
        - original_image_names: 原始图片名称列表。
        """
        return [instance.original_image_name for instance in self.instances]

    def get_instances_by_attribute(self, attr_name, value):
        """
        根据动态属性值获取符合条件的实例列表。

        参数：
        - attr_name: 要匹配的属性名称。
        - value: 要匹配的属性值。

        返回：
        - 符合条件的实例列表。
        """
        return [instance for instance in self.instances if instance.get_dynamic_attribute(attr_name) == value]

    def list_all_attributes(self):
        """
        列出所有实例的动态属性。

        返回：
        - 所有实例的动态属性名和值的字典列表。
        """
        attributes = []
        for instance in self.instances:
            instance_attributes = {attr: getattr(instance, attr) for attr in vars(instance)}
            attributes.append(instance_attributes)
        return attributes




