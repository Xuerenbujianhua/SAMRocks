def map_prediction_to_real_class(prediction_mapping,predicted_class):
    # 使用字典进行映射
    real_class = prediction_mapping.get(predicted_class, "Unknown")

    return real_class


# def map_prediction_to_real_class(sorted_classname_mapping,predicted_class):
#     # sorted_classname_mapping = dict(sorted(label_to_value.items()))
#
#     key, real_class = list(sorted_classname_mapping.items())[predicted_class]
#
#     return real_class
#这个匹配函数暂时无用