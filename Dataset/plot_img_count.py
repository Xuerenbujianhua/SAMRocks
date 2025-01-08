
import os
import matplotlib.pyplot as plt
# 配置中文字体，使用系统中的 SimHei 字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体 SimHei
plt.rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题


def get_image_counts(folder_path):
        categories = []
        image_counts = []

        # Iterate through folders and count images
        for category in os.listdir(folder_path):
            category_path = os.path.join(folder_path, category)

            if os.path.isdir(category_path):
                categories.append(category)
                image_counts.append(
                    len([img for img in os.listdir(category_path) if img.endswith(('jpg', 'png', 'jpeg'))]))

        return categories, image_counts

    # Function to plot the image counts in each category without grid and displaying the values on top of bars
def plot_image_counts(folder_path, title='Number of Images per Category', xlabel='Category',
                          ylabel='Number of Images'):
        categories, image_counts = get_image_counts(folder_path)

        plt.figure(figsize=(10, 6))
        bars = plt.bar(categories, image_counts, color='skyblue')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(rotation=45, ha="right")

        # Add Tags_For_testing_and_evaluation_purposes_only on top of the bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, int(yval), ha='center', va='bottom')

        plt.tight_layout()  # Adjust layout to prevent label overlap
        plt.grid(False)  # Remove grid lines

        plt.savefig(f'./results/bars_{title}.png',dpi=800)
        plt.show()
    # Sample call (path to be provided by the user, or replace with actual path if needed)
    # plot_image_counts('path_to_your_image_folder')

