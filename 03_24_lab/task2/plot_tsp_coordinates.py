import matplotlib.pyplot as plt
import numpy as np

# 读取TSP文件
def read_tsp_file(file_path):
    coordinates = []
    with open(file_path, 'r') as file:
        # 跳过文件头部分，直到找到NODE_COORD_SECTION
        for line in file:
            if line.strip() == "NODE_COORD_SECTION":
                break
        
        # 读取坐标部分
        for line in file:
            if line.strip() == "EOF":
                break
            parts = line.strip().split()
            if len(parts) >= 3:  # 确保行包含城市编号和两个坐标
                city_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                coordinates.append((city_id, x, y))
    
    return coordinates

# 绘制坐标点
def plot_cities(coordinates):
    # 提取x和y坐标
    city_ids = [coord[0] for coord in coordinates]
    x_coords = [coord[1] for coord in coordinates]
    y_coords = [coord[2] for coord in coordinates]
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 绘制红色点
    plt.scatter(x_coords, y_coords, color='red', s=50)
    
    # 添加城市标签
    for i, city_id in enumerate(city_ids):
        plt.annotate(str(city_id), (x_coords[i], y_coords[i]), xytext=(5, 5), 
                     textcoords='offset points')
    
    # 添加标题和标签
    plt.title('城市坐标可视化')
    plt.xlabel('X 坐标')
    plt.ylabel('Y 坐标')
    plt.grid(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 显示图形
    plt.show()

# 主函数
def main():
    file_path = r"c:\Users\Leslie\Desktop\学习资料\人工智能\人工智能实验\03_24_lab\task2\dj38.tsp"
    coordinates = read_tsp_file(file_path)
    plot_cities(coordinates)

if __name__ == "__main__":
    main()
