import numpy as np
import matplotlib.pyplot as plt
from TSP import GeneticAlgTSP
import time

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体

def plot_convergence(distances):
    """绘制收敛图，显示迭代过程中最短距离的变化"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(distances)), distances, 'b-')  # 修复变量名
    plt.title('TSP 优化收敛过程')
    plt.xlabel('迭代次数')
    plt.ylabel('路径距离')
    plt.grid(True)
    plt.savefig('convergence.svg')
    plt.show()

def plot_route(cities, route):
    """绘制最短路径连线图"""
    # 调整城市索引（从1开始）
    route_indices = [i-1 for i in route]
    
    # 获取路径上城市的坐标
    route_coords = cities[route_indices]
    
    # 添加起点回到终点的连线（闭合路径）
    route_coords = np.vstack([route_coords, route_coords[0]])
    
    plt.figure(figsize=(10, 10))
    # 绘制城市点
    plt.scatter(cities[:, 0], cities[:, 1], c='red', s=50)
    
    # 绘制路径连线
    plt.plot(route_coords[:, 0], route_coords[:, 1], 'b-', linewidth=1)
    
    # 添加城市编号标注
    for i, coord in enumerate(cities):
        plt.annotate(f"{i+1}", (coord[0], coord[1]), xytext=(5, 5), 
                     textcoords='offset points')
    
    plt.title('TSP 最优路径')
    plt.xlabel('X 坐标')
    plt.ylabel('Y 坐标')
    plt.grid(True)
    plt.savefig('optimal_route.svg')
    plt.show()

if __name__ == "__main__":
    # 指定TSP文件路径
    tsp_file = r"c:\Users\Leslie\Desktop\学习资料\人工智能\人工智能实验\03_24_lab\task2\dj38.tsp"
    
    # 创建遗传算法求解器实例
    tsp_solver = GeneticAlgTSP(file=tsp_file)
    
    # 运行算法
    print("正在运行遗传算法求解TSP问题...")
    best_route, best_distance = tsp_solver.run()
    
    print(f"最优路径: {best_route}")
    print(f"最优路径距离: {best_distance:.2f}")
    
    # 绘制收敛图
    print("绘制收敛曲线...")
    plot_convergence(tsp_solver.best_distances)
    
    # 绘制最优路径
    print("绘制最优路径...")
    plot_route(tsp_solver.cities, best_route)
