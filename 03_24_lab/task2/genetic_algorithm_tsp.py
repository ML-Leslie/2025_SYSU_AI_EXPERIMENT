import numpy as np
import matplotlib.pyplot as plt
import random
import time
from plot_tsp_coordinates import read_tsp_file

class GeneticAlgorithmTSP:
    def __init__(self, coordinates, pop_size=100, elite_size=20, mutation_rate=0.01, generations=500):
        self.coordinates = coordinates
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.best_routes = []
        self.best_distances = []
        
        # 提取坐标
        self.cities = [coord[0] for coord in coordinates]
        self.city_positions = {coord[0]: (coord[1], coord[2]) for coord in coordinates}
        
    def create_route(self):
        """创建一个随机路径"""
        return random.sample(self.cities, len(self.cities))
    
    def initial_population(self):
        """初始化种群"""
        return [self.create_route() for _ in range(self.pop_size)]
    
    def route_distance(self, route):
        """计算路径总距离"""
        total_distance = 0
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[(i + 1) % len(route)]
            x1, y1 = self.city_positions[from_city]
            x2, y2 = self.city_positions[to_city]
            total_distance += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return total_distance
    
    def rank_routes(self, population):
        """计算并排序种群中每条路径的适应度"""
        fitness_results = {i: self.route_distance(route) for i, route in enumerate(population)}
        return sorted(fitness_results.items(), key=lambda x: x[1])
    
    def selection(self, ranked_population):
        """选择操作"""
        selection_results = []
        # 精英选择
        for i in range(self.elite_size):
            selection_results.append(ranked_population[i][0])
        
        # 轮盘赌选择
        df = ranked_population.copy()
        # 距离取倒数作为适应度值，距离越短适应度越高
        fitness_sum = sum(1 / rank[1] for rank in df)
        rel_fitness = [(1 / rank[1]) / fitness_sum for rank in df]
        cumulative_prob = [sum(rel_fitness[:i+1]) for i in range(len(rel_fitness))]
        
        # 选择剩余个体
        for _ in range(self.pop_size - self.elite_size):
            pick = random.random()
            for i, cum_prob in enumerate(cumulative_prob):
                if pick <= cum_prob:
                    selection_results.append(df[i][0])
                    break
        
        return selection_results
    
    def mating_pool(self, population, selection_results):
        """构建交配池"""
        return [population[idx] for idx in selection_results]
    
    def breed(self, parent1, parent2):
        """交叉操作 - 使用部分映射交叉(PMX)"""
        child = [-1] * len(parent1)
        
        # 选择交叉的起始和结束点
        start, end = sorted(random.sample(range(len(parent1)), 2))
        
        # 从父亲1复制交叉区段
        for i in range(start, end + 1):
            child[i] = parent1[i]
        
        # 从父亲2填充剩余位置
        for i in range(len(parent2)):
            if parent2[i] not in child:
                # 找到一个尚未填充的位置
                for j in range(len(child)):
                    if child[j] == -1:
                        child[j] = parent2[i]
                        break
        
        return child
    
    def breed_population(self, mating_pool):
        """对整个交配池进行交叉操作"""
        children = []
        
        # 保留精英
        children.extend(mating_pool[:self.elite_size])
        
        # 对剩余个体进行交叉
        pool = random.sample(mating_pool, len(mating_pool))
        for i in range(self.elite_size, self.pop_size):
            children.append(self.breed(pool[i - self.elite_size], pool[i - self.elite_size + 1 if i < self.pop_size - 1 else 0]))
        
        return children
    
    def mutate(self, individual):
        """变异操作 - 使用交换变异"""
        for swapped in range(len(individual)):
            if random.random() < self.mutation_rate:
                swap_with = int(random.random() * len(individual))
                individual[swapped], individual[swap_with] = individual[swap_with], individual[swapped]
        return individual
    
    def mutate_population(self, population):
        """对整个种群进行变异操作"""
        # 精英不变异
        mutated_pop = population[:self.elite_size]
        
        # 对剩余个体进行变异
        for ind in population[self.elite_size:]:
            mutated_pop.append(self.mutate(ind))
        
        return mutated_pop
    
    def next_generation(self, current_pop):
        """生成下一代种群"""
        # 计算适应度并排序
        ranked_pop = self.rank_routes(current_pop)
        
        # 选择
        selection_results = self.selection(ranked_pop)
        
        # 构建交配池
        mating_pool = self.mating_pool(current_pop, selection_results)
        
        # 交叉
        children = self.breed_population(mating_pool)
        
        # 变异
        next_generation = self.mutate_population(children)
        
        return next_generation
    
    def run(self):
        """运行遗传算法"""
        print("开始运行遗传算法...")
        
        # 初始化种群
        population = self.initial_population()
        
        # 找到初始种群中的最佳路径
        best_route_idx = self.rank_routes(population)[0][0]
        best_route = population[best_route_idx]
        best_distance = self.route_distance(best_route)
        
        self.best_routes.append(best_route)
        self.best_distances.append(best_distance)
        
        print(f"初始最佳距离: {best_distance:.2f}")
        
        # 进化过程
        start_time = time.time()
        for i in range(self.generations):
            # 动态调整变异率
            self.mutation_rate = self.adaptive_mutation_rate(i)
            population = self.next_generation(population)
            population = self.next_generation(population)
            
            # 找到当前代的最佳路径
            current_best_idx = self.rank_routes(population)[0][0]
            current_best_route = population[current_best_idx]
            current_best_distance = self.route_distance(current_best_route)
            
            self.best_routes.append(current_best_route)
            self.best_distances.append(current_best_distance)
            
            if (i + 1) % 50 == 0 or i == 0 or i == self.generations - 1:
                print(f"第 {i+1} 代 - 最佳距离: {current_best_distance:.2f}")
        
        end_time = time.time()
        print(f"遗传算法运行完成，耗时: {end_time - start_time:.2f}秒")
        
        # 最终结果
        final_best_route = self.best_routes[-1]
        final_best_distance = self.best_distances[-1]
        
        print(f"最终最佳距离: {final_best_distance:.2f}")
        
        return final_best_route, final_best_distance
    
    def plot_progress(self):
        """绘制进化过程中的最佳距离变化"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.best_distances)
        plt.title('遗传算法优化过程')
        plt.xlabel('代数')
        plt.ylabel('最佳距离')
        plt.grid(True)
        plt.show()
    
    def plot_best_route(self, route):
        """绘制最佳路径"""
        plt.figure(figsize=(12, 10))
        
        # 绘制城市点
        x_coords = [self.city_positions[city][0] for city in route]
        y_coords = [self.city_positions[city][1] for city in route]
        
        # 添加起点到终点的连线，形成闭环
        x_coords.append(x_coords[0])
        y_coords.append(y_coords[0])
        
        # 绘制路径
        plt.plot(x_coords, y_coords, 'b-', linewidth=0.8)
        
        # 绘制城市点
        plt.scatter(x_coords[:-1], y_coords[:-1], c='red', s=50)
        
        # 标记起点（绿色）
        plt.scatter(x_coords[0], y_coords[0], c='green', s=100, marker='*')
        
        # 添加城市编号标签
        for i, city in enumerate(route):
            plt.annotate(str(city), (x_coords[i], y_coords[i]), xytext=(5, 5), 
                         textcoords='offset points', fontsize=9)
        
        plt.title(f'最优路径 (总距离: {self.route_distance(route):.2f})')
        plt.xlabel('X 坐标')
        plt.ylabel('Y 坐标')
        plt.grid(True)
        plt.show()
    def adaptive_mutation_rate(self, generation):
        """根据迭代进程动态调整变异率"""
        max_rate = 0.1
        min_rate = 0.001
        return max_rate - (max_rate - min_rate) * (generation / self.generations)

def main():
    # 读取TSP文件
    file_path = r"c:\Users\Leslie\Desktop\学习资料\人工智能\人工智能实验\03_24_lab\task2\dj38.tsp"
    coordinates = read_tsp_file(file_path)
    
    # 参数设置
    pop_size = 100
    elite_size = 20
    mutation_rate = 0.01
    generations = 1000
    
    # 创建并运行遗传算法
    ga = GeneticAlgorithmTSP(coordinates, pop_size, elite_size, mutation_rate, generations)
    best_route, best_distance = ga.run()
    
    # 可视化结果
    ga.plot_progress()
    ga.plot_best_route(best_route)
    

if __name__ == "__main__":
    main()
