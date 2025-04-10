import numpy as np
import matplotlib.pyplot as plt
import random
import time

class GeneticAlgTSP:
    def __init__(self, file:str, pop_size=100, elite_size=10, mutation_rate=0.02, generations=200):

        self.cities = np.array([coord[1:] for coord in self.read_tsp_file(file)])
        self.population = [] # 种群
        self.pop_size = pop_size # 种群大小
        self.elite_size = elite_size # 精英个体数量
        self.mutation_rate = mutation_rate # 变异率
        self.generations = generations # 迭代次数
        self.best_routes = []
        self.best_distances = []

        self.no_improvement_count = 0 # 无改进计数器

        self.distances_matrix = self.calculate_distances_matrix(self.cities) # 计算距离矩阵
        self.route_distance_cache = {}  # 添加路径距离缓存

    # 计算距离矩阵
    def calculate_distances_matrix(self, coordinates):
        print("计算距离矩阵...")
        n = len(coordinates)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(coordinates[i] - coordinates[j])
                distances[i][j] = dist
                distances[j][i] = dist
        print("距离矩阵计算完成")
        return distances
    
    # 读取TSP文件
    def read_tsp_file(self, file):
        coordinates = []
        with open(file, 'r') as file:
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
    
    # 创建一个随机路径 （从1到n的随机排列）
    def create_route(self):
        return random.sample(range(1,len(self.cities)+1), len(self.cities))
    
    # 初始化种群，大小为 pop_size
    def initial_population_another(self):
        self.population = [self.create_route() for _ in range(self.pop_size)]
        return self.population

    # 贪心算法生成路径，从一个城市开始，每次选择最近的未访问城市
    def greedy_route(self, start_city):
        
        # 调整为从0开始的索引
        start_idx = start_city - 1
        
        route = [start_city]  # 城市编号从1开始
        unvisited = set(range(1, len(self.cities) + 1))
        unvisited.remove(start_city)
        
        current_idx = start_idx
        
        while unvisited:
            # 找到距离当前城市最近的未访问城市
            next_city = min(unvisited, 
                            key=lambda city: self.distances_matrix[current_idx][city-1])
            
            route.append(next_city)
            unvisited.remove(next_city)
            current_idx = next_city - 1
            
        return route

    # 初始化种群，使用贪心算法
    def initial_population(self):
        print("初始化种群...")
        population = []

        for _ in range(self.pop_size):
            start_city = random.randint(1, len(self.cities))
            route = self.greedy_route(start_city)
            population.append(route)
        
        self.population = population
        print("种群初始化完成")
        return population
    
    # 计算路径总距离
    def route_distance(self, route):
        # 尝试从缓存中获取结果
        route_tuple = tuple(route)
        if route_tuple in self.route_distance_cache:
            return self.route_distance_cache[route_tuple]
        
        # Store route length and distance matrix locally to avoid attribute lookups in the loop
        route_len = len(route)
        dist_matrix = self.distances_matrix
        total = 0
        
        # Use local variables inside the loop to reduce lookups
        for i in range(route_len):
            from_idx = route[i] - 1
            to_idx = route[(i + 1) % route_len] - 1
            total += dist_matrix[from_idx][to_idx]
        
        # 存入缓存
        self.route_distance_cache[route_tuple] = total
        return total
    
    # 计算路径列表的适应度并返回按适应度排序的索引
    def rank_routes(self, routes):
        # 计算每条路径的距离
        fitness_results = {i: self.route_distance(route) for i, route in enumerate(routes)}
        
        # 返回按距离排序的索引
        return sorted(fitness_results.keys(), key=lambda x: fitness_results[x])
    
    # 交叉操作
    def cross(self, p1, p2):
        # 随机选择交叉点
        start = random.randint(0, len(p1) - 1)
        end = random.randint(0, len(p1) - 1)
        if start > end:
            start, end = end, start

        # 创建交叉部分的映射关系
        p1_cross_section = p1[start:end]
        p2_cross_section = p2[start:end]
        
        # 创建映射字典
        mapping_p1_to_p2 = {}
        mapping_p2_to_p1 = {}
        
        # 为交叉部分的城市建立映射关系
        for i in range(start, end):
            mapping_p1_to_p2[p1[i]] = p2[i]
            mapping_p2_to_p1[p2[i]] = p1[i]
        
        # 初始化子代
        child1 = [-1] * len(p1)
        child2 = [-1] * len(p2)
        
        # 复制交叉部分
        for i in range(start, end):
            child1[i] = p2[i]
            child2[i] = p1[i]
        
        # 填充子代1的剩余部分
        for i in range(len(p1)):
            if i < start or i >= end:
                city = p1[i]
                while city in p2_cross_section:
                    city = mapping_p2_to_p1[city]
                child1[i] = city
        
        # 填充子代2的剩余部分
        for i in range(len(p2)):
            if i < start or i >= end:
                city = p2[i]
                while city in p1_cross_section:
                    city = mapping_p1_to_p2[city]
                child2[i] = city

        return child1, child2
    
    # 变异操作
    def mutate(self, route):
        # 创建路径的副本，避免修改原始路径
        route_copy = route.copy()
        
        # 随机选择变异类型
        mutation_type = random.choice(["reverse", "swap", "insert", "scramble"])
        
        if mutation_type == "reverse":  # 2-opt变异，反转一段路径
            index1 = random.randint(0, len(route_copy) - 1)
            index2 = random.randint(0, len(route_copy) - 1)
            if index1 > index2:
                index1, index2 = index2, index1
            route_copy[index1:index2+1] = reversed(route_copy[index1:index2+1])
        
        elif mutation_type == "swap":  # 随机交换两个城市
            index1, index2 = random.sample(range(len(route_copy)), 2)
            route_copy[index1], route_copy[index2] = route_copy[index2], route_copy[index1]
        
        elif mutation_type == "insert":  # 将一个城市插入到另一个位置
            index1, index2 = random.sample(range(len(route_copy)), 2)
            city = route_copy.pop(index1)
            route_copy.insert(index2, city)
        
        elif mutation_type == "scramble":  # 随机打乱一小段路径
            index1 = random.randint(0, len(route_copy) - 1)
            index2 = random.randint(0, len(route_copy) - 1)
            if index1 > index2:
                index1, index2 = index2, index1
            segment = route_copy[index1:index2+1]
            random.shuffle(segment)
            route_copy[index1:index2+1] = segment
        
        return route_copy
    
    # 局部搜索
    def local_search(self, route, max_iterations=20, max_time=30):
        start_time = time.time()
        best_route = route.copy()
        best_distance = self.route_distance(best_route)
        iteration = 0
        
        while iteration < max_iterations:
            improved = False
            iteration += 1
            
            # 检查是否超时
            if time.time() - start_time > max_time:
                break
            
            # 随机采样边进行检查，而不是检查所有边
            edge_samples = min(500, len(route) // 2)  # 最多检查500条边
            edges_to_check = random.sample(range(1, len(route) - 2), edge_samples) # 排除首尾城市
            
            for i in edges_to_check:
                # 每次只检查一小部分j值
                j_range = min(50, len(route) - i)
                j_values = [i + j for j in range(2, j_range)]
                
                for j in j_values:
                    # 尝试2-opt交换：反转i到j之间的路径
                    new_route = best_route.copy()
                    new_route[i:j+1] = reversed(new_route[i:j+1])
                    new_distance = self.route_distance(new_route)
                    
                    # 如果有改进，更新最佳路径
                    if new_distance < best_distance:
                        best_distance = new_distance
                        best_route = new_route
                        improved = True
                        break  # 发现改进后立即跳出内层循环
                
                if improved:
                    break  # 发现改进后立即跳出外层循环
            
            # 如果没有改进，提前终止
            if not improved:
                break
        
        return best_route

    # 动态变异率
    def get_dynamic_mutation_rate(self, generation):
        # 最大和最小变异率
        max_rate = self.mutation_rate * 2
        min_rate = self.mutation_rate / 2
        
        # 根据迭代次数和无改进次数调整
        if self.no_improvement_count > 5:  # 长时间无改进，提高变异率
            return max_rate
        elif generation > 70:  # 后期降低变异率
            return min_rate
        else:
            # 线性衰减
            return max_rate - (max_rate - min_rate) * (generation / 100)
    # 锦标赛选择
    def elite_tournament_selection(self, current_gen):

        selected = []
        
        # 计算所有个体适应度并排序
        sorted_indices = self.rank_routes(current_gen)
        
        # 精英选择: 直接保留最优秀的elite_size个体
        elite_indices = sorted_indices[:self.elite_size]
        selected.extend([current_gen[i] for i in elite_indices])
        
        # 锦标赛选择: 填充剩余位置
        while len(selected) < self.pop_size:
            # 随机选择锦标赛参与者
            tournament_size = 5
            tournament = random.sample(range(len(current_gen)), tournament_size)
            
            # 找出锦标赛中最好的个体
            best_idx = min(tournament, key=lambda i: self.route_distance(current_gen[i]))
            selected.append(current_gen[best_idx])
        
        return selected

    # 生成下一代种群
    def next_generation(self, current_gen):
        parents = self.elite_tournament_selection(current_gen)

        offspring = []
        
        # 生成指定数量的后代
        for _ in range(self.pop_size // 2):  
            # 随机选择两个父母进行交叉
            parent1, parent2 = random.sample(parents, 2)
            
            # 交叉操作
            child1, child2 = self.cross(parent1, parent2)
            
            # 变异操作
            current_mutation_rate = self.get_dynamic_mutation_rate(self.generations)
            if random.random() < current_mutation_rate:
                child1 = self.mutate(child1)
            if random.random() < current_mutation_rate:
                child2 = self.mutate(child2)
                
            # 添加到后代种群 
            offspring.extend([child1, child2])
        
        combined_population = current_gen + offspring
        
        # 适应度排序
        sorted_indices = self.rank_routes(combined_population)
        
        # 选择最优的个体
        next_gen = [combined_population[i] for i in sorted_indices[:self.pop_size]]
        
        return next_gen
    
    # 运行遗传算法
    def run(self):
        # 初始化种群
        self.initial_population()
        start_time = time.time()

        # 迭代进化
        for generation in range(self.generations):
            self.population = self.next_generation(self.population)
            
            # 记录当前最优路径和距离
            best_route = self.population[0]

            # 对最优个体进行局部搜索
            if generation % 10 == 0 :
                best_route = self.local_search(best_route)

            # 将优化后的路径替换回种群中的最佳个体
            sorted_indices = self.rank_routes(self.population)
            best_idx = sorted_indices[0]
            self.population[best_idx] = best_route.copy()
            
            best_distance = self.route_distance(best_route)

            if len(self.best_distances) != 0 and self.best_distances[-1] >= best_distance:
                self.no_improvement_count += 1

            self.best_routes.append(best_route)
            self.best_distances.append(best_distance)
            
            print(f"Generation {generation}: Best Distance = {best_distance}")
        
        end_time = time.time()
        print(f"运行时间: {end_time - start_time:.2f}秒")
        # 返回最优路径和距离
        return self.best_routes[-1], self.best_distances[-1]
    
if __name__ == "__main__":
    tsp_solver = GeneticAlgTSP(file=r"c:\Users\Leslie\Desktop\学习资料\人工智能\人工智能实验\03_24_lab\task2\dj38.tsp")
    best_route, best_distance = tsp_solver.run()
    
    print(f"Best Route: {best_route}")
    print(f"Best Distance: {best_distance:.2f}")