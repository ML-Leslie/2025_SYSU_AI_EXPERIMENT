from queue import PriorityQueue
import time

# 最终状态是全局变量
goal = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 0))


def list_to_tuple(lst):
    return tuple(tuple(row) for row in lst)


def tuple_to_list(tpl):
    return [list(row) for row in tpl]


def A_star(org_puzzle):
    # 开始计时
    stary_time = time.time()

    # 初始化计数器
    nodes_expanded = 0
    nodes_generated = 0

    org_puzzle = list_to_tuple(org_puzzle)
    # print(org_puzzle)
    frontier = PriorityQueue()
    frontier.put((0, org_puzzle))  # 优先级队列，按照优先级排序
    came_from = {}
    cost_so_far = {}
    came_from[org_puzzle] = None
    cost_so_far[org_puzzle] = 0

    while not frontier.empty():
        _, current = frontier.get()
        nodes_expanded += 1
        # print("current:", tuple_to_list(current))

        if current == goal:
            # print("have found the goal")
            break

        neigh = neighbors(current)
        for next in neigh:
            nodes_generated += 1
            new_cost = cost_so_far[current] + 1
            next = list_to_tuple(next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + enhanced_heuristic(next)  # f(n)=g(n)+h(n)
                frontier.put((priority, next))
                came_from[next] = current

    # 计算用时
    elapsed_time = time.time() - stary_time

    # 当不存在目标状态时
    if current != goal:
        # print("No solution found")
        return None
    # 回溯路径
    # print("reconstruct_path has been called")

    path = reconstruct_path(came_from, org_puzzle)

    print(f"搜索用时: {elapsed_time:.4f}秒")
    print(f"扩展节点数: {nodes_expanded}")
    print(f"生成节点数: {nodes_generated}")
    print(f"解决方案长度: {len(path)}")

    return path


# 回溯路径
def reconstruct_path(came_from, start):
    current = goal
    path = []
    while current != start:
        # 找到空格0的位置
        prev = came_from[current]
        zero_i, zero_j = None, None
        for i in range(4):
            for j in range(4):
                if current[i][j] == 0:
                    zero_i, zero_j = i, j
                    break
            if zero_i is not None:
                break

        # 找到前一个状态的空格0位置处的数字
        moved_number = prev[zero_i][zero_j]
        # print(f"current:{current}, prev:{prev}, moved_number:{moved_number}")
        path.append(moved_number)

        current = prev

    path.reverse()  # 路径是从目标状态到初始状态，需要反转
    # print(path)
    return path


# 增强的启发式函数(结合了线性冲突)
def enhanced_heuristic(state):
    """
    增强的启发式函数，结合曼哈顿距离和线性冲突
    """
    # 基础曼哈顿距离
    distance = manhattan_heuristic(state)

    # 线性冲突计算
    conflicts = 0

    # 预先计算目标位置
    goal_positions = {}
    for i in range(4):
        for j in range(4):
            goal_positions[goal[i][j]] = (i, j)

    # 检查行冲突
    for i in range(4):
        for j in range(4):
            if state[i][j] == 0:
                continue
            gi, gj = goal_positions[state[i][j]]
            if gi == i:  # 该数字在正确的行上
                for k in range(j + 1, 4):
                    if state[i][k] != 0:
                        gki, gkj = goal_positions[state[i][k]]
                        # 如果另一个数字也在正确的行上，但相对顺序错误
                        if gki == i and gkj < gj:
                            conflicts += 1

    # 检查列冲突
    for j in range(4):
        for i in range(4):
            if state[i][j] == 0:
                continue
            gi, gj = goal_positions[state[i][j]]
            if gj == j:  # 该数字在正确的列上
                for k in range(i + 1, 4):
                    if state[k][j] != 0:
                        gki, gkj = goal_positions[state[k][j]]
                        # 如果另一个数字也在正确的列上，但相对顺序错误
                        if gkj == j and gki < gi:
                            conflicts += 1

    # 每个线性冲突至少需要2次额外移动
    return distance + 2 * conflicts


# Manhattan distance的启发式函数
def manhattan_heuristic(state):
    # 计算每个方块到其目标位置的曼哈顿距离总和
    distance = 0
    # 预先计算目标位置
    goal_positions = {}
    for i in range(4):
        for j in range(4):
            goal_positions[goal[i][j]] = (i, j)

    # 计算每个数字的曼哈顿距离
    for i in range(4):
        for j in range(4):
            if state[i][j] != 0:  # 不计算空格
                gi, gj = goal_positions[state[i][j]]
                distance += abs(i - gi) + abs(j - gj)
    return distance


def neighbors(state):
    """
    生成当前拼图状态的所有相邻状态
    通过将空格(值为0)与其相邻位置交换来生成新状态
    """
    neigh = []
    # 找到空格(0)的位置
    zero_i, zero_j = None, None
    for i in range(4):
        for j in range(4):
            if state[i][j] == 0:
                zero_i, zero_j = i, j
                break
        if zero_i is not None:
            break

    # 定义可能的移动方向：下、上、右、左
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # 对每个可能的方向尝试移动
    for di, dj in directions:
        # 计算新位置
        new_i, new_j = zero_i + di, zero_j + dj

        # 检查新位置是否在棋盘范围内
        if 0 <= new_i < 4 and 0 <= new_j < 4:
            # 创建拼图的深拷贝
            new_puzzle = [list(row) for row in state]

            # 交换空格与相邻位置的数字
            new_puzzle[zero_i][zero_j], new_puzzle[new_i][new_j] = (
                new_puzzle[new_i][new_j],
                new_puzzle[zero_i][zero_j],
            )

            # 保存新的拼图状态
            neigh.append(new_puzzle)

    return neigh


def print_puzzle(puzzle):
    print("org_puzzle:", puzzle)
    path = A_star(puzzle)
    if path is None:
        print("No solution found")
    else:
        print("list of moved numbers:", path)


if __name__ == "__main__":
    print("example-1：")
    org_puzzle_1 = [[1, 2, 4, 8], [5, 7, 11, 10], [13, 15, 0, 3], [14, 6, 9, 12]]
    print_puzzle(org_puzzle_1)

    print("example-2：")
    org_puzzle_2 = [[14, 10, 6, 0], [4, 9, 1, 8], [2, 3, 5, 11], [12, 13, 7, 15]]
    print_puzzle(org_puzzle_2)

    # print("example-3：")
    # org_puzzle_3 = [
    #     [5, 1, 3, 4],
    #     [2, 7, 8, 12],
    #     [9, 6, 11, 15],
    #     [0, 13, 10, 14]
    # ]
    # print_puzzle(org_puzzle_3)

    # print("example-4：")
    # org_puzzle_4 = [
    #     [6, 10, 3, 15],
    #     [14, 8, 7, 11],
    #     [5, 1, 0, 2],
    #     [13, 12, 9, 4]
    # ]
    # print_puzzle(org_puzzle_4)

    # print("example-5：")
    # org_puzzle_5 = [
    #     [11, 3, 1, 7],
    #     [4, 6, 8, 2],
    #     [15, 9, 10, 13],
    #     [14, 12, 5, 0]
    # ]
    # print_puzzle(org_puzzle_5)

    # print("example-6：")
    # org_puzzle_6 = [
    #     [0, 5, 15, 14],
    #     [7, 9, 6, 13],
    #     [1, 2, 12, 10],
    #     [8, 11, 4, 3]
    # ]
    # print_puzzle(org_puzzle_6)
