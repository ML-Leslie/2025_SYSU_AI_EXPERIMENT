import heapq
import time
import numpy as np
from collections import deque
from numba import cuda, jit


class PuzzleState:
    """表示15-puzzle的状态"""

    def __init__(self, board, moves=0, previous=None, action=None):
        if isinstance(board, list):
            self.board = board
            # 转换为numpy数组用于GPU计算
            self.np_board = np.array(board, dtype=np.int32)
        else:
            self.np_board = board
            self.board = board.tolist()
        self.moves = moves  # 已执行的移动次数
        self.previous = previous  # 前一个状态
        self.action = action  # 到达此状态的动作
        self.size = len(self.board)

        # 计算空格的位置
        blank_pos = np.where(self.np_board == 0)
        self.blank_pos = (blank_pos[0][0], blank_pos[1][0])

    def get_neighbors(self):
        """获取所有可能的下一个状态"""
        neighbors = []
        i, j = self.blank_pos
        directions = [("上", -1, 0), ("下", 1, 0), ("左", 0, -1), ("右", 0, 1)]

        for direction, di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.size and 0 <= nj < self.size:
                new_board = [row[:] for row in self.board]
                new_board[i][j], new_board[ni][nj] = new_board[ni][nj], new_board[i][j]
                neighbors.append(
                    PuzzleState(new_board, self.moves + 1, self, direction)
                )

        return neighbors

    def manhattan_distance(self):
        """计算曼哈顿距离启发式 (GPU加速版本)"""
        return calculate_manhattan_distance(self.np_board)

    def linear_conflicts(self):
        """计算线性冲突 (GPU加速版本)"""
        return calculate_linear_conflicts(self.np_board)

    def heuristic(self):
        """结合曼哈顿距离和线性冲突的启发函数"""
        return self.manhattan_distance() + self.linear_conflicts()

    def is_goal(self):
        """检查是否达到目标状态"""
        for i in range(self.size):
            for j in range(self.size):
                expected = i * self.size + j + 1
                if i == self.size - 1 and j == self.size - 1:
                    expected = 0  # 右下角是0（空格）
                if self.board[i][j] != expected:
                    return False
        return True

    def get_path(self):
        """获取从初始状态到当前状态的路径"""
        path = []
        current = self
        while current.previous:
            path.append(current.action)
            current = current.previous
        return path[::-1]  # 反转路径

    def __lt__(self, other):
        """比较函数，用于优先队列排序"""
        return (self.moves + self.heuristic()) < (other.moves + other.heuristic())

    def __eq__(self, other):
        """相等性比较"""
        return self.board == other.board

    def __hash__(self):
        """哈希函数，用于在集合中去重"""
        return hash(tuple(tuple(row) for row in self.board))

    def __str__(self):
        """字符串表示"""
        result = ""
        for row in self.board:
            result += " ".join(f"{num:2d}" for num in row) + "\n"
        return result


# 用CUDA加速的曼哈顿距离计算
@cuda.jit(device=True)
def manhattan_distance_kernel(board, size):
    distance = 0
    for i in range(size):
        for j in range(size):
            value = board[i, j]
            if value != 0:  # 忽略空格
                goal_i = (value - 1) // size
                goal_j = (value - 1) % size
                distance += abs(i - goal_i) + abs(j - goal_j)
    return distance


@cuda.jit
def manhattan_distance_cuda(board, result, size):
    result[0] = manhattan_distance_kernel(board, size)


def calculate_manhattan_distance(board):
    """计算曼哈顿距离的CUDA加速包装函数"""
    # 配置CUDA
    d_board = cuda.to_device(board)
    result = np.zeros(1, dtype=np.int32)
    d_result = cuda.to_device(result)

    # 启动内核
    manhattan_distance_cuda[1, 1](d_board, d_result, board.shape[0])

    # 获取结果
    return d_result.copy_to_host()[0]


# 用CUDA加速的线性冲突计算
@cuda.jit(device=True)
def count_conflicts(tiles, tiles_count):
    """计算一行或一列中的冲突数"""
    conflicts = 0
    for i in range(tiles_count):
        for j in range(i + 1, tiles_count):
            # 如果目标位置的相对顺序与当前相反，则存在冲突
            if tiles[i, 1] > tiles[j, 1]:
                conflicts += 1
    return conflicts


@cuda.jit(device=True)
def linear_conflicts_kernel(board, size):
    """计算线性冲突"""
    conflicts = 0

    # 临时存储一行或一列的片段(最多size个元素)
    row_tiles = cuda.local.array(shape=(16, 2), dtype=np.int32)
    col_tiles = cuda.local.array(shape=(16, 2), dtype=np.int32)

    # 检查行冲突
    for i in range(size):
        row_count = 0
        for j in range(size):
            value = board[i, j]
            if value != 0:
                goal_i = (value - 1) // size
                if goal_i == i:  # 如果这个数字的目标行就是当前行
                    row_tiles[row_count, 0] = value
                    row_tiles[row_count, 1] = (value - 1) % size  # 目标列
                    row_count += 1

        # 计算当前行内的冲突
        conflicts += count_conflicts(row_tiles, row_count)

    # 检查列冲突
    for j in range(size):
        col_count = 0
        for i in range(size):
            value = board[i, j]
            if value != 0:
                goal_j = (value - 1) % size
                if goal_j == j:  # 如果这个数字的目标列就是当前列
                    col_tiles[col_count, 0] = value
                    col_tiles[col_count, 1] = (value - 1) // size  # 目标行
                    col_count += 1

        # 计算当前列内的冲突
        conflicts += count_conflicts(col_tiles, col_count)

    return 2 * conflicts  # 每个冲突增加2步移动


@cuda.jit
def linear_conflicts_cuda(board, result, size):
    """CUDA内核函数计算线性冲突"""
    result[0] = linear_conflicts_kernel(board, size)


def calculate_linear_conflicts(board):
    """计算线性冲突的CUDA加速包装函数"""
    # 配置CUDA
    d_board = cuda.to_device(board)
    result = np.zeros(1, dtype=np.int32)
    d_result = cuda.to_device(result)

    # 启动内核
    linear_conflicts_cuda[1, 1](d_board, d_result, board.shape[0])

    # 获取结果
    return d_result.copy_to_host()[0]


def is_solvable(board, size=4):
    """检查15-puzzle是否有解"""
    # 将二维数组转为一维
    flat_board = [num for row in board for num in row]

    # 计算逆序数
    inversions = 0
    for i in range(len(flat_board)):
        if flat_board[i] == 0:
            continue
        for j in range(i + 1, len(flat_board)):
            if flat_board[j] != 0 and flat_board[i] > flat_board[j]:
                inversions += 1

    # 找到空格所在的行（从底部数）
    blank_row = 0
    for i in range(size):
        for j in range(size):
            if board[i][j] == 0:
                blank_row = size - i
                break

    # 检查是否有解
    if size % 2 == 0:  # 偶数大小的拼图
        return (inversions + blank_row) % 2 == 1
    else:  # 奇数大小的拼图
        return inversions % 2 == 0


def a_star_search(initial_state):
    """使用A*算法解决15-puzzle"""
    open_set = []  # 优先队列
    closed_set = set()  # 已访问状态集合

    # 将初始状态加入优先队列
    heapq.heappush(open_set, initial_state)

    # 统计信息
    nodes_expanded = 0
    max_queue_size = 1

    start_time = time.time()

    while open_set:
        # 获取评估函数值最小的状态
        current = heapq.heappop(open_set)

        # 检查是否达到目标
        if current.is_goal():
            end_time = time.time()
            path = current.get_path()
            return {
                "path": path,
                "path_length": len(path),
                "nodes_expanded": nodes_expanded,
                "max_queue_size": max_queue_size,
                "time_taken": end_time - start_time,
            }

        # 将当前状态加入已访问集合
        closed_set.add(current)
        nodes_expanded += 1

        # 生成所有可能的下一个状态
        for neighbor in current.get_neighbors():
            if neighbor in closed_set:
                continue

            # 将新状态加入优先队列
            heapq.heappush(open_set, neighbor)

        # 更新最大队列大小
        max_queue_size = max(max_queue_size, len(open_set))

    return None  # 无解


def ida_star_search(initial_state, max_iterations=1000000):
    """使用IDA*算法解决15-puzzle"""
    # 统计信息
    nodes_expanded = 0
    max_queue_size = 1
    iteration_count = 0

    start_time = time.time()

    # 初始界限设为初始状态的启发式估计
    bound = initial_state.heuristic()

    while iteration_count < max_iterations:
        # 记录下一次迭代的界限
        next_bound = float("inf")
        # 使用栈进行深度优先搜索
        stack = deque([(initial_state, 0)])
        # 已访问状态集合（仅在当前路径上）
        visited = {initial_state}

        while stack:
            current, depth = stack.pop()

            if depth == 0:  # 重置访问集合（仅在回溯到根节点时）
                visited = {current}

            # 更新最大队列大小
            max_queue_size = max(max_queue_size, len(stack) + 1)

            # 估计总代价
            f = current.moves + current.heuristic()

            # 如果超过当前界限，更新下一个界限并跳过
            if f > bound:
                next_bound = min(next_bound, f)
                continue

            # 检查是否达到目标
            if current.is_goal():
                end_time = time.time()
                path = current.get_path()
                return {
                    "path": path,
                    "path_length": len(path),
                    "nodes_expanded": nodes_expanded,
                    "max_queue_size": max_queue_size,
                    "time_taken": end_time - start_time,
                }

            nodes_expanded += 1

            # 生成所有可能的下一个状态
            neighbors = list(current.get_neighbors())
            # 按启发式值排序，以便先尝试更有希望的状态
            neighbors.sort(key=lambda x: x.heuristic())

            for neighbor in neighbors:
                if neighbor not in visited:
                    stack.append((neighbor, depth + 1))
                    visited.add(neighbor)

        # 如果下一个界限是无穷大，说明无解
        if next_bound == float("inf"):
            return None

        # 更新界限进行下一次迭代
        bound = next_bound
        iteration_count += 1

        # 每100次迭代打印一次进度
        if iteration_count % 100 == 0:
            print(f"IDA*: 已完成 {iteration_count} 次迭代，当前界限 {bound}")

    # 如果达到最大迭代次数
    print(f"达到最大迭代次数 {max_iterations}，停止搜索")
    return None


def print_solution(result, algorithm_name):
    """打印解决方案"""
    if result:
        print(f"\n{algorithm_name} 算法找到解决方案:")
        print(f"步骤数: {result['path_length']}")
        print(f"展开节点数: {result['nodes_expanded']}")
        print(f"最大队列大小: {result['max_queue_size']}")
        print(f"用时: {result['time_taken']:.4f} 秒")
        print(f"解决路径: {' -> '.join(result['path'])}")
    else:
        print(f"\n{algorithm_name} 算法无法找到解决方案。")


def main():
    try:
        # 初始化CUDA设备
        cuda.select_device(0)
        print("CUDA设备初始化成功，使用GPU加速")
    except:
        print(
            "CUDA设备初始化失败，请确保您的计算机有支持CUDA的GPU并已安装正确的驱动程序"
        )
        return

    # 示例:
    initial_board = [[11, 3, 1, 7], [4, 6, 8, 2], [15, 9, 10, 13], [14, 12, 5, 0]]

    # 检查问题是否有解
    if not is_solvable(initial_board):
        print("给定的15-puzzle无解!")
        return

    initial_state = PuzzleState(initial_board)
    print("初始状态:")
    print(initial_state)

    print("计算启发式值中...")
    print(f"曼哈顿距离: {initial_state.manhattan_distance()}")
    print(f"线性冲突: {initial_state.linear_conflicts()}")
    print(f"总启发式值: {initial_state.heuristic()}")

    # 使用A*算法
    print("\n正在使用A*算法求解...")
    a_star_result = a_star_search(initial_state)
    print_solution(a_star_result, "A*")

    # 使用IDA*算法
    print("\n正在使用IDA*算法求解...")
    ida_star_result = ida_star_search(initial_state)
    print_solution(ida_star_result, "IDA*")


if __name__ == "__main__":
    main()
