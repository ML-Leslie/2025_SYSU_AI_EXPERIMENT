import heapq
import time
import gc


MOVE_TABLE = [[] for _ in range(16)]
for i in range(16):
    x, y = i % 4, i // 4
    if x > 0:
        MOVE_TABLE[i].append(-1)  # 左
    if x < 3:
        MOVE_TABLE[i].append(1)  # 右
    if y > 0:
        MOVE_TABLE[i].append(-4)  # 上
    if y < 3:
        MOVE_TABLE[i].append(4)  # 下

goal_tuple = tuple(range(1, 16)) + (0,)


class Puzzle_State:
    def __init__(self, board, moves=0, previous=None, action=None):
        self.board = board  # 一维tuple
        self.moves = moves
        self.previous = previous
        self.action = action
        self.size = 4
        self.blank_pos = self.board.index(0)  # 空格位置

    def manhattan(self):
        result = 0
        for index, value in enumerate(self.board):
            if value != 0 and value != index + 1:
                # 计算当前坐标
                i, j = divmod(index, self.size)
                # 计算目标位置
                goal_i, goal_j = divmod(value - 1, self.size)
                # 计算曼哈顿距离
                result += abs(i - goal_i) + abs(j - goal_j)
        return result

    def is_end_state(self):
        return self.board == goal_tuple

    def generate_next_states(self):
        global MOVE_TABLE
        results = []
        idxz = self.blank_pos  # 空格位置
        l = list(self.board)  # 转换为列表以便修改
        for m in MOVE_TABLE[idxz]:
            l[idxz], l[idxz + m] = l[idxz + m], l[idxz]  # 交换空格和目标位置
            results.append(
                Puzzle_State(
                    tuple(l),
                    self.moves + 1,
                    self,
                    l[idxz],  # 记录移动的数字
                )
            )
            l[idxz + m], l[idxz] = l[idxz], l[idxz + m]  # 还原交换
        return results

    def record_path(self):
        path = []
        current = self
        while current:
            path.append(current.action)
            current = current.previous
        path.reverse()
        # 去掉第一个动作，因为它是None
        path = path[1:]
        return path

    def f(self):
        return self.manhattan() + self.moves

    def __lt__(self, other):
        if self.f() == other.f():
            return self.moves > other.moves
        return self.f() < other.f()


def A_star(init_board):
    # 统计信息
    start_time = time.time()
    expenfd_nodes = 0

    # 采用优先队列进行排序
    myheap = []
    visited = set()

    heapq.heappush(myheap, init_board)

    while myheap:
        now_board = heapq.heappop(myheap)

        # check_end_state
        if now_board.is_end_state():
            end_time = time.time()
            path = now_board.record_path()
            result = {
                "path": path,
                "path_len": len(path),
                "expanded_nodes": expenfd_nodes,
                "time": end_time - start_time,
            }
            return result

        visited.add(tuple(now_board.board))
        expenfd_nodes += 1

        # 垃圾回收
        # gc.collect()

        # 生成下一层
        for next_board in now_board.generate_next_states():
            if tuple(next_board.board) in visited:
                continue
            heapq.heappush(myheap, next_board)

    return None


def print_board(board):
    size = int(len(board) ** 0.5)
    for i in range(size):
        print(" ".join(f"{board[i * size + j]:2d}" for j in range(size)))


def print_all(algo, result):
    if result:
        print(f"{algo}算法求解成功！")
        print(f"路径长度: {result['path_len']}")
        print(f"展开节点数: {result['expanded_nodes']}")
        print(f"运行时间: {result['time']}秒")
        print(f"路径：", result["path"])


if __name__ == "__main__":
    # 初始化board
    init_board_1 = (
        1,
        2,
        4,
        8,
        5,
        7,
        11,
        10,
        13,
        15,
        0,
        3,
        14,
        6,
        9,
        12,
    )
    # 64s
    init_board_2 = (
        14,
        10,
        6,
        0,
        4,
        9,
        1,
        8,
        2,
        3,
        5,
        11,
        12,
        13,
        7,
        15,
    )
    # kuai
    init_board_3 = (
        5,
        1,
        3,
        4,
        2,
        7,
        8,
        12,
        9,
        6,
        11,
        15,
        0,
        13,
        10,
        14,
    )
    # 400s
    init_board_4 = (
        6,
        10,
        3,
        15,
        14,
        8,
        7,
        11,
        5,
        1,
        0,
        2,
        13,
        12,
        9,
        4,
    )
    # 解不出来
    init_board_5 = (
        11,
        3,
        1,
        7,
        4,
        6,
        8,
        2,
        15,
        9,
        10,
        13,
        14,
        12,
        5,
        0,
    )

    init_board_6 = (
        0,
        5,
        15,
        14,
        7,
        9,
        6,
        13,
        1,
        2,
        12,
        10,
        8,
        11,
        4,
        3,
    )

    for i in range(2, 3):
        init_board = eval(f"init_board_{i}")
        print(f"\n初始状态 {i}:")
        print_board(init_board)

        # # 检查问题是否有解
        # if not Puzzle_State(init_board).is_end_state():
        #     print("给定的15-puzzle无解!")
        #     continue

        init_state = Puzzle_State(init_board)

        # 使用A*算法求解
        print("\n正在使用A*算法求解...")
        result = A_star(init_state)
        print_all("A*", result)

    # # 使用A*算法求解
    # print("\n正在使用A*算法求解...")
    # result = A_star(init_state)
    # print_all("A*", result)
