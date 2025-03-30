import heapq
import time

# 使用tuple来表示board


class Puzzle_State:
    def __init__(self, board, moves=0, previous=None, action=None):
        self.board = board
        self.moves = moves
        self.previous = previous
        self.action = action
        self.size = 4
        self.blank_pos = None
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    self.blank_pos = (i, j)
                    break

    def manhattan(self):
        result = 0
        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i][j]
                if value != 0:
                    # 计算目标位置
                    goal_i = (value - 1) // 4
                    goal_j = (value - 1) % 4
                    # 计算曼哈顿距离
                    result += abs(i - goal_i) + abs(j - goal_j)
        return result

    def is_end_state(self):
        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i][j]
                if value != 0:
                    # 计算目标位置
                    goal_i = (value - 1) // 4
                    goal_j = (value - 1) % 4
                    if (i, j) != (goal_i, goal_j):
                        return False
        return True

    def generate_next_states(self):
        results = []
        i, j = self.blank_pos
        dires = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ]  # 上下左右
        for mi, mj in dires:
            ni, nj = i + mi, j + mj
            if 0 <= ni < 4 and 0 <= nj < 4:
                value = self.board[ni][nj]
                new_board = [list(row) for row in self.board]
                new_board[i][j], new_board[ni][nj] = new_board[ni][nj], new_board[i][j]
                new_board = [tuple(row) for row in new_board]
                results.append(Puzzle_State(new_board, self.moves + 1, self, value))
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

    def h(self):
        return self.manhattan() + self.moves

    def __lt__(self, other):
        return self.h() < other.h()


def A_star(init_board):
    # 统计信息
    start_time = time.time()
    expenfd_nodes = 0
    depth = 0
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

        # 生成下一层
        for next_board in now_board.generate_next_states():
            if tuple(next_board.board) not in visited:
                heapq.heappush(myheap, next_board)

    return None


def print_board(board):
    for row in board:
        print(" ".join(f"{num:2d}" for num in row))


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
        (1, 2, 4, 8),
        (5, 7, 11, 10),
        (13, 15, 0, 3),
        (14, 6, 9, 12),
    )

    init_board_2 = (
        (14, 10, 6, 0),
        (4, 9, 1, 8),
        (2, 3, 5, 11),
        (12, 13, 7, 15),
    )

    # init_board_3 = (
    #     (5, 1, 3, 4),
    #     (2, 7, 8, 12),
    #     (9, 6, 11, 15),
    #     (0, 13, 10, 14),
    # )

    # init_board_4 = (
    #     (6, 10, 3, 15),
    #     (14, 8, 7, 11),
    #     (5, 1, 0, 2),
    #     (13, 12, 9, 4),
    # )

    # init_board_5 = (
    #     (11, 3, 1, 7),
    #     (4, 6, 8, 2),
    #     (15, 9, 10, 13),
    #     (14, 12, 5, 0),
    # )

    # init_board_6 = (
    #     (0, 5, 15, 14),
    #     (7, 9, 6, 13),
    #     (1, 2, 12, 10),
    #     (8, 11, 4, 3),
    # )

    for i in range(1, 3):
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
