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


def manhattan(board):
    result = 0
    for index, value in enumerate(board):
        if value != 0 and value != index + 1:
            # 计算当前坐标
            i, j = divmod(index, 4)
            # 计算目标位置
            goal_i, goal_j = divmod(value - 1, 4)
            # 计算曼哈顿距离
            result += abs(i - goal_i) + abs(j - goal_j)
    return result


def linear_conflicts(board):
    conflicts = 0

    # 计算行冲突
    for row in range(4):
        # 获取当前行中的所有数字
        row_tiles = []
        for col in range(4):
            val = board[row * 4 + col]
            if val != 0:  # 跳过空格
                goal_row = (val - 1) // 4
                goal_col = (val - 1) % 4
                if goal_row == row:  # 这个数字的目标行是当前行
                    row_tiles.append((val, col, goal_col))

        # 检查当前行中的冲突
        for i in range(len(row_tiles)):
            for j in range(i + 1, len(row_tiles)):
                _, col1, goal_col1 = row_tiles[i]
                _, col2, goal_col2 = row_tiles[j]
                if (col1 < col2 and goal_col1 > goal_col2) or (
                    col1 > col2 and goal_col1 < goal_col2
                ):
                    conflicts += 1

    # 计算列冲突
    for col in range(4):
        # 获取当前列中的所有数字
        col_tiles = []
        for row in range(4):
            val = board[row * 4 + col]
            if val != 0:  # 跳过空格
                goal_row = (val - 1) // 4
                goal_col = (val - 1) % 4
                if goal_col == col:  # 这个数字的目标列是当前列
                    col_tiles.append((val, row, goal_row))

        # 检查当前列中的冲突
        for i in range(len(col_tiles)):
            for j in range(i + 1, len(col_tiles)):
                _, row1, goal_row1 = col_tiles[i]
                _, row2, goal_row2 = col_tiles[j]
                if (row1 < row2 and goal_row1 > goal_row2) or (
                    row1 > row2 and goal_row1 < goal_row2
                ):
                    conflicts += 1

    return conflicts * 2  # 每个冲突需要至少2步额外移动


def generate_next_states(board):
    global MOVE_TABLE
    results = []
    idxz = board.index(0)  # 空格位置
    l = list(board)  # 转换为列表以便修改
    for m in MOVE_TABLE[idxz]:
        l[idxz], l[idxz + m] = l[idxz + m], l[idxz]  # 交换空格和目标位置
        results.append(tuple(l))  # 记录移动的数字
        l[idxz + m], l[idxz] = l[idxz], l[idxz + m]  # 还原交换
    return results


def record_path(parent, init_board):
    path = []
    current = goal_tuple
    while current != init_board:
        prev = parent[current]
        # 找出移动的数字
        moved_tile = next(
            current[i] for i in range(16) if current[i] != prev[i] and prev[i] == 0
        )
        path.append(moved_tile)
        current = prev
    path.reverse()
    return path


def A_star(init_board):
    global goal_tuple
    # 统计信息
    start_time = time.time()
    nodes_expanded = 0

    # 采用堆进行排序
    myheap = []
    visited = set()
    parent = {}
    parent[init_board] = None
    moves = {}
    moves[init_board] = 0

    heapq.heappush(myheap, (0, init_board))

    while myheap:
        _, now_board = heapq.heappop(myheap)

        # check_end_state
        if now_board == goal_tuple:
            end_time = time.time()
            path = record_path(parent, init_board)
            result = {
                "path": path,
                "path_len": len(path),
                "expanded_nodes": nodes_expanded,
                "time": end_time - start_time,
            }
            return result

        visited.add(now_board)
        nodes_expanded += 1

        if len(visited) % 10000 == 0:
            gc.collect()
        # 生成下一层
        for next_board in generate_next_states(now_board):
            new_g = moves[now_board] + 1 
            if next_board not in moves or new_g < moves[next_board]:
                moves[next_board] = new_g
                new_h = manhattan(next_board) + linear_conflicts(next_board)
                new_f = new_g + new_h

                heapq.heappush(myheap, (new_f, next_board))
                parent[next_board] = now_board  # 记录父节点             
    return None


def IDA_star(initial_board):
    global goal_tuple
    # 统计信息
    start_time = time.time()
    nodes_expanded = 0

    path = []
    visited = set()
    visited.add(initial_board)
    path = [initial_board]
    parent = {}
    parent[initial_board] = None

    bound = manhattan(initial_board) + linear_conflicts(initial_board)
    
    def DFS(moves,bound):
        nonlocal nodes_expanded
        now_board = path[-1]

        # 计算启发值
        now_f = moves + manhattan(now_board) + linear_conflicts(now_board)
        if now_f > bound:
            return (False,now_f)
        if now_board == goal_tuple:
            return (True,0)
        
        min_value = float("inf")
        for next_board in generate_next_states(now_board):
            if next_board in visited:
                continue
            nodes_expanded += 1
            visited.add(next_board)
            path.append(next_board)
            parent[next_board] = now_board  # 记录父节点
            result = DFS(moves + 1, bound)
            if result[0]:
                return (True,0)
            min_value = min(min_value, result[1]) # 一轮结束了，记录这一轮最小的bound
            path.pop()
            parent.pop(next_board)  # 删除父节点记录
            visited.remove(next_board)
        return (False,min_value)

    while True:
        state, new_bound = DFS(0, bound)
        if state:
            end_time = time.time()
            result = {
                "path": record_path(parent, initial_board),
                "path_len": len(path) - 1,
                "expanded_nodes": nodes_expanded,
                "time": end_time - start_time,
            }
            return result
        bound = new_bound
        if bound == float("inf"):
            break
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

    for i in range(6, 7):
        init_board = eval(f"init_board_{i}")
        print(f"\n初始状态 {i}:")
        print_board(init_board)

        
        # 使用A*算法求解
        print("\n正在使用A*算法求解...")
        result = A_star(init_board)
        print_all("A*", result)

        # 使用IDA*算法求解
        print("\n正在使用IDA*算法求解...")
        result = IDA_star(init_board)
        print_all("IDA*", result)
