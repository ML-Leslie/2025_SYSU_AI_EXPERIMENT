import re

tuple_3 = ("On(tony,mike)",)
tuple_2 = ("On(mike,john)",)
tuple_1 = ("Green(tony)",)
tuple_4 = ("~Green(john)",)
tuple_5 = ("~On(x,y)","~Green(x)","Green(y)",)



myset = {
    tuple_1: 1,
    tuple_2: 2,
    tuple_3: 3,
    tuple_4: 4,
    tuple_5: 5,

}



visit = set()


def print_set(myset):
    count = 1
    for i in myset:
        print(count, " ", i)
        count += 1


def extract_inside_parenthesis(s):
    start = s.find("(")
    end = s.find(")")
    if start != -1 and end != -1:
        return s[start + 1 : end]
    return None


# # 假设所有小写字母开头的是变量，大写字母开头的是常量或谓词
# def is_variable(term):
#     return term and term[0].islower() and term.isalnum()


def is_variable(term):
    """判断一个项是否为变量（假设变量是单个小写字母）"""
    return len(term) == 1 and term.islower()


def unify(i, j):
    if i == j:
        return

    for m in range(len(i)):
        for k in range(len(j)):
            # 提取谓词名称
            w1_1 = i[m].split("(")[0]  # 去掉~符号再比较谓词
            w1_2 = j[k].split("(")[0]

            # 检查谓词是否相同（忽略否定符号）
            pred1 = w1_1[1:] if w1_1.startswith("~") else w1_1
            pred2 = w1_2[1:] if w1_2.startswith("~") else w1_2

            if pred1 != pred2:
                continue  # 谓词不同，不能归结

            # 检查是否一个是肯定一个是否定
            neg1 = i[m].startswith("~")
            neg2 = j[k].startswith("~")

            if neg1 == neg2:
                continue  # 都是肯定或都是否定，不能归结

            # 提取变量和常量
            w2_1 = extract_inside_parenthesis(i[m])
            w2_2 = extract_inside_parenthesis(j[k])

            # 处理多变量情况
            vars_1 = [v.strip() for v in w2_1.split(",")] if w2_1 else []
            vars_2 = [v.strip() for v in w2_2.split(",")] if w2_2 else []

            # 检查参数数量是否相同
            if len(vars_1) != len(vars_2):
                continue

            # 检查是否有两个常量不相等
            skip_this_pair = False
            for var1, var2 in zip(vars_1, vars_2):
                # 如果两者都是常量且不相等
                if not is_variable(var1) and not is_variable(var2) and var1 != var2:
                    skip_this_pair = True
                    break
            if skip_this_pair:
                continue

            # 构建变量替换映射
            var_map = {}
            for var1, var2 in zip(vars_1, vars_2):
                if var1 != var2:
                    if is_variable(var1) and not is_variable(var2):
                        var_map[var1] = var2
                    elif is_variable(var2) and not is_variable(var1):
                        var_map[var2] = var1
                    # elif is_variable(var1) and is_variable(var2):
                    #     # 两个都是变量，选择一个进行替换
                    #     var_map[var1] = var2

            # 如果没有变量替换需要进行，但谓词相同且一正一反，则可以直接归结
            print("R", "[", myset[i], chr(ord('a') + m) ,",", myset[j], chr(ord('a') + k) ,"]", end="")
            # print(f"R[{myset[i]}.{m + 1}, {myset[j]}.{k + 1}]", end="")
            # print(f"R[{myset[i]}{chr(ord('a') + m)}, {myset[j]}{chr(ord('a') + k)}]", end="")
            for var, val in var_map.items():
                print("(", var, "=", val, ")", end="")

            # 更新访问过的元素
            visit.add(i)
            visit.add(j)

            # 创建新的子句，移除归结的字面量
            new_i = list(i[:m] + i[m + 1 :])
            new_j = list(j[:k] + j[k + 1 :])

            # 应用变量替换
            resolved_clause = []

            # 处理i中剩余的文字
            # 处理i中剩余的文字
            for item in new_i:
                new_item = item
                for var, val in var_map.items():
                    new_item = new_item.replace(f"({var}", f"({val}").replace(f",{var}", f",{val}")
                resolved_clause.append(new_item)

            # 处理j中剩余的文字
            for item in new_j:
                new_item = item
                for var, val in var_map.items():
                    new_item = new_item.replace(f"({var}", f"({val}").replace(f",{var}", f",{val}")
                resolved_clause.append(new_item)

            # 转换回元组并添加到集合
            new_tuple = tuple(resolved_clause)
            print(new_tuple)

            if new_tuple:  # 只有当新tuple非空时才添加到myset
                myset[new_tuple] = len(myset) + 1

            # 找到一个归结后返回
            return


def resolution(myset):
    """一阶逻辑归结算法主函数"""
    print("\n开始归结过程...")

    found_empty_clause = False
    iteration = 1

    while not found_empty_clause:
        print(f"\n第{iteration}轮归结:")
        iteration += 1

        # 获取所有未访问过的元组
        tuples = [t for t in myset.keys() if t not in visit]

        # 记录本轮是否有新子句生成
        new_clause_added = False

        # 遍历所有可能的元组对
        for i in range(len(tuples)):
            for j in range(i + 1, len(tuples)):
                tuple_i = tuples[i]
                tuple_j = tuples[j]

                # 如果元组已经在访问集合中，则跳过
                if tuple_i in visit or tuple_j in visit:
                    continue

                # 记录当前子句集合大小
                old_size = len(myset)

                # 尝试归结
                unify(tuple_i, tuple_j)

                # 检查是否有新子句生成
                if len(myset) > old_size:
                    new_clause_added = True

                    # 检查是否生成了空子句
                    if () in myset:
                        print("\n归结生成了空子句，证明成功！")
                        found_empty_clause = True
                        break

            if found_empty_clause:
                break

        # 如果没有新子句生成，则停止归结
        if not new_clause_added:
            print("\n归结完成，没有新的子句生成")
            break

    return found_empty_clause


def resolution_unit_preference(myset):
    """单元子句优先策略"""
    print("\n开始归结过程，优先使用单元子句...")

    found_empty_clause = False
    iteration = 1

    while not found_empty_clause:
        print(f"\n第{iteration}轮归结:")
        iteration += 1

        # 获取所有未访问过的元组
        tuples = [t for t in myset.keys() if t not in visit]

        # 按子句长度排序，单元子句优先
        tuples.sort(key=len)

        # 记录本轮是否有新子句生成
        new_clause_added = False

        # 优先处理单元子句
        unit_clauses = [t for t in tuples if len(t) == 1]

        for unit in unit_clauses:
            # 尝试与其他子句归结
            for other in tuples:
                if unit == other or other in visit or unit in visit:
                    continue

                # 记录当前子句集合大小
                old_size = len(myset)

                # 尝试归结
                unify(unit, other)

                # 检查是否有新子句生成
                if len(myset) > old_size:
                    new_clause_added = True
                    # 检查是否生成了空子句
                    if () in myset:
                        print("\n归结生成了空子句，证明成功！")
                        return True

        # 处理剩余子句对...
        # (与原resolution函数类似)

        # 如果没有新子句生成，则停止归结
        if not new_clause_added:
            print("\n归结完成，没有新的子句生成")
            break

    return found_empty_clause


def resolution_bfs(myset):
    """广度优先搜索策略的归结实现"""
    print("\n开始归结过程，使用BFS策略...")

    # 创建队列来存储待处理的子句
    from collections import deque

    queue = deque([([], list(myset.keys()))])  # (已访问子句, 待处理子句)

    # 记录已经尝试过的归结对，避免重复工作
    attempted_resolutions = set()

    # 记录每轮迭代的结果
    iteration = 1

    while queue:
        visited, remaining = queue.popleft()

        print(f"\n第{iteration}轮归结:")
        iteration += 1

        # 记录本轮是否有新子句生成
        new_clause_added = False

        # 优先处理层次较浅的子句对
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                clause_i = remaining[i]
                clause_j = remaining[j]

                # 跳过已经尝试过的归结对
                resolution_pair = frozenset([clause_i, clause_j])
                if resolution_pair in attempted_resolutions:
                    continue

                attempted_resolutions.add(resolution_pair)

                # 记录当前子句集合大小
                old_size = len(myset)

                # 尝试归结
                unify(clause_i, clause_j)

                # 检查是否有新子句生成
                if len(myset) > old_size:
                    new_clause_added = True

                    # 获取新生成的子句
                    new_clauses = [
                        c
                        for c in myset.keys()
                        if c not in visited and c not in remaining
                    ]

                    # 检查是否生成了空子句
                    if () in new_clauses:
                        print("\n归结生成了空子句，证明成功！")
                        return True

                    # 将新状态加入队列 - BFS特性
                    new_visited = visited + [clause_i, clause_j]
                    new_remaining = [
                        c for c in remaining if c != clause_i and c != clause_j
                    ] + new_clauses
                    queue.append((new_visited, new_remaining))

        # 如果没有新子句生成，则继续处理下一个状态
        if not new_clause_added and not queue:
            print("\n归结完成，没有新的子句生成")

    return () in myset


def resolution_dfs(myset):
    """深度优先搜索策略的归结实现"""
    print("\n开始归结过程，使用DFS策略...")

    # 创建栈来存储待处理的状态
    stack = [([], list(myset.keys()))]  # (已访问子句, 待处理子句)

    # 记录已经尝试过的归结对，避免重复工作
    attempted_resolutions = set()

    # 记录每轮迭代的结果
    iteration = 1
    max_depth = 20  # 设置最大深度，防止无限递归

    while stack and iteration <= max_depth:
        visited, remaining = stack.pop()  # DFS使用栈，从末尾取元素

        print(f"\n第{iteration}轮归结 (深度={len(visited)//2}):")
        iteration += 1

        # 没有子句可处理，继续下一个状态
        if not remaining:
            continue

        # 选择第一个子句与其他子句归结 - DFS特性
        clause_i = remaining[0]
        remaining_without_i = remaining[1:]

        resolution_found = False

        for j, clause_j in enumerate(remaining_without_i):
            # 跳过已经尝试过的归结对
            resolution_pair = frozenset([clause_i, clause_j])
            if resolution_pair in attempted_resolutions:
                continue

            attempted_resolutions.add(resolution_pair)

            # 记录当前子句集合大小
            old_size = len(myset)

            # 尝试归结
            unify(clause_i, clause_j)

            # 检查是否有新子句生成
            if len(myset) > old_size:
                resolution_found = True

                # 获取新生成的子句
                new_clauses = [
                    c for c in myset.keys() if c not in visited and c not in remaining
                ]

                # 检查是否生成了空子句
                if () in new_clauses:
                    print("\n归结生成了空子句，证明成功！")
                    return True

                # 将新状态优先加入栈 - DFS特性
                new_visited = visited + [clause_i, clause_j]
                new_remaining = (
                    new_clauses + remaining_without_i[:j] + remaining_without_i[j + 1 :]
                )
                stack.append((new_visited, new_remaining))

                # DFS特性：找到一个归结就立即深入
                break

        # 如果当前子句与任何其他子句都无法归结，则尝试下一个子句
        if not resolution_found:
            stack.append((visited, remaining_without_i))

    if iteration > max_depth:
        print(f"\n达到最大深度{max_depth}，停止归结")

    return () in myset


if __name__ == "__main__":
    print_set(myset)
    print("\n开始归结演示:")
    resolution(myset)

    print("\n归结后的结果集合:")
    print_set(myset)
