import re

tuple_3 = ("On(tony,mike)",)
tuple_2 = ("On(mike,john)",)
tuple_1 = ("Green(tony)",)
tuple_4 = ("~Green(john)",)
tuple_5 = (
    "~On(x,y)",
    "~Green(x)",
    "Green(y)",
)

myset = {
    tuple_1: 1,
    tuple_2: 2,
    tuple_3: 3,
    tuple_4: 4,
    tuple_5: 5,
}


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


def is_variable(term):
    """判断一个项是否为变量（假设变量是单个小写字母）"""
    return len(term) == 1 and term.islower()


def can_resolve(literal_1, literal_2):
    """检查两个字面量是否可以归结"""
    # 提取谓词名称
    w1_1 = literal_1.split("(")[0]
    w1_2 = literal_2.split("(")[0]

    # 检查谓词是否相同（忽略否定符号）
    pred1 = w1_1[1:] if w1_1.startswith("~") else w1_1
    pred2 = w1_2[1:] if w1_2.startswith("~") else w1_2

    if pred1 != pred2:
        return False  # 谓词不同，不能归结

    # 检查是否一个是肯定一个是否定
    neg1 = literal_1.startswith("~")
    neg2 = literal_2.startswith("~")

    if neg1 == neg2:
        return False  # 都是肯定或都是否定，不能归结

    # 提取变量和常量
    w2_1 = extract_inside_parenthesis(literal_1)
    w2_2 = extract_inside_parenthesis(literal_2)

    # 处理多变量情况
    vars_1 = [v.strip() for v in w2_1.split(",")] if w2_1 else []
    vars_2 = [v.strip() for v in w2_2.split(",")] if w2_2 else []

    # 检查参数数量是否相同
    if len(vars_1) != len(vars_2):
        return False

    # 检查是否有两个常量不相等
    for var1, var2 in zip(vars_1, vars_2):
        # 如果两者都是常量且不相等
        if not is_variable(var1) and not is_variable(var2) and var1 != var2:
            return False

    # 可以归结
    return True


def perform_resolution(myset, clause_i, clause_j, pos_i, pos_j):
    """执行归结操作"""
    m = pos_i
    k = pos_j

    # 获取要归结的字面量
    literal_i = clause_i[m]
    literal_j = clause_j[k]

    # 提取变量和常量
    w2_1 = extract_inside_parenthesis(literal_i)
    w2_2 = extract_inside_parenthesis(literal_j)

    # 处理多变量情况
    vars_1 = [v.strip() for v in w2_1.split(",")] if w2_1 else []
    vars_2 = [v.strip() for v in w2_2.split(",")] if w2_2 else []

    # 构建变量替换映射
    var_map = {}
    for var1, var2 in zip(vars_1, vars_2):
        if var1 != var2:
            if is_variable(var1) and not is_variable(var2):
                var_map[var1] = var2
            elif is_variable(var2) and not is_variable(var1):
                var_map[var2] = var1

    # 打印归结信息
    print(
        f"R[{myset[clause_i]}{chr(ord('a') + m)}, {myset[clause_j]}{chr(ord('a') + k)}]",
        end="",
    )
    for var, val in var_map.items():
        print(f"({var}={val})", end="")

    # 创建新的子句，移除归结的字面量
    new_i = list(clause_i[:m] + clause_i[m + 1 :])
    new_j = list(clause_j[:k] + clause_j[k + 1 :])

    # 应用变量替换
    resolved_clause = []

    # 处理i中剩余的文字
    for item in new_i:
        new_item = item
        for var, val in var_map.items():
            new_item = new_item.replace(f"({var}", f"({val}").replace(
                f",{var}", f",{val}"
            )
        resolved_clause.append(new_item)

    # 处理j中剩余的文字
    for item in new_j:
        new_item = item
        for var, val in var_map.items():
            new_item = new_item.replace(f"({var}", f"({val}").replace(
                f",{var}", f",{val}"
            )
        resolved_clause.append(new_item)

    # 转换回元组并添加到集合
    new_tuple = tuple(resolved_clause)
    print(new_tuple)

    if new_tuple:  # 只有当新tuple非空时才添加到myset
        myset[new_tuple] = len(myset) + 1

    return new_tuple


def resolution(myset):
    """修改后的一阶逻辑归结算法主函数 - 确保每个元素只使用一次"""
    print("\n开始归结过程...")

    found_empty_clause = False
    iteration = 1
    MAX_ITERATIONS = 20  # 设置最大迭代次数

    # 用于记录已尝试过的归结对
    attempted_pairs = set()

    while not found_empty_clause and iteration <= MAX_ITERATIONS:
        print(f"\n第{iteration}轮归结:")
        iteration += 1

        # 获取所有元组
        tuples = list(myset.keys())

        # 记录本轮是否有新子句生成
        new_clause_added = False

        # 记录本轮迭代中每个子句中已使用的元素位置
        used_elements = {clause: set() for clause in tuples}

        # 遍历所有可能的元组对
        for i in range(len(tuples)):
            for j in range(i + 1, len(tuples)):
                tuple_i = tuples[i]
                tuple_j = tuples[j]

                # 避免重复尝试相同的归结对
                pair_key = (
                    (tuple_i, tuple_j)
                    if hash(tuple_i) < hash(tuple_j)
                    else (tuple_j, tuple_i)
                )
                if pair_key in attempted_pairs:
                    continue

                attempted_pairs.add(pair_key)

                # 记录当前子句集合大小
                old_size = len(myset)

                # 尝试对未使用的元素进行归结
                for m in range(len(tuple_i)):
                    # 如果元素已被使用，则跳过
                    if m in used_elements[tuple_i]:
                        continue

                    for k in range(len(tuple_j)):
                        # 如果元素已被使用，则跳过
                        if k in used_elements[tuple_j]:
                            continue

                        # 检查这两个位置的元素是否可以归结
                        if can_resolve(tuple_i[m], tuple_j[k]):
                            # 标记这些位置的元素为已使用
                            used_elements[tuple_i].add(m)
                            used_elements[tuple_j].add(k)

                            # 执行归结
                            perform_resolution(myset, tuple_i, tuple_j, m, k)

                # 检查是否有新子句生成
                if len(myset) > old_size:
                    new_clause_added = True

                    # 检查是否生成了空子句
                    if () in myset:
                        print("\n发现空子句，成功归结！")
                        found_empty_clause = True
                        break
            if found_empty_clause:
                break

        # 如果没有新子句生成，则停止归结
        if not new_clause_added:
            print("\n归结完成，没有新的子句生成")
            break

    return found_empty_clause


if __name__ == "__main__":
    print_set(myset)
    print("\n开始归结演示:")
    resolution(myset)

    print("\n归结后的结果集合:")
    print_set(myset)
