# tuple_1 = ("B(alice)",)
# tuple_2 = ("B(bob)",)
# tuple_3 = ("B(charlie)",)
# tuple_4 = ("M(alice,coffee)",)
# tuple_5 = ("M(bob,tea)",)
# tuple_6 = ("~B(x)", "H(x)", "D(x)")
# tuple_7 = ("~H(y)", "~D(y,juice)")
# tuple_8 = ("M(z,juice)", "~S(z)")
# tuple_9 = ("~M(alice,milk)", "~M(bob,milk)")
# tuple_10 = ("M(alice,v)", "M(bob,v)")
# tuple_11 = ("~B(w)", "~D(w,water)", "S(w)")
#
# myset = {
#     tuple_1: 1,
#     tuple_2: 2,
#     tuple_3: 3,
#     tuple_4: 4,
#     tuple_5: 5,
#     tuple_6: 6,
#     tuple_7: 7,
#     tuple_8: 8,
#     tuple_9: 9,
#     tuple_10: 10,
#     tuple_11: 11,
# }






# 用例 1
# tuple_1 = ("GradStudent(sue)",)
# tuple_2 = ("~GradStudent(x)", "Student(x)")
# tuple_3 = ("~Student(x)", "HardWorker(x)")
# tuple_4 = ("~HardWorker(sue)",)
#
# myset = {tuple_1: 1, tuple_2: 2, tuple_3: 3, tuple_4: 4}
#
# myset = {
#     tuple_1: 1,
#     tuple_2: 2,
#     tuple_3: 3,
#     tuple_4: 4,
# }

# 用例2
tuple_1 = ("A(tony)",)
tuple_2 = ("A(mike)",)
tuple_3 = ("A(john)",)
tuple_4 = ("L(tony,rain)",)
tuple_5 = ("L(tony,snow)",)
tuple_6 = ("~A(x)", "S(x)", "C(x)")
tuple_7 = ("~C(y)", "~L(y,rain)")
tuple_8 = ("L(z,snow)", "~S(z)")
tuple_9 = ("~L(tony,u)", "~L(mike,u)")
tuple_10 = ("L(tony,v)", "L(mike,v)")
tuple_11 = ("~A(w)", "~C(w)", "S(w)")

myset = {
    tuple_1: 1,
    tuple_2: 2,
    tuple_3: 3,
    tuple_4: 4,
    tuple_5: 5,
    tuple_6: 6,
    tuple_7: 7,
    tuple_8: 8,
    tuple_9: 9,
    tuple_10: 10,
    tuple_11: 11,
}

# 用例 3
# tuple_1 = ("On(tony,mike)",)
# tuple_2 = ("On(mike,john)",)
# tuple_3 = ("Green(tony)",)
# tuple_4 = ("~Green(john)",)
# tuple_5 = ("~On(x,y)","~Green(x)","Green(y)",)
#
# myset = {
#     tuple_1: 1,
#     tuple_2: 2,
#     tuple_3: 3,
#     tuple_4: 4,
#     tuple_5: 5,
# }

# ===================================================================

parents = {}

size = len(myset)
next_clause_id = size + 1  # 重新编码新生成的子句集


def ifin(myset, clause):
    for key in myset:
        if set(key) == set(clause):
            return True
    return False


def print_set(myset):
    count = 1
    for i in myset:
        print(count, i)
        count += 1


def extract_inside_parenthesis(s):
    start = s.find("(")
    end = s.find(")")
    if start != -1 and end != -1:
        return s[start + 1 : end]
    return None


# 规定变量格式是：长度为1且为小写字母
def is_variable(term):
    return len(term) == 1 and term.islower()


def unify(i, j):
    global next_clause_id
    if i == j:
        return

    for m in range(len(i)):
        for k in range(len(j)):
            # w1表示谓词名称，neg表示肯定与否定，w2表示常量或者变量（注意检查顺序）
            # 提取谓词名称
            w1_1 = i[m].split("(")[0]
            w1_2 = j[k].split("(")[0]

            # 检查谓词是否相同（忽略否定符号）
            pred1 = w1_1[1:] if w1_1.startswith("~") else w1_1
            pred2 = w1_2[1:] if w1_2.startswith("~") else w1_2

            if pred1 != pred2:
                continue

            # 检查是否一个是肯定一个是否定
            neg1 = i[m].startswith("~")
            neg2 = j[k].startswith("~")

            if neg1 == neg2:
                continue

            # 提取变量和常量
            w2_1 = extract_inside_parenthesis(i[m])
            w2_2 = extract_inside_parenthesis(j[k])

            # 处理多变量
            vars_1 = [v for v in w2_1.split(",")] if w2_1 else []
            vars_2 = [v for v in w2_2.split(",")] if w2_2 else []

            # 检查参数数量是否相同
            if len(vars_1) != len(vars_2):
                continue

            # 检查是否有两个常量不相等，或者排除两个变量不相等的情况
            skip_this_pair = False
            for var1, var2 in zip(vars_1, vars_2):
                # 如果两者都是常量且不相等
                if not is_variable(var1) and not is_variable(var2) and var1 != var2:
                    skip_this_pair = True
                    break
                # 如果两个都是变量，但是名字不同
                if is_variable(var1) and is_variable(var2) and var1 != var2:
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

            # 创建新的子句，移除归结的字面量
            new_i = list(i[:m] + i[m + 1 :])
            new_j = list(j[:k] + j[k + 1 :])

            new_clause = []

            # 处理i和j中剩余的文字
            for item in new_i:
                new_item = item
                for var, val in var_map.items():
                    new_item = new_item.replace(f"({var}", f"({val}").replace(
                        f",{var}", f",{val}"
                    )  # 两种情况
                new_clause.append(new_item)

            for item in new_j:
                new_item = item
                for var, val in var_map.items():
                    new_item = new_item.replace(f"({var}", f"({val}").replace(
                        f",{var}", f",{val}"
                    )
                new_clause.append(new_item)

            # 去除重复元素
            new_clause = list(dict.fromkeys(new_clause))

            new_tuple = tuple(new_clause)

            if ifin(myset, new_tuple) == 0:
                print(
                    f"{next_clause_id} R[{myset[i]}{chr(ord('a') + m)}, {myset[j]}{chr(ord('a') + k)}]",
                    end="",
                )
                for var, val in var_map.items():
                    print(f"({var} = {val})", end="")
                print(new_tuple)
                myset[new_tuple] = next_clause_id
                parents[new_tuple] = (i, j, m, k, var_map)  # 记录父子句和替换
                next_clause_id += 1

            return


# 输出证明路径
def print_proof():
    global size

    if () not in myset:
        print("没有找到证明")
        return

    # 从空子句开始回溯
    proof_steps = []
    queue = [()]
    visited = set()

    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        if current in parents:  # 这一条件筛去了初始的子句
            p1, p2, m, k, var_map = parents[current]
            step = (current, p1, p2, m, k, var_map)
            proof_steps.append(step)

            if p1 not in visited:
                queue.append(p1)
            if p2 not in visited:
                queue.append(p2)

    # 创建新的编号映射，从size+1开始为所有在证明路径中出现的子句分配新编号
    new_numbering = {}
    next_id = size + 1

    # 收集所有在证明路径中出现的子句
    all_clauses_in_proof = set()
    for current, p1, p2, _, _, _ in proof_steps:
        if myset[current] > size:
            all_clauses_in_proof.add(current)
        if myset[p1] > size:
            all_clauses_in_proof.add(p1)
        if myset[p2] > size:
            all_clauses_in_proof.add(p2)

    # 按照原始编号顺序分配新编号
    sorted_clauses = sorted(all_clauses_in_proof, key=lambda clause: myset[clause])
    for clause in sorted_clauses:
        new_numbering[clause] = next_id
        next_id += 1

    # 反向输出证明步骤，使用新的编号
    for current, p1, p2, m, k, var_map in reversed(proof_steps):
        print(
            f"{new_numbering[current] if current in new_numbering else myset[current]} R[{new_numbering[p1] if p1 in new_numbering else myset[p1]}{chr(ord('a') + m) if not len(p1) == 1 else ''}, {new_numbering[p2] if p2 in new_numbering else myset[p2]}{chr(ord('a') + k) if not len(p2) == 1 else ''}]",
            end="",
        )
        for var, val in var_map.items():
            print(f"({var} = {val})", end="")
        print(f" {current}")


def resolution(myset):
    staus = True
    while staus:
        # for k in range(len(myset)):
            for i in range(len(myset)):
                for j in range(i + 1, len(myset)):
                    tuple_i = list(myset.keys())[i]
                    tuple_j = list(myset.keys())[j]

                    unify(tuple_i, tuple_j)

                    if () in myset:
                        print_proof()
                        return


if __name__ == "__main__":
    print_set(myset)
    resolution(myset)


# https://www.runoob.com/python/att-dictionary-fromkeys.html
