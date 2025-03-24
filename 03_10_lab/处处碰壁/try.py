import re


tuple_1 = ("A(tony)",)
tuple_2 = ("A(mike)",)
tuple_3 = ("A(john)",)
tuple_4 = ("L(tony,rain)",)
tuple_5 = ("L(tony,snow)",)
tuple_6 = ("~A(x)","S(x)","C(x)")
tuple_7 = ("~C(y)","~L(y,rain)")
tuple_8 = ("L(z,snow)","~S(z)")
tuple_9 = ("~L(tony,u)","~L(mike,u)")
tuple_10 = ("L(tony,v)","L(mike,v)")
tuple_11 = ("~A(w)","~C(w)","S(w)")


myset = {tuple_1: 1, tuple_2: 2, tuple_3: 3, tuple_4: 4, tuple_5: 5, tuple_6: 6, tuple_7: 7, tuple_8: 8, tuple_9: 9, tuple_10: 10, tuple_11: 11}

parents = {}
size = len(myset)
next_clause_id = size + 1


def ifin(myset, clause):
    for key in myset:
        if set(key) == set(clause):
            return True
    return False


def in_myset(myset, clause):
    for key in myset:
        if set(key) == set(clause):
            return True
    return False


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
    global next_clause_id  # 声明使用全局变量
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

            # 应用变量替换
            resolved_clause = []

            # 处理i中剩余的文字
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

            # 去除resolved_clause中的重复元素
            resolved_clause = list(dict.fromkeys(resolved_clause))

            # 转换回元组并添加到集合
            new_tuple = tuple(resolved_clause)

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

            # 找到一个归结后返回
            return


# 新增函数，输出证明路径
def print_proof():
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

        if current in parents:
            p1, p2, m, k, var_map = parents[current]
            step = (current, p1, p2, m, k, var_map)
            proof_steps.append(step)

            if p1 not in visited:
                queue.append(p1)
            if p2 not in visited:
                queue.append(p2)

    # 反向输出证明步骤
    print("\n证明路径:")
    for current, p1, p2, m, k, var_map in reversed(proof_steps):
        print(
            f"{myset[current]} R[{myset[p1]}{chr(ord('a') + m)}, {myset[p2]}{chr(ord('a') + k)}]",
            end="",
        )
        for var, val in var_map.items():
            print(f"({var} = {val})", end="")
        print(f" {current}")


def resolution(myset):
    found_empty_clause = False
    iteration = 1

    # while not found_empty_clause:
    #     print(f"\n第{iteration}轮归结:")
    #     iteration += 1
    #
    #     # 记录本轮是否有新子句生成
    #     new_clause_added = False

    # 遍历所有可能的元组对
    for k in range(len(myset)):
        for i in range(len(myset)):
            for j in range(i + 1, len(myset)):
                tuple_i = list(myset.keys())[i]
                tuple_j = list(myset.keys())[j]

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
                    print_proof()
                    # found_empty_clause = True
                    # break
                    return
        #     if found_empty_clause:
        #         break
        # if found_empty_clause:
        #     break
        # 如果没有新子句生成，则停止归结
        if not new_clause_added:
            print("\n归结完成，没有新的子句生成")
            break

    return found_empty_clause


if __name__ == "__main__":
    print_set(myset)
    print("\n开始归结演示:")
    resolution(myset)

    # print("\n归结后的结果集合:")
    # print_set(myset)
