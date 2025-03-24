import re

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

# 改为集合，每个元素是(子句, ID)的元组
myset = {
    tuple_1,
    tuple_2,
    tuple_3,
    tuple_4,
    tuple_5,
    tuple_6,
    tuple_7,
    tuple_8,
    tuple_9,
    tuple_10,
    tuple_11,
}
size = len(myset)
next_clause_id = size + 1


def ifin(myset, clause):
    for item in myset:
        if set(item[0]) == set(clause):
            return True
    return False


def in_myset(myset, clause):
    for item in myset:
        if set(item[0]) == set(clause):
            return True
    return False


# def print_set(myset):
#     # 因为集合无序，为了有序打印，我们可以按ID排序
#     sorted_items = sorted(myset, key=lambda x: x[1])
#     for item in sorted_items:
#         print(item[1], " ", item[0])


def extract_inside_parenthesis(s):
    start = s.find("(")
    end = s.find(")")
    if start != -1 and end != -1:
        return s[start + 1 : end]
    return None


def is_variable(term):
    """判断一个项是否为变量（假设变量是单个小写字母）"""
    return len(term) == 1 and term.islower()


def unify(i, j):
    global next_clause_id, myset  # 声明使用全局变量
    if i == j:
        return

    # 找到i, j在myset中对应的ID
    i_id = None
    j_id = None
    for item in myset:
        if item[0] == i:
            i_id = item[1]
        if item[0] == j:
            j_id = item[1]

    if i_id is None or j_id is None:
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

            # 创建新的子句，移除归结的字面量
            new_i = list(i[:m] + i[m + 1 :])
            new_j = list(j[:k] + j[k + 1 :])

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

            if not ifin(myset, new_tuple):
                print(
                    f"{next_clause_id} R[{i_id}{chr(ord('a') + m)}, {j_id}{chr(ord('a') + k)}]",
                    end="",
                )
                for var, val in var_map.items():
                    print(f"({var} = {val})", end="")
                print(new_tuple)
                myset.add((new_tuple, next_clause_id))  # 使用add而不是append
                next_clause_id += 1

            # 找到一个归结后返回
            return


def resolution(myset):
    # global myset  # 声明使用全局变量
    found_empty_clause = False
    iteration = 1
    new_clause_added = False

    # 因为集合是无序的，我们需要转换为列表进行处理
    myset = list(myset)

    # 遍历所有可能的元组对
    for k in range(len(myset)):
        # 使用列表进行双重循环，确保有序性
        for i in range(len(myset)):
            for j in range(i + 1, len(myset)):
                tuple_i = myset[i]
                tuple_j = myset[j]

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

        # 如果没有新子句生成，则停止归结
        if not new_clause_added:
            print("\n归结完成，没有新的子句生成")
            break

    return found_empty_clause


if __name__ == "__main__":
   #  print("\n开始归结演示:")
    resolution(myset)
