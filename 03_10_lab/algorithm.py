tuple_1 = ("GradStudent(sue)",)
tuple_2 = ("~GradStudent(x)", "Student(x)")
tuple_3 = ("~Student(x)", "HardWorker(x)")
tuple_4 = ("~HardWorker(sue)",)

myset = {tuple_1: 1, tuple_2: 2, tuple_3: 3, tuple_4: 4}

visit = set()  # 记录已经更改过的元素（那就不要用它了）


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


def unify(i, j):
    if i == j:
        return
    for m in range(len(i)):
        for k in range(len(j)):
            # 设置三个判断条件：谓词：W1，变量名：W2，逻辑：W3（注意判断顺序）
            w1_1 = i[m].split("(")[0]
            w1_2 = j[k].split("(")[0]
            w1_bool = (w1_1 in w1_2) or (w1_2 in w1_1)
            if not w1_bool:
                continue

            w3_1 = i[m][0]
            w3_2 = j[k][0]
            w3_bool = (w3_1 == "~" or w3_2 == "~") and (w3_1 != w3_2)
            if not w3_bool:
                continue

            w2_1 = extract_inside_parenthesis(i[m])
            w2_2 = extract_inside_parenthesis(j[k])
            w2_bool = w2_1 != w2_2

            # 处理多变量情况，拆分多个变量
            vars_1 = [v.strip() for v in w2_1.split(",")] if w2_1 else []
            vars_2 = [v.strip() for v in w2_2.split(",")] if w2_2 else []

            # 确定变量替换映射
            var_map = {}
            if len(vars_1) == len(vars_2):  # 确保变量数量相同
                for var1, var2 in zip(vars_1, vars_2):
                    # 如果两个变量不同，确定哪个是常量哪个是变量
                    if var1 != var2:
                        if len(var1) == 1:  # 假设单字符是变量
                            var_map[var1] = var2
                        elif len(var2) == 1:
                            var_map[var2] = var1

            # if not var_map :  # 旧的单变量逻辑
            #     ch_w = w2_2 if len(w2_2) < len(w2_1) else w2_1
            #     st_w = w2_1 if len(w2_2) < len(w2_1) else w2_2
            #     var_map = {ch_w: st_w}

            # if not var_map:  # 如果没有变量替换，继续下一个循环
            #     continue

            print("R", "[", myset[i], ",", myset[j], "]", end="")

            for var, val in var_map.items():
                print("(", var, "=", val, ")", end="")

            # 更新访问过的元素
            visit.add(i)
            visit.add(j)

            # 创建新的tuple，删去m和k位置的元素
            new_i = i[:m] + i[m + 1 :]
            new_j = j[:k] + j[k + 1 :]

            # 在新tuple中应用变量替换
            for var, val in var_map.items():
                new_i = tuple(item.replace(f"({var})", f"({val})") for item in new_i)
                new_j = tuple(item.replace(f"({var})", f"({val})") for item in new_j)

            new_tuple = new_i + new_j
            print(new_tuple)
            if new_tuple:  # 只有当新tuple非空时才添加到myset
                myset[new_tuple] = len(myset) + 1

            return  # 一次归结后返回


def resolution(myset):
    """
    遍历myset中的所有tuple对，调用unify函数实现归结
    排除已经访问过的tuple
    """
    #  print("开始归结过程...")

    # 获取所有未访问过的tuple
    tuples = [t for t in myset.keys() if t not in visit]

    # 遍历所有可能的tuple对
    for i in range(len(tuples)):
        for j in range(i + 1, len(tuples)):
            tuple_i = tuples[i]
            tuple_j = tuples[j]

            # 如果tuple已经在访问集合中，则跳过
            if tuple_i in visit or tuple_j in visit:
                continue

            # print(f"\n尝试归结: {tuple_i} 和 {tuple_j}")
            unify(tuple_i, tuple_j)

    # 检查是否有新的tuple被添加到myset中
    new_tuples = [t for t in myset.keys() if t not in visit]
    if new_tuples:
        # print("\n发现新的tuple，继续归结...")
        resolution(myset)  # 递归调用继续归结新产生的tuple
    else:
        print("\n归结完成，没有新的tuple生成")


if __name__ == "__main__":
    print_set(myset)
    print("\n开始归结演示:")
    resolution(myset)

    print("\n归结后的结果集合:")
    print_set(myset)

# if __name__ == "__main__":
#     print_set(myset)
#     unify(tuple_3, tuple_4)
