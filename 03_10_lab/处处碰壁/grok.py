from copy import deepcopy

# 表示文字（Literal）
class Literal:
    def __init__(self, predicate, args, negated=False):
        self.predicate = predicate  # 谓词名，例如 "A"
        self.args = args            # 参数列表，例如 ["tony"]
        self.negated = negated      # 是否否定，例如 True 表示 "~A"

    def __str__(self):
        prefix = "~" if self.negated else ""
        return f"{prefix}{self.predicate}({','.join(self.args)})"

    def __eq__(self, other):
        return (self.predicate == other.predicate and
                self.args == other.args and
                self.negated == other.negated)

    def __hash__(self):
        return hash(str(self))

# 表示子句（Clause）
class Clause:
    def __init__(self, literals):
        self.literals = set(literals)  # 文字集合

    def __str__(self):
        return "{" + ", ".join(str(lit) for lit in self.literals) + "}"

    def __eq__(self, other):
        return self.literals == other.literals

    def __hash__(self):
        return hash(frozenset(self.literals))

# 变量替换（Unification）
def unify(lit1, lit2):
    if lit1.predicate != lit2.predicate or lit1.negated == lit2.negated:
        return None  # 谓词不同或否定状态相同，无法归结
    if len(lit1.args) != len(lit2.args):
        return None  # 参数数量不同，无法归结

    substitution = {}
    for arg1, arg2 in zip(lit1.args, lit2.args):
        if arg1.startswith("X") or arg1.startswith("Y") or arg1.startswith("Z"):  # arg1 是变量
            substitution[arg1] = arg2
        elif arg2.startswith("X") or arg2.startswith("Y") or arg2.startswith("Z"):  # arg2 是变量
            substitution[arg2] = arg1
        elif arg1 != arg2:
            return None  # 参数不匹配
    return substitution

# 应用替换
def apply_substitution(clause, substitution):
    new_literals = set()
    for lit in clause.literals:
        new_args = []
        for arg in lit.args:
            new_args.append(substitution.get(arg, arg))
        new_literals.add(Literal(lit.predicate, new_args, lit.negated))
    return Clause(new_literals)

# 归结算法
def resolution(kb):
    clauses = set(kb)  # 知识库中的子句集合
    new_clauses = set()  # 新生成的子句
    step = 1

    while True:
        # 遍历所有子句对
        new_added = False
        clauses_list = list(clauses)
        for i in range(len(clauses_list)):
            for j in range(i + 1, len(clauses_list)):
                c1, c2 = clauses_list[i], clauses_list[j]

                # 寻找互补文字
                for lit1 in c1.literals:
                    for lit2 in c2.literals:
                        substitution = unify(lit1, lit2)
                        if substitution:
                            # 应用替换
                            c1_sub = apply_substitution(c1, substitution)
                            c2_sub = apply_substitution(c2, substitution)

                            # 归结：移除互补文字，合并剩余文字
                            new_literals = (c1_sub.literals | c2_sub.literals) - {lit1} - {lit2}
                            new_clause = Clause(new_literals)

                            # 输出归结步骤
                            print(f"{step}. R[{i+1}, {j+1}] -> {new_clause}")

                            if not new_clause.literals:  # 空子句
                                print("Empty clause found, KB is unsatisfiable.")
                                return False

                            if new_clause not in clauses and new_clause not in new_clauses:
                                new_clauses.add(new_clause)
                                new_added = True
                            step += 1

        # 将新子句加入知识库
        clauses.update(new_clauses)
        new_clauses.clear()

        if not new_added:  # 没有新子句生成
            print("No new clauses generated, KB may be satisfiable.")
            return True

# 创建知识库（KB）
# 创建知识库（KB）
def create_kb():
    kb = [
        Clause([Literal("on", ["tony", "mike"])]),  # {on(tony, mike)}
        Clause([Literal("on", ["mike", "john"])]),  # {on(mike, john)}
        Clause([Literal("green", ["tony"])]),       # {green(tony)}
        Clause([Literal("green", ["john"], negated=True)]),  # {~green(john)}
        Clause([Literal("on", ["XX", "YY"], negated=True),   # {~on(XX, YY), ~green(XX), green(YY)}
                Literal("green", ["XX"], negated=True),
                Literal("green", ["YY"])])
    ]
    return kb

# 主函数
def main():
    kb = create_kb()

    # 打印初始知识库
    print("Initial Knowledge Base (KB):")
    for i, clause in enumerate(kb, 1):
        print(f"{i}. {clause}")

    print("\nResolution Steps:")
    resolution(kb)

if __name__ == "__main__":
    main()