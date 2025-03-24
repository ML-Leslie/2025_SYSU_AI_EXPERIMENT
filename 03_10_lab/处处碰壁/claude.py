class Term:
    """表示一阶逻辑中的项（常量、变量或函数）"""

    def __init__(self, symbol, args=None):
        self.symbol = symbol
        self.args = args if args is not None else []

    def __eq__(self, other):
        if not isinstance(other, Term):
            return False
        return self.symbol == other.symbol and self.args == other.args

    def __hash__(self):
        return hash((self.symbol, tuple(self.args)))

    def __str__(self):
        if not self.args:
            return self.symbol
        return f"{self.symbol}({', '.join(str(arg) for arg in self.args)})"

    def is_variable(self):
        """判断一个项是否为变量（大写字母开头）"""
        return self.symbol[0].islower() and len(self.symbol) == 1 and len(self.args) == 0

    def substitute(self, substitution):
        """应用替换到项"""
        if self.is_variable() and self in substitution:
            return substitution[self]
        if not self.args:
            return self
        new_args = [arg.substitute(substitution) for arg in self.args]
        return Term(self.symbol, new_args)


class Literal:
    """表示一阶逻辑中的文字（原子公式或其否定）"""

    def __init__(self, predicate, args, negated=False):
        self.predicate = predicate
        self.args = args
        self.negated = negated

    def __eq__(self, other):
        if not isinstance(other, Literal):
            return False
        return (self.predicate == other.predicate and
                self.args == other.args and
                self.negated == other.negated)

    def __hash__(self):
        return hash((self.predicate, tuple(self.args), self.negated))

    def __str__(self):
        atom = f"{self.predicate}({', '.join(str(arg) for arg in self.args)})"
        return f"¬{atom}" if self.negated else atom

    def negate(self):
        """返回文字的否定"""
        return Literal(self.predicate, self.args, not self.negated)

    def substitute(self, substitution):
        """应用替换到文字"""
        new_args = [arg.substitute(substitution) for arg in self.args]
        return Literal(self.predicate, new_args, self.negated)


class Clause:
    """表示文字的析取（子句）"""

    def __init__(self, literals, id=None):
        self.literals = set(literals)
        self.id = id  # 为子句添加ID，以便跟踪归结过程

    def __eq__(self, other):
        if not isinstance(other, Clause):
            return False
        return self.literals == other.literals

    def __hash__(self):
        return hash(frozenset(self.literals))

    def __str__(self):
        if not self.literals:
            return "□"  # 空子句表示矛盾
        return " ∨ ".join(str(lit) for lit in self.literals)

    def substitute(self, substitution):
        """应用替换到子句"""
        new_literals = {lit.substitute(substitution) for lit in self.literals}
        return Clause(new_literals, self.id)


def variables_in_term(term):
    """返回项中的所有变量"""
    if term.is_variable():
        return {term}

    vars_set = set()
    for arg in term.args:
        vars_set.update(variables_in_term(arg))
    return vars_set


def variables_in_literal(literal):
    """返回文字中的所有变量"""
    vars_set = set()
    for arg in literal.args:
        vars_set.update(variables_in_term(arg))
    return vars_set


def variables_in_clause(clause):
    """返回子句中的所有变量"""
    vars_set = set()
    for literal in clause.literals:
        vars_set.update(variables_in_literal(literal))
    return vars_set


def rename_variables(clause, suffix):
    """重命名子句中的变量，避免冲突"""
    variables = variables_in_clause(clause)
    substitution = {var: Term(var.symbol + suffix) for var in variables}
    new_clause = clause.substitute(substitution)
    new_clause.id = clause.id  # 保留原始ID
    return new_clause


def unify(term1, term2, substitution=None):
    """尝试统一两个项，返回最一般统一替换（MGU）"""
    if substitution is None:
        substitution = {}

    # 应用当前替换
    t1 = term1.substitute(substitution) if isinstance(term1, Term) else term1
    t2 = term2.substitute(substitution) if isinstance(term2, Term) else term2

    if t1 == t2:
        return substitution

    if t1.is_variable():
        return unify_var(t1, t2, substitution)

    if t2.is_variable():
        return unify_var(t2, t1, substitution)

    if t1.symbol != t2.symbol or len(t1.args) != len(t2.args):
        return None  # 无法统一

    # 递归统一参数
    for a1, a2 in zip(t1.args, t2.args):
        substitution = unify(a1, a2, substitution)
        if substitution is None:
            return None

    return substitution


def unify_var(var, term, substitution):
    """变量与项的统一"""
    # 检查是否已经有替换
    if var in substitution:
        return unify(substitution[var], term, substitution)

    # 发生时检查 - 如果变量出现在term中，则无法统一
    if isinstance(term, Term) and var in variables_in_term(term):
        return None

    # 添加新的替换
    new_subst = substitution.copy()
    new_subst[var] = term
    return new_subst


def resolve(clause1, clause2):
    """对两个子句进行归结，返回所有可能的归结式"""
    # 变量重命名以避免冲突
    c1 = rename_variables(clause1, "_1")
    c2 = rename_variables(clause2, "_2")

    resolvents = []

    for lit1 in c1.literals:
        for lit2 in c2.literals:
            # 尝试归结互补文字
            if lit1.predicate == lit2.predicate and lit1.negated != lit2.negated:
                # 尝试统一参数
                subst = None
                if len(lit1.args) == len(lit2.args):
                    subst = {}
                    for a1, a2 in zip(lit1.args, lit2.args):
                        temp_subst = unify(a1, a2, subst)
                        if temp_subst is None:
                            subst = None
                            break
                        subst = temp_subst

                if subst is not None:
                    # 创建归结式
                    new_literals = {l.substitute(subst) for l in c1.literals if l != lit1}
                    new_literals.update({l.substitute(subst) for l in c2.literals if l != lit2})
                    resolvent = Clause(new_literals)
                    # 记录本次归结所使用的父子句
                    resolvent.parents = (clause1.id, clause2.id)
                    resolvent.used_literals = (str(lit1), str(lit2))
                    resolvents.append(resolvent)

    return resolvents


def resolution_with_trace(clauses):
    """使用归结原理进行定理证明，并记录归结过程"""
    new_clauses = set(clauses)
    all_clauses = new_clauses.copy()

    # 用于记录每一步归结的信息
    trace = []

    step = 1
    while True:
        pairs = [(c1, c2) for c1 in new_clauses for c2 in all_clauses]
        new_clauses = set()

        for c1, c2 in pairs:
            resolvents = resolve(c1, c2)
            for resolvent in resolvents:
                resolvent.id = len(all_clauses) + len(new_clauses) + 1  # 为新子句分配ID

                # 记录该步归结信息
                trace_info = {
                    'step': step,
                    'clause1': c1,
                    'clause2': c2,
                    'resolvent': resolvent,
                    'resolvent_id': resolvent.id,
                    'used_literals': resolvent.used_literals
                }

                if len(resolvent.literals) == 0:  # 空子句表示矛盾
                    trace.append(trace_info)
                    return True, trace  # 证明成功

                if resolvent not in all_clauses:
                    new_clauses.add(resolvent)
                    trace.append(trace_info)
                    step += 1

        if not new_clauses or new_clauses.issubset(all_clauses):
            return False, trace  # 无法继续归结

        all_clauses.update(new_clauses)


def setup_clauses():
    """设置题目给定的子句集合"""
    # 创建常量
    john = Term("john")
    mike = Term("mike")
    tony = Term("tony")
    rain = Term("rain")
    snow = Term("snow")

    # 创建变量
    x = Term("x")
    y = Term("y")
    z = Term("z")
    u = Term("u")
    v = Term("v")
    w = Term("w")

    # 定义谓词
    A = lambda t: Literal("A", [t])
    C = lambda t: Literal("C", [t])
    S = lambda t: Literal("S", [t])
    L = lambda t1, t2: Literal("L", [t1, t2])

    # 构建子句集合
    clauses = [
        Clause({A(john)}, 1),  # 子句1: A(john)
        Clause({A(mike)}, 2),  # 子句2: A(mike)
        Clause({A(tony)}, 3),  # 子句3: A(tony)
        Clause({L(tony, rain)}, 4),  # 子句4: L(tony, rain)
        Clause({L(tony, snow)}, 5),  # 子句5: L(tony, snow)
        Clause({C(x), S(x), A(x).negate()}, 6),  # 子句6: C(x) ∨ S(x) ∨ ¬A(x)
        Clause({L(y, rain).negate(), C(y).negate()}, 7),  # 子句7: ¬L(y, rain) ∨ ¬C(y)
        Clause({L(z, snow), S(z).negate()}, 8),  # 子句8: L(z, snow) ∨ ¬S(z)
        Clause({L(mike, u).negate(), L(tony, u).negate()}, 9),  # 子句9: ¬L(mike, u) ∨ ¬L(tony, u)
        Clause({L(mike, v), L(tony, v)}, 10),  # 子句10: L(mike, v) ∨ L(tony, v)
        Clause({C(w).negate(), A(w).negate(), S(w)}, 11)  # 子句11: ¬C(w) ∨ ¬A(w) ∨ S(w)
    ]

    return clauses


def print_resolution_trace(trace):
    """打印归结过程"""
    print("====== 归结推理过程 ======")
    for step_info in trace:
        resolvent = step_info['resolvent']
        parents = step_info['clause1'], step_info['clause2']
        used_literals = step_info['used_literals']

        print(f"R[{parents[0].id},{parents[1].id}] = 归结 {used_literals[0]} 和 {used_literals[1]}")
        print(f"生成子句 {resolvent.id}: {resolvent}")
        print("-" * 40)


def main():
    """主函数，运行归结过程"""
    clauses = setup_clauses()

    print("初始子句集合:")
    for clause in clauses:
        print(f"子句 {clause.id}: {clause}")
    print("\n")

    success, trace = resolution_with_trace(clauses)

    print_resolution_trace(trace)

    if success:
        print("归结结果: 成功（推出空子句，证明定理）")
    else:
        print("归结结果: 失败（无法推出空子句）")


if __name__ == "__main__":
    main()