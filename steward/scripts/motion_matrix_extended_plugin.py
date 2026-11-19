"""Extend only test observation/job deadlines for the separately labelled timeout reruns."""

import ast
import copy
import json
import os
from pathlib import Path


class ExtendedDeadlines(ast.NodeTransformer):
    """Keep sleeps, control thresholds and success assertions intact while extending waits."""

    def __init__(self, multiplier):
        self.multiplier = multiplier
        self.changes = []

    def scale(self, value, reason):
        if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
            previous = value.value
            value.value *= self.multiplier
            self.changes.append({"line": value.lineno, "kind": reason, "before": previous, "after": value.value})
        return value

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Add) and isinstance(node.left, ast.Call) and ast.unparse(node.left.func) == "time.time":
            node.right = self.scale(node.right, "test wall deadline")
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        name = ast.unparse(node.func)
        for keyword in node.keywords:
            if keyword.arg in ("timeout", "timeout_s"):
                keyword.value = self.scale(keyword.value, "explicit request/job wait")
        if name == "_explore" and len(node.args) > 1:
            node.args[1] = self.scale(node.args[1], "exploration job wait")
        if name == "Chores" and not any(k.arg == "timeout_s" for k in node.keywords):
            node.keywords.append(ast.keyword(arg="timeout_s", value=ast.Constant(180.0*self.multiplier)))
            self.changes.append({"line": node.lineno, "kind": "Chores job wait", "before": 180, "after": 180*self.multiplier})
        return node

    def visit_Dict(self, node):
        self.generic_visit(node)
        for index, key in enumerate(node.keys):
            if isinstance(key, ast.Constant) and key.value == "timeoutS":
                node.values[index] = self.scale(node.values[index], "HTTP job wait payload")
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        arguments = node.args.posonlyargs + node.args.args
        for argument, default in zip(arguments[-len(node.args.defaults):], node.args.defaults):
            if argument.arg in ("timeout", "timeout_s"):
                self.scale(default, "test helper default wait")
        return node


def extend_functions(source, names, multiplier, namespace, filename):
    """Compile selected test functions and local helpers, checking every success assertion."""
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and (n.name in names or n.name.startswith("_"))]
    original = [ast.dump(n) for f in functions for n in ast.walk(f) if isinstance(n, ast.Assert)]
    transformer = ExtendedDeadlines(multiplier)
    modified = [transformer.visit(copy.deepcopy(f)) for f in functions]
    assertions = [ast.dump(n) for f in modified for n in ast.walk(f) if isinstance(n, ast.Assert)]
    if original != assertions:
        raise ValueError("deadline extension changed a success assertion")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    compiled = ast.fix_missing_locations(ast.Module(body=[future, *modified], type_ignores=[]))
    exec(compile(compiled, filename, "exec"), namespace)
    return transformer.changes


def pytest_collection_modifyitems(items):
    multiplier = float(os.environ["STEWARD_MATRIX_TIMEOUT_MULTIPLIER"])
    if multiplier <= 1:
        raise ValueError("extended reruns require a timeout multiplier greater than one")
    modules = {}
    for item in items:
        modules.setdefault(item.module, []).append(item)
    changes = []
    for module, selected in modules.items():
        namespace = dict(vars(module))
        path = Path(module.__file__)
        changes.extend(extend_functions(path.read_text(encoding="utf-8-sig"),
                                        {i.originalname for i in selected}, multiplier, namespace, str(path)))
        for item in selected:
            item.obj = namespace[item.originalname]
    output = Path(os.environ["STEWARD_MATRIX_CASE_OUTPUT"])
    (output/"deadline-overrides.json").write_text(json.dumps(changes, indent=2)+"\n", encoding="utf-8")
