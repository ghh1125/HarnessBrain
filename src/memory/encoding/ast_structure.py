"""Normalized component ASTs and ordered tree edit distance."""

import ast
from typing import Any, Optional


AstTree = dict[str, Any]

_STRUCTURAL_NODE_TYPES = {
    "FunctionDef", "AsyncFunctionDef", "Lambda", "arguments", "arg",
    "Return", "Delete", "Assign", "AnnAssign", "AugAssign", "NamedExpr",
    "For", "AsyncFor", "While", "If", "With", "AsyncWith",
    "Match", "match_case", "Raise", "Try", "TryStar", "Assert",
    "Call", "Await", "Yield", "YieldFrom",
    "BoolOp", "BinOp", "UnaryOp", "IfExp", "Dict", "Set",
    "ListComp", "SetComp", "DictComp", "GeneratorExp", "comprehension",
    "Compare", "Attribute", "Subscript", "Starred", "Slice", "keyword",
    "Break", "Continue", "Pass",
    "And", "Or", "Add", "Sub", "Mult", "MatMult", "Div", "Mod", "Pow",
    "LShift", "RShift", "BitOr", "BitXor", "BitAnd", "FloorDiv",
    "Invert", "Not", "UAdd", "USub",
    "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "Is", "IsNot", "In", "NotIn",
}


def _iter_functional_units(tree: ast.AST):
    """Yield module functions and class methods without duplicating nested nodes."""
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield child


def _ast_to_forest(node: ast.AST) -> list[AstTree]:
    """Keep behavior-bearing AST nodes and promote children of lexical nodes."""
    children: list[AstTree] = []
    for _, value in ast.iter_fields(node):
        ast_children: list[ast.AST] = []
        if isinstance(value, ast.AST):
            ast_children = [value]
        elif isinstance(value, list):
            ast_children = [item for item in value if isinstance(item, ast.AST)]
        for child in ast_children:
            children.extend(_ast_to_forest(child))

    node_type = type(node).__name__
    if node_type in _STRUCTURAL_NODE_TYPES:
        return [{"label": node_type, "children": children}]
    return children


def _ast_to_tree(node: ast.AST) -> AstTree:
    """Convert a functional unit into its normalized structural AST."""
    forest = _ast_to_forest(node)
    if len(forest) != 1:
        return {"label": type(node).__name__, "children": forest}
    return forest[0]


def component_ast_signature(
    source: str,
    component_keywords: list[str],
) -> Optional[AstTree]:
    """Return a normalized AST for the functions most relevant to a component."""
    if not source:
        return None
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError):
        return None

    keywords = [keyword.lower() for keyword in component_keywords if keyword]
    ranked: list[tuple[tuple[int, int], ast.AST]] = []
    for node in _iter_functional_units(tree):
        segment = (ast.get_source_segment(source, node) or "").lower()
        name = node.name.lower()
        name_hits = sum(keyword in name for keyword in keywords)
        total_hits = sum(keyword in segment for keyword in keywords)
        if total_hits:
            ranked.append(((name_hits, total_hits), node))

    if not ranked:
        return None

    best_score = max(score for score, _ in ranked)
    matched = [
        node
        for score, node in ranked
        if score == best_score
    ]
    matched.sort(key=lambda item: getattr(item, "lineno", 0))
    return {
        "label": "Component",
        "children": [_ast_to_tree(node) for node in matched],
    }


def _is_tree(value: Any) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("label"), str):
        return False
    children = value.get("children")
    return isinstance(children, list) and all(_is_tree(child) for child in children)


def ast_node_count(tree: Optional[AstTree]) -> int:
    """Count nodes in a serialized AST tree."""
    if not _is_tree(tree):
        return 0
    return 1 + sum(ast_node_count(child) for child in tree["children"])


def _postorder_index(tree: AstTree) -> tuple[list[str], list[int]]:
    """Return one-based postorder labels and leftmost-leaf indices."""
    labels = [""]
    leftmost = [0]

    def visit(node: AstTree) -> int:
        child_indices = [visit(child) for child in node["children"]]
        index = len(labels)
        labels.append(node["label"])
        leftmost.append(leftmost[child_indices[0]] if child_indices else index)
        return index

    visit(tree)
    return labels, leftmost


def _keyroots(leftmost: list[int]) -> list[int]:
    """Return the largest postorder index for each leftmost leaf."""
    last_for_leaf: dict[int, int] = {}
    for index in range(1, len(leftmost)):
        last_for_leaf[leftmost[index]] = index
    return sorted(last_for_leaf.values())


def ast_tree_edit_distance(
    left: Optional[AstTree],
    right: Optional[AstTree],
) -> Optional[int]:
    """Compute ordered TED with unit insertion, deletion, and relabel costs."""
    if not _is_tree(left) or not _is_tree(right):
        return None
    if left == right:
        return 0

    left_labels, leftmost_left = _postorder_index(left)
    right_labels, leftmost_right = _postorder_index(right)
    left_size = len(left_labels) - 1
    right_size = len(right_labels) - 1
    tree_distance = [
        [0] * (right_size + 1)
        for _ in range(left_size + 1)
    ]

    for left_root in _keyroots(leftmost_left):
        for right_root in _keyroots(leftmost_right):
            left_start = leftmost_left[left_root]
            right_start = leftmost_right[right_root]
            forest_distance = [
                [0] * (right_root - right_start + 2)
                for _ in range(left_root - left_start + 2)
            ]

            for left_index in range(left_start, left_root + 1):
                row = left_index - left_start + 1
                forest_distance[row][0] = forest_distance[row - 1][0] + 1
            for right_index in range(right_start, right_root + 1):
                column = right_index - right_start + 1
                forest_distance[0][column] = (
                    forest_distance[0][column - 1] + 1
                )

            for left_index in range(left_start, left_root + 1):
                row = left_index - left_start + 1
                for right_index in range(right_start, right_root + 1):
                    column = right_index - right_start + 1
                    delete_cost = forest_distance[row - 1][column] + 1
                    insert_cost = forest_distance[row][column - 1] + 1

                    if (
                        leftmost_left[left_index] == left_start
                        and leftmost_right[right_index] == right_start
                    ):
                        relabel_cost = (
                            0
                            if left_labels[left_index] == right_labels[right_index]
                            else 1
                        )
                        replace_cost = (
                            forest_distance[row - 1][column - 1]
                            + relabel_cost
                        )
                        distance = min(delete_cost, insert_cost, replace_cost)
                        forest_distance[row][column] = distance
                        tree_distance[left_index][right_index] = distance
                    else:
                        prefix_row = leftmost_left[left_index] - left_start
                        prefix_column = (
                            leftmost_right[right_index] - right_start
                        )
                        replace_cost = (
                            forest_distance[prefix_row][prefix_column]
                            + tree_distance[left_index][right_index]
                        )
                        forest_distance[row][column] = min(
                            delete_cost,
                            insert_cost,
                            replace_cost,
                        )

    return tree_distance[left_size][right_size]


def ast_signature_similarity(
    left: Optional[AstTree],
    right: Optional[AstTree],
) -> Optional[float]:
    """Return 1 - normalized ordered tree edit distance."""
    distance = ast_tree_edit_distance(left, right)
    if distance is None:
        return None
    normalizer = ast_node_count(left) + ast_node_count(right)
    if normalizer == 0:
        return None
    return max(0.0, min(1.0, 1.0 - distance / normalizer))
