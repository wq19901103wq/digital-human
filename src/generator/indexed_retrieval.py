"""Replace two exact storage operations in supported frozen retrieval functions."""
from __future__ import annotations

import ast
from functools import update_wrapper
import inspect
import textwrap
from types import FunctionType


def nodes(source):
    return ast.parse(textwrap.dedent(source)).body


def same(left, right):
    return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)


def indexed_retrieve(original):
    """Leave every scoring/ordering instruction intact; unknown snapshots fall back."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    except (OSError, TypeError, SyntaxError):
        return original
    exclusions = nodes('''
        allowed = {str(row['id']) for row in filter_rows(self._rows, history_case)}
        exclude_ids = set(exclude_ids or ()) | {str(row['id']) for row in self._rows
                                               if str(row['id']) not in allowed}
    ''')
    lexical = nodes('''
        sample_text = " ".join(row["context"])
        sample_terms = self._terms_by_id[str(row["id"])]
        overlap = sum(min(count, sample_terms.get(term, 0)) for term, count in query_terms.items())
        length_similarity = 1.0 / (1.0 + abs(len(query) - len(sample_text)) / 20.0)
    ''')

    class Indexed(ast.NodeTransformer):
        def __init__(self):
            self.exclusions = 0
            self.lexical = 0
            self.loops = 0

        def generic_visit(self, node):
            node = super().generic_visit(node)
            for field, body in ast.iter_fields(node):
                if not isinstance(body, list):
                    continue
                for expected, replacement, counter in (
                    (exclusions, 'exclude_ids = self._disk_reader.excluded_ids(history_case, exclude_ids)', 'exclusions'),
                    (lexical, 'length_similarity = 1.0 / (1.0 + abs(len(query) - self._disk_reader.context_length(str(row["id"]))) / 20.0)\noverlap = indexed_overlaps.get(str(row["id"]), 0)', 'lexical'),
                ):
                    for position in range(len(body) - len(expected) + 1):
                        if all(same(actual, wanted) for actual, wanted in
                               zip(body[position:position + len(expected)], expected)):
                            body[position:position + len(expected)] = nodes(replacement)
                            setattr(self, counter, getattr(self, counter) + 1)
                            break
                setattr(node, field, body)
            return node

        def visit_For(self, node):
            node = self.generic_visit(node)
            if same(node.target, ast.Name(id='row', ctx=ast.Store())) and same(
                    node.iter, ast.Name(id='candidates', ctx=ast.Load())):
                self.loops += 1
                return [*nodes('indexed_overlaps = self._disk_reader.term_overlaps(query_terms)'), node]
            return node

    transform = Indexed()
    tree = transform.visit(tree)
    if (transform.exclusions, transform.lexical, transform.loops) != (1, 1, 1) or original.__closure__:
        return original
    ast.fix_missing_locations(tree)
    namespace = {}
    exec(compile(tree, original.__code__.co_filename, 'exec'), original.__globals__, namespace)
    compiled = namespace[original.__name__]
    result = FunctionType(compiled.__code__, original.__globals__, original.__name__, original.__defaults__)
    result.__kwdefaults__ = original.__kwdefaults__
    return update_wrapper(result, original)
