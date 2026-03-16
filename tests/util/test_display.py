"""Tests for lutris.util.display module, particularly annotation safety
when optional dependencies like GnomeDesktop are unavailable."""

import ast
from pathlib import Path
from unittest import TestCase


class TestDisplayAnnotationSafety(TestCase):
    """Verify that type annotations in display.py don't reference
    conditionally-imported modules in a way that would fail at runtime.

    Commit 0c88229ac added a bare type annotation
    `-> Optional[GnomeDesktop.RROutput]` which is evaluated at class
    definition time. When GnomeDesktop is None (library unavailable),
    this causes an AttributeError that prevents Lutris from launching.

    The fix is to quote such annotations so they become string literals
    and are not evaluated at runtime.
    """

    DISPLAY_MODULE_PATH = Path(__file__).resolve().parents[2] / "lutris" / "util" / "display.py"

    # Modules that are conditionally imported and may be None at runtime
    CONDITIONAL_MODULES = {"GnomeDesktop"}

    def _parse_display_module(self):
        source = self.DISPLAY_MODULE_PATH.read_text(encoding="utf-8")
        return ast.parse(source, filename=str(self.DISPLAY_MODULE_PATH))

    def _find_unquoted_refs_in_annotation(self, annotation, modules):
        """Find references to conditional modules in an annotation AST node
        that are NOT wrapped in a string (i.e., would be evaluated at runtime)."""
        if annotation is None:
            return []
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            # String annotation — safe, not evaluated at runtime
            return []
        if isinstance(annotation, ast.Name) and annotation.id in modules:
            return [annotation.id]
        if isinstance(annotation, ast.Attribute):
            if isinstance(annotation.value, ast.Name) and annotation.value.id in modules:
                return [f"{annotation.value.id}.{annotation.attr}"]
        # Recurse into subscripts like Optional[GnomeDesktop.RROutput]
        if isinstance(annotation, ast.Subscript):
            refs = self._find_unquoted_refs_in_annotation(annotation.value, modules)
            if isinstance(annotation.slice, ast.Tuple):
                for elt in annotation.slice.elts:
                    refs.extend(self._find_unquoted_refs_in_annotation(elt, modules))
            else:
                refs.extend(self._find_unquoted_refs_in_annotation(annotation.slice, modules))
            return refs
        if isinstance(annotation, ast.BinOp):
            return self._find_unquoted_refs_in_annotation(
                annotation.left, modules
            ) + self._find_unquoted_refs_in_annotation(annotation.right, modules)
        return []

    def test_no_unquoted_conditional_module_annotations(self):
        """All type annotations referencing conditionally-imported modules
        (like GnomeDesktop) must be quoted strings to prevent AttributeError
        when the module is unavailable."""
        tree = self._parse_display_module()
        errors = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check return annotation
                for ref in self._find_unquoted_refs_in_annotation(node.returns, self.CONDITIONAL_MODULES):
                    errors.append(
                        f"Line {node.returns.lineno}: unquoted annotation '{ref}' in return type of '{node.name}()'"
                    )
                # Check argument annotations
                all_args = (
                    node.args.posonlyargs
                    + node.args.args
                    + node.args.kwonlyargs
                    + ([node.args.vararg] if node.args.vararg else [])
                    + ([node.args.kwarg] if node.args.kwarg else [])
                )
                for arg in all_args:
                    for ref in self._find_unquoted_refs_in_annotation(arg.annotation, self.CONDITIONAL_MODULES):
                        errors.append(
                            f"Line {arg.annotation.lineno}: unquoted annotation '{ref}' "
                            f"in parameter '{arg.arg}' of '{node.name}()'"
                        )
            elif isinstance(node, ast.AnnAssign):
                for ref in self._find_unquoted_refs_in_annotation(node.annotation, self.CONDITIONAL_MODULES):
                    errors.append(f"Line {node.annotation.lineno}: unquoted annotation '{ref}' in variable annotation")

        self.assertEqual(
            errors, [], "Found unquoted annotations referencing conditionally-imported modules:\n" + "\n".join(errors)
        )

    def test_gnomedesktop_import_is_guarded(self):
        """GnomeDesktop import must be wrapped in try/except with a fallback."""
        tree = self._parse_display_module()

        # Find the try/except that imports GnomeDesktop
        found_guarded_import = False
        found_fallback = False

        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                # Check if any handler sets GnomeDesktop = None or
                # LIB_GNOME_DESKTOP_AVAILABLE = False
                for handler in node.handlers:
                    for child in ast.walk(handler):
                        if isinstance(child, ast.Assign):
                            for target in child.targets:
                                if isinstance(target, ast.Name):
                                    if target.id == "GnomeDesktop":
                                        found_fallback = True
                # Check if the try body imports GnomeDesktop
                for child in ast.walk(node):
                    if isinstance(child, ast.ImportFrom):
                        for alias in child.names:
                            if alias.name == "GnomeDesktop":
                                found_guarded_import = True

        self.assertTrue(found_guarded_import, "GnomeDesktop import should be inside a try block")
        self.assertTrue(found_fallback, "GnomeDesktop import guard should set GnomeDesktop = None in except")
