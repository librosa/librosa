#!/usr/bin/env python
"""Scan module docstrings for usage of functions relying on a specific parameter (sr)"""
# This script exists primarily in service of issue #1708, and is designed to help
# us identify docstring example calls which depend on librosa functions accepting the sr
# parameter.
#
# It is then up to the user of this script to audit the docstring in question to determine
# if any correction is necessary.
#
# Script co-authored by chatgpt v4 and Brian McFee

import pkgutil
import inspect
import ast
import numpydoc.docscrape as nds

import librosa


def find_functions_with_param(package, param_name):
    """
    Scan all functions in a package and its submodules to find functions that include a given parameter.

    Parameters:
    - package: The package to scan. Should be the actual package/module object, not a string.
    - param_name: The name of the parameter to look for.

    Returns:
    - A set of strings, each containing the module path and function name where the parameter is found.
    """
    functions_with_param = []

    def check_function(module, func):
        """Check if a function has the given parameter and add it to the list if it does."""
        sig = inspect.signature(func)
        if param_name in sig.parameters:
            functions_with_param.append(f"{module.__name__}.{func.__name__}")

    def visit_module(module):
        """Visit a module, check its functions, and recursively visit its submodules."""
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj):
                check_function(module, obj)
            elif inspect.ismodule(obj) and obj.__package__ == module.__package__:
                # Recursively visit submodules, but only within the same package
                visit_module(obj)

    # Start the scanning process with the root package
    visit_module(package)

    return set(functions_with_param)


def find_functions(package):
    """
    Scan all functions in a package and its submodules to find functions.

    Parameters:
    - package: The package to scan. Should be the actual package/module object, not a string.

    Returns:
    - A set of functions in the package.
    """
    functions = []

    def visit_module(module):
        """Visit a module, check its functions, and recursively visit its submodules."""
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj):
                functions.append(obj)
            elif inspect.ismodule(obj) and obj.__package__ == module.__package__:
                # Recursively visit submodules, but only within the same package
                visit_module(obj)

    # Start the scanning process with the root package
    visit_module(package)

    return set(functions)


class LibrosaCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.librosa_calls = []

    def visit_Call(self, node):
        # Recursively check the call node to see if it's a librosa call
        if self.is_librosa_call(node.func):
            # Extract the full function name (including submodules) and arguments
            func_name = self.get_full_func_name(node.func)
            args = [ast.unparse(arg) for arg in node.args]  # Converts arguments back to Python code snippets
            kwargs = [arg.arg for arg in node.keywords]  # Converts arguments back to Python code snippets
            self.librosa_calls.append((func_name, args + kwargs))
        # Continue traversing the tree
        self.generic_visit(node)

    def is_librosa_call(self, node):
        # If the node represents an attribute access (e.g., librosa.feature.mfcc)
        if isinstance(node, ast.Attribute):
            return self.is_librosa_call(node.value)
        # If the node is a name and it's 'librosa', we've found a librosa call
        elif isinstance(node, ast.Name):
            return node.id == 'librosa'
        return False

    def get_full_func_name(self, node):
        # Collect parts of the function name (including submodules)
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return '.'.join(reversed(parts))


def extract_librosa_calls(code_blocks, target_functions, desc=""):
    for i, block in enumerate(code_blocks, start=1):
        # Parse the code block into an AST
        tree = ast.parse(block)

        # Create a visitor instance and visit the nodes
        visitor = LibrosaCallVisitor()
        visitor.visit(tree)

        # Print the librosa function calls found in this block
        if visitor.librosa_calls:
            offending = False
            for func_name, args in visitor.librosa_calls:
                if func_name in target_functions and "sr" not in args and func_name != "librosa.load":
                    offending = True
            if not offending:
                return
            print(f"{desc} code block {i} has the following librosa calls:")
            for func_name, args in visitor.librosa_calls:
                if func_name in target_functions and "sr" not in args and func_name != "librosa.load":
                    print(f"  - {func_name}({', '.join(args)})")
            print()



def extract_code_blocks(docstring):
    # Parse the docstring
    parsed_doc = nds.NumpyDocString(docstring)

    code_blocks = []

    # Extract the "Examples" section (which is a list)
    examples = parsed_doc['Examples']
    
    current_block = []

    for line in examples:
        stripped_line = line.lstrip()  # Strip leading whitespace

        if stripped_line.startswith('>>>') or stripped_line.startswith('...'):
            # Remove '>>>' or '...' from the start, but preserve indentation for continuation lines
            code_line = stripped_line[4:] if stripped_line.startswith('>>>') else stripped_line[3:]
            current_block.append(code_line)
        elif current_block:
            # If we hit a non-code line and there's an ongoing block, end and save the current block
            code_blocks.append('\n'.join(current_block))
            current_block = []

    # Add the last block if it exists
    if current_block:
        code_blocks.append('\n'.join(current_block))

    return code_blocks


if __name__ == '__main__':
    target_functions = find_functions_with_param(librosa, "sr")

    # Extract code blocks from the function's docstring
    all_functions = find_functions(librosa)

    for function in all_functions:
        if function.__doc__ is None:
            continue
        try:
            code_blocks = extract_code_blocks(function.__doc__)
            extract_librosa_calls(code_blocks, target_functions, desc=function.__name__)
        except:
            print(f"Could not parse {function.__name__}, skipping.  CHECK MANUALLY.")
