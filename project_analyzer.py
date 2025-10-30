import os
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass

@dataclass
class FunctionInfo:
    """Informacje o funkcji"""
    name: str
    args: List[str]
    returns: str = None
    docstring: str = None
    
@dataclass
class FileInfo:
    """Informacje o pliku"""
    functions: List[FunctionInfo]
    imports: Set[str]
    imported_by: Set[str] = None

def get_function_info(node: ast.FunctionDef) -> FunctionInfo:
    """
    Ekstrahuje szczegółowe informacje o funkcji z AST node.
    """
    # Zbieramy argumenty
    args = []
    for arg in node.args.args:
        if hasattr(arg, 'annotation') and arg.annotation:
            if isinstance(arg.annotation, ast.Name):
                args.append(f"{arg.arg}: {arg.annotation.id}")
            elif isinstance(arg.annotation, ast.Subscript):
                # Dla typów złożonych jak List[str]
                args.append(f"{arg.arg}: {ast.unparse(arg.annotation)}")
        else:
            args.append(arg.arg)
            
    # Sprawdzamy typ zwracany
    returns = None
    if node.returns:
        returns = ast.unparse(node.returns)
        
    # Pobieramy docstring jeśli istnieje
    docstring = ast.get_docstring(node)
    
    return FunctionInfo(
        name=node.name,
        args=args,
        returns=returns,
        docstring=docstring
    )

def get_imports(tree: ast.AST) -> Set[str]:
    """
    Analizuje importy w pliku.
    """
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                imports.add(name.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for name in node.names:
                    imports.add(f"{node.module}.{name.name}")
    return imports

def analyze_file(file_path: str) -> FileInfo:
    """
    Analizuje pojedynczy plik Python.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            tree = ast.parse(file.read())
        
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(get_function_info(node))
            elif isinstance(node, ast.ClassDef):
                # Dodaj metody klas
                for class_node in ast.walk(node):
                    if isinstance(class_node, ast.FunctionDef):
                        func_info = get_function_info(class_node)
                        func_info.name = f"{node.name}.{func_info.name}"
                        functions.append(func_info)
        
        imports = get_imports(tree)
        return FileInfo(functions=sorted(functions, key=lambda x: x.name), imports=imports)
    
    except Exception as e:
        print(f"Error analyzing file {file_path}: {str(e)}")
        return FileInfo(functions=[], imports=set())

def analyze_project_structure(start_path: str) -> Dict[str, FileInfo]:
    """
    Analizuje strukturę całego projektu.
    """
    project_structure = {}
    
    # Najpierw zbieramy wszystkie informacje o plikach
    for root, dirs, files in os.walk(start_path):
        # Pomijamy katalogi systemowe
        dirs[:] = [d for d in dirs if d not in ['__pycache__', 'venv', '.git', '.idea']]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, start_path)
                project_structure[relative_path] = analyze_file(file_path)
    
    # Następnie analizujemy zależności między plikami
    for file_path, file_info in project_structure.items():
        file_info.imported_by = set()
        module_name = os.path.splitext(file_path)[0].replace(os.sep, '.')
        
        # Sprawdzamy, które pliki importują ten moduł
        for other_path, other_info in project_structure.items():
            if any(imp.startswith(module_name) for imp in other_info.imports):
                file_info.imported_by.add(other_path)
    
    return project_structure

def print_project_structure(structure: Dict[str, FileInfo], indent: str = '  ') -> None:
    """
    Wyświetla strukturę projektu w czytelnym formacie.
    """
    print("\nStruktura projektu:\n")
    
    # Grupowanie plików według katalogów
    directories = {}
    for file_path, file_info in structure.items():
        dir_path = os.path.dirname(file_path)
        if dir_path not in directories:
            directories[dir_path] = {}
        directories[dir_path][os.path.basename(file_path)] = file_info
    
    # Wyświetlanie struktury
    for dir_path in sorted(directories.keys()):
        if dir_path:
            print(f"{indent}📁 {dir_path}/")
        else:
            print("📁 ./")
        
        for file_name, file_info in sorted(directories[dir_path].items()):
            print(f"{indent * 2}📄 {file_name}")
            
            # Wyświetl importy
            if file_info.imports:
                print(f"{indent * 3}📥 Imports:")
                for imp in sorted(file_info.imports):
                    print(f"{indent * 4}- {imp}")
            
            # Wyświetl "imported by"
            if file_info.imported_by:
                print(f"{indent * 3}📤 Imported by:")
                for imp in sorted(file_info.imported_by):
                    print(f"{indent * 4}- {imp}")
            
            # Wyświetl funkcje
            if file_info.functions:
                print(f"{indent * 3}🔧 Functions:")
                for func in file_info.functions:
                    # Wyświetl nazwę funkcji i argumenty
                    args_str = ", ".join(func.args)
                    func_str = f"{func.name}({args_str})"
                    if func.returns:
                        func_str += f" -> {func.returns}"
                    print(f"{indent * 4}➤ {func_str}")
                    
                    # Wyświetl docstring jeśli istnieje
                    if func.docstring:
                        doc_lines = func.docstring.strip().split('\n')
                        print(f"{indent * 5}📝 {doc_lines[0]}")
                        for line in doc_lines[1:]:
                            print(f"{indent * 5}   {line.strip()}")
            else:
                print(f"{indent * 3}(brak funkcji)")
            print()

def main():
    """
    Główna funkcja programu.
    """
    start_path = '.'
    print(f"Analizuję projekt w katalogu: {os.path.abspath(start_path)}")
    structure = analyze_project_structure(start_path)
    print_project_structure(structure)

if __name__ == "__main__":
    main()
