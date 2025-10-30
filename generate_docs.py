import os
import ast  # Abstract Syntax Trees - do analizy kodu bez jego wykonywania
import argparse

def analyze_file(filepath):
    """Analizuje pojedynczy plik Python i zwraca informacje."""
    print(f"Analizuję: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            tree = ast.parse(content)
    except Exception as e:
        print(f"  Błąd podczas parsowania {filepath}: {e}")
        return None

    module_docstring = ast.get_docstring(tree)
    items = []
    imports = set()

    for node in ast.walk(tree):
        # Wyciąganie importów wewnątrz projektu (prosta heurystyka)
        if isinstance(node, ast.ImportFrom):
            if node.module and not node.level: # Pomijamy importy relatywne na razie dla uproszczenia
                 # Tutaj można dodać logikę sprawdzającą, czy 'node.module' to plik z projektu
                 imports.add(node.module)
        elif isinstance(node, ast.Import):
             for alias in node.names:
                 # Tutaj można dodać logikę sprawdzającą, czy 'alias.name' to plik z projektu
                 imports.add(alias.name)

        # Wyciąganie klas i funkcji z docstringami
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            docstring = ast.get_docstring(node)
            item_type = type(node).__name__.replace("Def", "") # Function, AsyncFunction, Class
            items.append({
                "type": item_type,
                "name": node.name,
                "docstring": docstring.strip() if docstring else "Brak docstringu."
            })

    return {
        "filepath": filepath,
        "module_docstring": module_docstring.strip() if module_docstring else "Brak docstringu modułu.",
        "imports": sorted(list(imports)), # Podstawowe zależności
        "items": items
    }

def generate_documentation(project_root, output_file):
    """Generuje dokumentację dla projektu w pliku Markdown."""
    all_data = []
    for root, _, files in os.walk(project_root):
        # Proste pomijanie folderów wirtualnego środowiska
        if 'venv' in root or 'env' in root or '__pycache__' in root:
             continue
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                file_data = analyze_file(filepath)
                if file_data:
                    all_data.append(file_data)

    # Sortowanie plików dla lepszej czytelności (opcjonalne)
    all_data.sort(key=lambda x: x['filepath'])

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Dokumentacja Projektu Bota Tradingowego\n\n")
        f.write("## Struktura Plików i Główne Komponenty\n\n")

        for data in all_data:
            relative_path = os.path.relpath(data['filepath'], project_root)
            f.write(f"### Plik: `{relative_path}`\n\n")
            f.write(f"**Opis modułu:**\n```\n{data['module_docstring']}\n```\n\n")

            if data.get('imports'):
                 f.write(f"**Podstawowe importy (potencjalne zależności):**\n")
                 for imp in data['imports']:
                      f.write(f"- `{imp}`\n")
                 f.write("\n")


            if data['items']:
                f.write("**Główne klasy/funkcje:**\n\n")
                for item in data['items']:
                    f.write(f"- **`{item['name']}`** ({item['type']})\n")
                    f.write(f"  ```\n  {item['docstring']}\n  ```\n")
            else:
                f.write("*Brak zdefiniowanych głównych klas/funkcji w tym pliku.*\n")
            f.write("\n---\n\n")

    print(f"Dokumentacja wygenerowana w: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generuje podstawową dokumentację z kodu Python.")
    parser.add_argument("project_root", help="Ścieżka do głównego katalogu projektu.")
    parser.add_argument("-o", "--output", default="project_docs.md", help="Nazwa pliku wyjściowego Markdown.")
    args = parser.parse_args()

    generate_documentation(args.project_root, args.output)