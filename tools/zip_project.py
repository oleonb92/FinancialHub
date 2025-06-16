import zipfile
import os

def zip_project(output_filename="FinancialHub.zip", exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = [".venv", "__pycache__", ".git", ".DS_Store", "data", "venv"]

    base_dir = os.getcwd()
    with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(base_dir):
            # Excluir carpetas no deseadas
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                if any(x in root for x in exclude_dirs):
                    continue
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, base_dir)
                zipf.write(filepath, arcname)
    
    print(f"✅ Proyecto comprimido exitosamente como: {output_filename}")

if __name__ == "__main__":
    zip_project()