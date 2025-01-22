import json

# Load the notebook
with open("DA0101EN-Review-Introduction-20231003-1696291200.jupyterlite.ipynb", "r") as f:
    notebook = json.load(f)

# Extract Markdown cells
markdown_cells = [cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "markdown"]

# Write to a Markdown file
with open("output.md", "w") as f:
    for cell in markdown_cells:
        f.write("".join(cell) + "\n\n")

