"""Step 4: Generate all figures and tables."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Generating figures...")
from src.plotting.figure_generator import generate_all_figures
generate_all_figures()
print("Figures done.")

print("Generating tables...")
from src.tables.table_generator import generate_all_tables
generate_all_tables()
print("Tables done.")

print("All outputs generated successfully!")
