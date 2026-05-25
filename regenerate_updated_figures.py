"""Regenerate updated figures only."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from src.plotting.figure_generator import fig_system_architecture, fig_accuracy_curves

print("Regenerating system architecture diagram...")
fig_system_architecture()

print("\nRegenerating accuracy curves...")
fig_accuracy_curves()

print("\nDone.")
