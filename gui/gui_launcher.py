import tkinter as tk
from .advanced_gui import AdvancedImageProcessorGUI # MODIFIED: Relative import

def main():
    root = tk.Tk()
    # The title is usually set within the BaseImageProcessorGUI or its subclasses
    app = AdvancedImageProcessorGUI(root)
    root.mainloop()

# This __main__ block allows running the GUI launcher directly for testing.
# To run this: `python -m gui.gui_launcher` from the project root directory.
if __name__ == '__main__':
    # Need to add project root to sys.path for core imports (indirectly via advanced_gui) to work
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    main()