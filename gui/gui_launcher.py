import tkinter as tk
from .advanced_gui import AdvancedImageProcessorGUI

def main():
    root = tk.Tk()
    app = AdvancedImageProcessorGUI(root)
    root.mainloop()

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    main()
