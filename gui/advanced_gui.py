from .gui_core import BaseImageProcessorGUI # MODIFIED: Relative import
from tkinter import ttk, messagebox
import tkinter as tk
import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
# These core imports assume 'image_processing_toolkit-sidrah' is in PYTHONPATH
# or you are running main.py from the root directory.
from core.filters import apply_filter
from core.morphology import apply_morph_operation
from core.features import apply_feature_detection
import os

class AdvancedImageProcessorGUI(BaseImageProcessorGUI):
    def __init__(self, root):
        super().__init__(root)
        self.create_advanced_controls()
        
    def create_advanced_controls(self):
        # Notebook for tabs
        self.notebook = ttk.Notebook(self.control_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Operations Tab
        self.ops_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.ops_tab, text="Operations")
        super().create_operation_controls() # Call base method if it exists (it does now)
        self.create_operation_controls() # Then add specific controls for this class
        
        # Parameters Tab
        self.param_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.param_tab, text="Parameters")
        self.create_parameter_controls()
        
        # Metrics Tab
        self.metrics_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.metrics_tab, text="Metrics")
        self.create_metrics_controls()
        
        # Histogram Button (moved from control_frame directly to be part of advanced controls)
        ttk.Button(self.ops_tab, text="Show Histogram", # Placed in ops_tab for better organization
                  command=self.show_histogram).grid(row=3, columnspan=2, pady=10, sticky='ew')
    
    def create_operation_controls(self): # This overrides the base method
        # Operation Type Selection
        ttk.Label(self.ops_tab, text="Operation Type:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.op_type = tk.StringVar(value="filter")
        self.op_type_combo = ttk.Combobox(self.ops_tab, textvariable=self.op_type,
                                        values=["filter", "morphology", "features"]) # "transform" renamed to "features" for clarity
        self.op_type_combo.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        
        # Operation Selection
        ttk.Label(self.ops_tab, text="Operation:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.operation = tk.StringVar()
        self.op_combo = ttk.Combobox(self.ops_tab, textvariable=self.operation, state="readonly")
        self.op_combo.grid(row=1, column=1, padx=5, pady=5, sticky='ew')
        
        # Update operations when type changes
        def update_operations(*args):
            op_type_val = self.op_type.get()
            if op_type_val == "filter":
                self.op_combo['values'] = ["blur", "gaussian", "median", "bilateral", "sobel", "laplacian", "canny", "threshold", "adaptive_threshold", "equalize", "color_hist_equal", "color_sepia"]
            elif op_type_val == "morphology":
                self.op_combo['values'] = ["erosion", "dilation", "opening", "closing", "gradient", "tophat", "blackhat"]
            elif op_type_val == "features": # Maps to feature detection
                self.op_combo['values'] = ["harris_corners", "shi_tomasi", "sift", "surf", "orb", "blob", "contours"]
            else:
                self.op_combo['values'] = []
            
            if self.op_combo['values']:
                self.operation.set(self.op_combo['values'][0]) # Set to first option
            else:
                self.operation.set("") # Clear selection
        
        self.op_type.trace_add('write', update_operations)
        update_operations() # Initial call
        
        # Process Button
        ttk.Button(self.ops_tab, text="Process Image", 
                  command=self.process_image).grid(row=2, columnspan=2, pady=10, sticky='ew') # process_image defined below
    
    def create_parameter_controls(self):
        # Kernel Size (common for many ops)
        ttk.Label(self.param_tab, text="Kernel Size:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.kernel_size = tk.IntVar(value=3)
        # Kernel size usually odd, ensure spinbox reflects this if possible (e.g. by validation or increment)
        ttk.Spinbox(self.param_tab, from_=1, to=31, increment=2, textvariable=self.kernel_size).grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        
        # Iterations (common for morphology)
        ttk.Label(self.param_tab, text="Iterations:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.iterations = tk.IntVar(value=1)
        ttk.Spinbox(self.param_tab, from_=1, to=10, textvariable=self.iterations).grid(row=1, column=1, padx=5, pady=5, sticky='ew')
        
        # Threshold (common for Canny, Harris, some filters)
        ttk.Label(self.param_tab, text="Threshold/T1:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.threshold1_val = tk.DoubleVar(value=100) # For Canny T1, or general threshold
        ttk.Scale(self.param_tab, from_=0, to=255, variable=self.threshold1_val, orient=tk.HORIZONTAL).grid(row=2, column=1, padx=5, pady=5, sticky='ew')
        ttk.Label(self.param_tab, textvariable=self.threshold1_val).grid(row=2, column=2, padx=5, pady=5, sticky='w')


        # Threshold2 (common for Canny)
        ttk.Label(self.param_tab, text="Threshold2 (Canny):").grid(row=3, column=0, padx=5, pady=5, sticky='w')
        self.threshold2_val = tk.DoubleVar(value=200) # For Canny T2
        ttk.Scale(self.param_tab, from_=0, to=255, variable=self.threshold2_val, orient=tk.HORIZONTAL).grid(row=3, column=1, padx=5, pady=5, sticky='ew')
        ttk.Label(self.param_tab, textvariable=self.threshold2_val).grid(row=3, column=2, padx=5, pady=5, sticky='w')
    
    def create_metrics_controls(self):
        self.metrics_text = tk.Text(self.metrics_tab, height=10, state='disabled', wrap=tk.WORD)
        self.metrics_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Button(self.metrics_tab, text="Clear Metrics",
                  command=self.clear_metrics).pack(pady=5, fill=tk.X)
    
    def clear_metrics(self):
        self.metrics_text.config(state='normal')
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.config(state='disabled')
    
    def update_metrics(self, metrics):
        self.metrics_text.config(state='normal')
        self.metrics_text.delete(1.0, tk.END)
        if metrics:
            for k, v in metrics.items():
                self.metrics_text.insert(tk.END, f"{k.upper()}: {v:.4f}\n" if isinstance(v, float) else f"{k.upper()}: {v}\n")
        else:
            self.metrics_text.insert(tk.END, "No metrics available for this operation.\n")
        self.metrics_text.config(state='disabled')
    
    def show_histogram(self):
        if self.original_input_image is None: # Use the stored original image for histogram
            messagebox.showwarning("Warning", "No image loaded to show histogram for.")
            return
            
        hist_window = tk.Toplevel(self.root)
        hist_window.title("Image Histogram (Input Image)")
        
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        current_display_image = self.original_input_image # Use the loaded image for histogram

        if len(current_display_image.shape) == 2 or current_display_image.shape[2] == 1:  # Grayscale
            img_gray = current_display_image
            if len(current_display_image.shape) == 3 and current_display_image.shape[2] == 1:
                img_gray = current_display_image[:,:,0]
            ax.hist(img_gray.ravel(), 256, [0, 256], color='gray')
            ax.set_title('Grayscale Histogram')
        else:  # Color
            colors = ('b', 'g', 'r')
            chans = cv2.split(current_display_image) # Split into B, G, R
            for i, col in enumerate(colors):
                hist = cv2.calcHist([chans[i]], [0], None, [256], [0, 256])
                ax.plot(hist, color=col)
            ax.set_title('Color Histogram')
            ax.legend(['Blue', 'Green', 'Red'])
        
        ax.set_xlim([0, 256])
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=hist_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def process_image(self): # This overrides the base method
        try:
            if self.current_image is None:
                raise ValueError("No image loaded")
            
            op_type_val = self.op_type.get()
            operation_val = self.operation.get()
            
            if not operation_val:
                raise ValueError("No operation selected")
            
            self.status_var.set(f"Processing: {operation_val}...")
            self.root.update_idletasks()
            
            # Create temp directory if it doesn't exist (safer)
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Use a unique name for temp input to avoid conflicts
            base_name, ext = os.path.splitext(os.path.basename("temp_input.png")) # Use png to preserve quality
            temp_input_path = os.path.join(temp_dir, f"{base_name}_{np.random.randint(10000, 99999)}{ext}")
            
            cv2.imwrite(temp_input_path, self.current_image)
            
            result_img = None
            metrics = {}
            output_file_path = None # To store the path of the processed image from core functions

            kernel_val = self.kernel_size.get()
            iter_val = self.iterations.get()
            thresh1_val = self.threshold1_val.get()
            thresh2_val = self.threshold2_val.get()

            if op_type_val == "filter":
                # Pass relevant parameters to apply_filter
                filter_params = {'kernel_size': kernel_val} # Default
                if operation_val == "canny":
                    filter_params.update({'threshold1': int(thresh1_val), 'threshold2': int(thresh2_val)})
                elif operation_val in ["threshold", "adaptive_threshold"]:
                     filter_params.update({'threshold_value': int(thresh1_val)}) # Example, adapt your core function
                # Add more specific param handling if your core.filters.apply_filter expects them by name
                
                output_file_path, metrics = apply_filter(temp_input_path, operation_val, **filter_params)
                if output_file_path: result_img = cv2.imread(output_file_path)

            elif op_type_val == "morphology":
                output_file_path = apply_morph_operation(
                    temp_input_path, operation_val,
                    kernel_size=kernel_val, iterations=iter_val
                )
                if output_file_path: result_img = cv2.imread(output_file_path)
                # metrics = {} # Morphology ops in your example didn't return metrics

            elif op_type_val == "features":
                feature_params = {}
                if operation_val == "harris_corners":
                    feature_params.update({'threshold': thresh1_val / 25500.0}) # Example scaling for Harris threshold
                
                output_file_path = apply_feature_detection(
                    temp_input_path, operation_val, **feature_params
                )
                if output_file_path: result_img = cv2.imread(output_file_path)
                # metrics = {} # Feature ops in your example didn't return metrics
            
            if result_img is None:
                raise ValueError(f"Operation '{operation_val}' failed or did not produce an image.")

            self.display_image(result_img, self.output_canvas, "output")
            self.update_metrics(metrics)
            self.status_var.set(f"Completed: {operation_val}")

            # Clean up temporary files
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
            if output_file_path and os.path.exists(output_file_path) and temp_dir in output_file_path:
                # Only delete if it's confirmed to be a temp output file
                os.remove(output_file_path)
            
        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred during processing '{operation_val}':\n{str(e)}")
            self.status_var.set(f"Error processing: {operation_val}")
            import traceback
            print(f"Processing Error: {operation_val}\n{traceback.format_exc()}") # Print full traceback for debugging
            
# This __main__ block is for testing advanced_gui.py directly.
# To run this: `python -m gui.advanced_gui` from the project root directory.
if __name__ == "__main__":
    # Need to add project root to sys.path for core imports to work when run directly
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    root = tk.Tk()
    app = AdvancedImageProcessorGUI(root)
    root.mainloop()