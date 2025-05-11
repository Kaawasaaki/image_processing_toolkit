import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

class BaseImageProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.current_image = None
        self.tk_img = None # Important to keep a reference
        self.output_tk_img = None # Separate reference for output canvas image
        self.setup_window()
        self.create_widgets()
        self.setup_bindings()
        
    def setup_window(self):
        self.root.title("Image Processor")
        self.root.geometry("1200x800")
        self.style = ttk.Style()
        self.style.configure('TButton', font=('Helvetica', 10))
        self.style.configure('TFrame', padding=10)
        
    def create_widgets(self):
        # Main Paned Window
        self.main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_pane.pack(fill=tk.BOTH, expand=True)
        
        # Control Panel (Left)
        self.control_frame = ttk.Frame(self.main_pane, width=300) # Initial width suggestion
        self.main_pane.add(self.control_frame, weight=0) # Weight 0 means it won't expand initially
        
        # Image Display (Right)
        self.image_frame = ttk.Frame(self.main_pane)
        self.main_pane.add(self.image_frame, weight=1) # Weight 1 means it will expand
        
        # Common Widgets
        ttk.Label(self.control_frame, text="Image Processor", 
                font=('Helvetica', 14, 'bold')).pack(pady=10)
        
        self.load_btn = ttk.Button(self.control_frame, text="Load Image", 
                                 command=self.load_image)
        self.load_btn.pack(pady=10, fill=tk.X)
        
        # Image Canvases
        self.canvas_frame = ttk.Frame(self.image_frame)
        self.canvas_frame.pack(expand=True, fill=tk.BOTH)
        
        # Use a paned window for canvases to allow resizing
        self.canvas_pane = ttk.PanedWindow(self.canvas_frame, orient=tk.HORIZONTAL)
        self.canvas_pane.pack(fill=tk.BOTH, expand=True)

        self.input_canvas_frame = ttk.Frame(self.canvas_pane, relief=tk.SUNKEN, borderwidth=1)
        self.input_canvas = tk.Canvas(self.input_canvas_frame, bg='lightgrey') # Changed bg for visibility
        self.input_canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas_pane.add(self.input_canvas_frame, weight=1)
        
        self.output_canvas_frame = ttk.Frame(self.canvas_pane, relief=tk.SUNKEN, borderwidth=1)
        self.output_canvas = tk.Canvas(self.output_canvas_frame, bg='lightgrey') # Changed bg for visibility
        self.output_canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas_pane.add(self.output_canvas_frame, weight=1)
        
        # Status Bar
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status_var, 
                relief=tk.SUNKEN, anchor=tk.W, padding=(5,2)).pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_bindings(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        # Bind resize event to redraw images correctly scaled
        self.input_canvas.bind("<Configure>", lambda e, c=self.input_canvas, i_type="input": self.on_canvas_resize(e, c, i_type))
        self.output_canvas.bind("<Configure>", lambda e, c=self.output_canvas, i_type="output": self.on_canvas_resize(e, c, i_type))
        self.original_input_image = None # Store unscaled input image
        self.original_output_image = None # Store unscaled output image

    def on_canvas_resize(self, event, canvas, image_type):
        if image_type == "input" and self.original_input_image is not None:
            self._display_image_on_canvas(self.original_input_image, canvas, "tk_img")
        elif image_type == "output" and self.original_output_image is not None:
            self._display_image_on_canvas(self.original_output_image, canvas, "output_tk_img")

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
            ("All files", "*.*")
        ])
        if path:
            try:
                self.current_image = cv2.imread(path)
                if self.current_image is None:
                    # Try with PIL if cv2 fails for some formats, then convert
                    try:
                        img_pil_load = Image.open(path)
                        self.current_image = cv2.cvtColor(np.array(img_pil_load), cv2.COLOR_RGB2BGR if img_pil_load.mode == 'RGB' else cv2.COLOR_RGBA2BGRA)
                    except Exception as pil_e:
                         raise ValueError(f"Invalid image file. OpenCV error. PIL error: {pil_e}")

                if self.current_image is None:
                    raise ValueError("Invalid image file or format not supported.")

                self.original_input_image = self.current_image.copy()
                self.original_output_image = None # Clear previous output
                self.output_canvas.delete("all") # Clear output canvas

                # Call display_image, which now handles storing the PhotoImage reference correctly
                self.display_image(self.original_input_image, self.input_canvas, "input")
                self.status_var.set(f"Loaded: {os.path.basename(path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
                self.status_var.set("Failed to load image")
    
    def _display_image_on_canvas(self, cv_img, canvas, tk_image_attr_name):
        """Internal helper to display CV image on a canvas, managing PhotoImage lifecycle."""
        if cv_img is None:
            canvas.delete("all")
            setattr(self, tk_image_attr_name, None) # Clear the image reference
            return

        try:
            if len(cv_img.shape) == 2:  # Grayscale
                img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
            elif cv_img.shape[2] == 4: # BGR-Alpha
                img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGB) # Convert to RGB for PIL
            else:  # Color (BGR)
                img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            
            img_pil = Image.fromarray(img_rgb)
            
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()

            # If canvas is not yet drawn, its size might be 1x1. Use a default or skip.
            if canvas_width <= 1 or canvas_height <= 1:
                 # This can happen during initial layout. We can either
                 # 1. Postpone drawing (e.g. canvas.after(10, lambda: self._display_image_on_canvas(...)) )
                 # 2. Or use a reasonable default thumbnail size for the very first draw
                 # For now, let's ensure it's at least a small size to avoid division by zero or tiny images.
                 target_w, target_h = 500, 400 # Default if canvas not ready
            else:
                target_w, target_h = canvas_width, canvas_height

            img_pil.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
            
            # Store the PhotoImage as an attribute of self to prevent garbage collection
            photo_image = ImageTk.PhotoImage(img_pil)
            setattr(self, tk_image_attr_name, photo_image)
            
            canvas.delete("all")
            canvas.create_image(canvas_width//2, canvas_height//2, 
                              image=photo_image, anchor=tk.CENTER)
        except Exception as e:
            # messagebox.showerror("Display Error", f"Cannot display image on canvas:\n{str(e)}")
            print(f"Display Error on canvas {canvas}: {str(e)}") # Print to console to avoid too many popups
            self.status_var.set("Error displaying image")

    def display_image(self, cv_img, canvas, image_type="input"):
        """
        Displays an OpenCV image on the specified canvas.
        `image_type` can be "input" or "output" to use different PhotoImage attributes.
        """
        if image_type == "input":
            self.original_input_image = cv_img.copy() if cv_img is not None else None
            self._display_image_on_canvas(self.original_input_image, canvas, "tk_img")
        elif image_type == "output":
            self.original_output_image = cv_img.copy() if cv_img is not None else None
            self._display_image_on_canvas(self.original_output_image, canvas, "output_tk_img")
        else:
            raise ValueError("image_type must be 'input' or 'output'")

    def on_close(self):
        if messagebox.askokcancel("Quit", "Do you want to exit?"):
            self.root.destroy()

    def create_operation_controls(self):
        # This method should be overridden by subclasses
        pass # No longer raising NotImplementedError to allow base class to run if subclass doesn't override
    
    def process_image(self):
        # This method should be overridden by subclasses
        pass # No longer raising NotImp
