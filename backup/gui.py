"""
gui.py - Tkinter GUI Interface Module
=====================================
Giao diện đồ họa cho hệ thống đa camera
Bao gồm: camera controls, video display, parking status, system monitoring
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any
import logging
from PIL import Image, ImageTk
import cv2
import numpy as np

# Import modules
from .multi_camera_manager import MultiCameraManager

logger = logging.getLogger(__name__)

class MultiCameraParkingGUI:
    """
    Giao diện GUI chính cho hệ thống đa camera
    
    Components:
    - Camera control panel (start/stop/pause/resume)
    - Video display với real-time feed
    - Parking spots status monitoring
    - System statistics dashboard
    - Manual controls và settings
    """
    
    def __init__(self, manager: MultiCameraManager):
        """
        Khởi tạo GUI
        
        Args:
            manager (MultiCameraManager): Manager instance
        """
        self.manager = manager
        
        # Tkinter components
        self.root = tk.Tk()
        self.root.title("Multi-Camera Parking Management System v1.0")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # GUI state
        self.current_frame = None
        self.update_job = None
        self.is_updating = False
        
        # Update intervals (seconds)
        self.video_update_interval = 0.033  # ~30 FPS
        self.status_update_interval = 1.0   # 1 second
        self.parking_update_interval = 2.0  # 2 seconds
        
        # Threading locks
        self.gui_lock = threading.Lock()
        
        self._setup_styles()
        self._setup_gui()
        self._start_update_loops()
        
        logger.info("🎮 GUI initialized successfully")
    
    # ==================== SETUP METHODS ====================
    
    def _setup_styles(self):
        """Setup ttk styles for better appearance"""
        style = ttk.Style()
        
        # Configure button styles
        style.configure('Success.TButton', foreground='white')
        style.configure('Danger.TButton', foreground='white')
        style.configure('Warning.TButton', foreground='black')
        
        # Configure label styles
        style.configure('Title.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Status.TLabel', font=('Courier', 9))
    
    def _setup_gui(self):
        """Setup tất cả GUI components"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Create panels
        self._create_control_panel(main_frame)
        self._create_display_panel(main_frame)
        self._create_parking_panel(main_frame)
    
    # ==================== PANEL CREATION METHODS ====================
    
    def _create_control_panel(self, parent):
        """Tạo control panel bên trái"""
        control_frame = ttk.LabelFrame(parent, text="Camera Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        control_frame.columnconfigure(0, weight=1)
        
        # Camera listbox
        self._create_camera_listbox(control_frame)
        
        # Individual camera controls
        self._create_individual_controls(control_frame)
        
        # System controls
        self._create_system_controls(control_frame)
        
        # Status display
        self._create_status_display(control_frame)
        
        # Make control_frame expand
        control_frame.rowconfigure(12, weight=1)
    
    def _create_camera_listbox(self, parent):
        """Tạo camera listbox với scrollbar"""
        ttk.Label(parent, text="Available Cameras:", style='Title.TLabel').grid(
            row=0, column=0, sticky=tk.W, pady=(0, 5)
        )
        
        # Listbox với scrollbar
        listbox_frame = ttk.Frame(parent)
        listbox_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        listbox_frame.columnconfigure(0, weight=1)
        listbox_frame.rowconfigure(0, weight=1)
        
        self.camera_listbox = tk.Listbox(listbox_frame, height=8, font=('Arial', 9))
        self.camera_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.camera_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.camera_listbox.configure(yscrollcommand=scrollbar.set)
        
        # Bind selection event
        self.camera_listbox.bind('<<ListboxSelect>>', self._on_camera_select)
    
    def _create_individual_controls(self, parent):
        """Tạo individual camera control buttons"""
        # Start/Stop buttons
        btn_frame1 = ttk.Frame(parent)
        btn_frame1.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        btn_frame1.columnconfigure(0, weight=1)
        btn_frame1.columnconfigure(1, weight=1)
        
        ttk.Button(btn_frame1, text="▶ Start", command=self._start_selected_camera, 
                  style='Success.TButton').grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 2))
        ttk.Button(btn_frame1, text="⏹ Stop", command=self._stop_selected_camera,
                  style='Danger.TButton').grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(2, 0))
        
        # Pause/Resume buttons
        btn_frame2 = ttk.Frame(parent)
        btn_frame2.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        btn_frame2.columnconfigure(0, weight=1)
        btn_frame2.columnconfigure(1, weight=1)
        
        ttk.Button(btn_frame2, text="⏸ Pause", command=self._pause_selected_camera,
                  style='Warning.TButton').grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 2))
        ttk.Button(btn_frame2, text="▶ Resume", command=self._resume_selected_camera,
                  style='Success.TButton').grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(2, 0))
    
    def _create_system_controls(self, parent):
        """Tạo system control buttons"""
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(row=4, column=0, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(parent, text="System Controls:", style='Title.TLabel').grid(
            row=5, column=0, sticky=tk.W, pady=(0, 5)
        )
        
        system_buttons = [
            ("🚀 Start All Cameras", self._start_all_cameras, 6),
            ("🛑 Stop All Cameras", self._stop_all_cameras, 7),
            ("🔄 Restart System", self._restart_system, 8),
            ("🧹 Cleanup Resources", self._cleanup_resources, 9),
        ]
        
        for text, command, row in system_buttons:
            ttk.Button(parent, text=text, command=command).grid(
                row=row, column=0, sticky=(tk.W, tk.E), pady=2
            )
    
    def _create_status_display(self, parent):
        """Tạo status display text widget"""
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(row=10, column=0, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(parent, text="System Status:", style='Title.TLabel').grid(
            row=11, column=0, sticky=tk.W, pady=(0, 5)
        )
        
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=12, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)
        
        self.status_text = tk.Text(status_frame, height=12, width=35, font=('Courier', 8), 
                                  wrap=tk.WORD, bg='#f8f8f8')
        self.status_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        status_scrollbar = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        status_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.status_text.configure(yscrollcommand=status_scrollbar.set)
    
    def _create_display_panel(self, parent):
        """Tạo video display panel bên phải"""
        display_frame = ttk.LabelFrame(parent, text="Camera View", padding="10")
        display_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(1, weight=1)
        
        # Camera info
        self._create_camera_info(display_frame)
        
        # Video canvas
        self._create_video_canvas(display_frame)
        
        # Display controls
        self._create_display_controls(display_frame)
    
    def _create_camera_info(self, parent):
        """Tạo camera information display"""
        self.info_frame = ttk.Frame(parent)
        self.info_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.info_frame.columnconfigure(1, weight=1)
        
        ttk.Label(self.info_frame, text="Camera:", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky=tk.W
        )
        self.info_label = ttk.Label(self.info_frame, text="No camera selected", font=('Arial', 10))
        self.info_label.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        
        # Performance info
        self.perf_label = ttk.Label(self.info_frame, text="", font=('Arial', 9), foreground='gray')
        self.perf_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(2, 0))
    
    def _create_video_canvas(self, parent):
        """Tạo video canvas với scrollbars"""
        canvas_frame = ttk.Frame(parent)
        canvas_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        self.canvas = tk.Canvas(canvas_frame, bg='black', width=800, height=600)
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Canvas scrollbars (if needed)
        canvas_h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        canvas_v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=canvas_h_scroll.set, yscrollcommand=canvas_v_scroll.set)
    
    def _create_display_controls(self, parent):
        """Tạo display control buttons"""
        control_info_frame = ttk.Frame(parent)
        control_info_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        control_info_frame.columnconfigure(2, weight=1)
        
        ttk.Button(control_info_frame, text="📊 Export Report", 
                  command=self._export_report).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(control_info_frame, text="📱 Switch View", 
                  command=self._switch_camera_dialog).grid(row=0, column=1, padx=(0, 5))
        
        # Status indicators
        self.connection_status = ttk.Label(control_info_frame, text="🔴 Offline", font=('Arial', 9))
        self.connection_status.grid(row=0, column=2, sticky=tk.E)
    
    def _create_parking_panel(self, parent):
        """Tạo parking status panel ở dưới"""
        parking_frame = ttk.LabelFrame(parent, text="Parking Spots Status", padding="10")
        parking_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        parking_frame.columnconfigure(0, weight=1)
        parking_frame.rowconfigure(0, weight=1)
        
        # Notebook cho multiple cameras
        self.spots_notebook = ttk.Notebook(parking_frame)
        self.spots_notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure notebook to expand
        parent.rowconfigure(1, weight=0, minsize=200)  # Fixed height for parking panel
    
    # ==================== UPDATE METHODS ====================
    
    def _start_update_loops(self):
        """Bắt đầu các update loops"""
        # Video update (high frequency)
        self._schedule_video_update()
        
        # Status update (medium frequency)
        self._schedule_status_update()
        
        # Parking update (low frequency)
        self._schedule_parking_update()
        
        logger.info("🔄 GUI update loops started")
    
    def _schedule_video_update(self):
        """Schedule video frame update"""
        self._update_video_display()
        # Schedule next update
        self.root.after(int(self.video_update_interval * 1000), self._schedule_video_update)
    
    def _schedule_status_update(self):
        """Schedule status info update"""
        self._update_camera_list()
        self._update_status_display()
        self._update_connection_status()
        # Schedule next update
        self.root.after(int(self.status_update_interval * 1000), self._schedule_status_update)
    
    def _schedule_parking_update(self):
        """Schedule parking spots update"""
        self._update_parking_spots()
        # Schedule next update
        self.root.after(int(self.parking_update_interval * 1000), self._schedule_parking_update)
    
    # ==================== GUI UPDATE METHODS ====================
    
    def _update_camera_list(self):
        """Update danh sách cameras trong listbox"""
        try:
            current_selection = None
            if self.camera_listbox.curselection():
                current_selection = self.camera_listbox.curselection()[0]
            
            self.camera_listbox.delete(0, tk.END)
            
            for i, (camera_id, config) in enumerate(self.manager.camera_configs.items()):
                # Status icons
                status_icon = "🟢" if camera_id in self.manager.active_cameras else "🔴"
                display_icon = "📺" if camera_id == self.manager.current_display_camera else "  "
                priority_icon = "⭐" * config.priority if config.priority <= 3 else "⭐⭐⭐"
                
                # Format text
                text = f"{display_icon} {status_icon} {config.name} {priority_icon}"
                self.camera_listbox.insert(tk.END, text)
                
                # Color coding
                if camera_id in self.manager.active_cameras:
                    self.camera_listbox.itemconfig(i, {'fg': 'darkgreen'})
                else:
                    self.camera_listbox.itemconfig(i, {'fg': 'darkred'})
            
            # Restore selection
            if current_selection is not None and current_selection < self.camera_listbox.size():
                self.camera_listbox.selection_set(current_selection)
                
        except Exception as e:
            logger.error(f"❌ Error updating camera list: {e}")
    
    def _update_status_display(self):
        """Update system status text"""
        try:
            system_status = self.manager.get_system_status()
            
            # Format status information
            uptime_hours = system_status['system_info']['uptime_seconds'] / 3600
            
            status_lines = [
                f"⏱️  Uptime: {uptime_hours:.1f}h",
                f"📹 Active: {system_status['system_info']['active_cameras']}/{system_status['system_info']['configured_cameras']}",
                f"📺 Display: {system_status['system_info']['current_display'] or 'None'}",
                f"🎯 Max Concurrent: {system_status['system_info']['max_concurrent']}",
                "",
                "📊 Aggregated Stats:",
                f"  📼 Frames: {system_status['aggregated_stats']['total_frames_processed']:,}",
                f"  📤 Events: {system_status['aggregated_stats']['total_events_generated']:,}",
                f"  🚗 Plates: {system_status['aggregated_stats']['total_plates_detected']:,}",
                f"  📊 Occupancy: {system_status['aggregated_stats']['occupancy_rate']}%",
                "",
                "💾 Resources:",
                f"  🧠 RAM: {system_status['resource_usage']['memory_usage_mb']:.0f} MB",
                f"  🖥️  GPU: {system_status['resource_usage']['gpu_memory_mb']:.0f} MB",
                f"  🧵 Threads: {system_status['resource_usage']['active_threads']}",
                "",
                "🔧 System:",
                f"  🚀 Starts: {system_status['system_stats']['cameras_started']}",
                f"  🛑 Stops: {system_status['system_stats']['cameras_stopped']}",
                f"  🔄 Restarts: {system_status['system_stats']['system_restarts']}",
                f"  🧹 Cleanups: {system_status['system_stats']['memory_cleanups']}"
            ]
            
            # Update text widget
            self.status_text.delete(1.0, tk.END)
            self.status_text.insert(1.0, "\n".join(status_lines))
            
        except Exception as e:
            logger.error(f"❌ Error updating status display: {e}")
    
    def _update_video_display(self):
        """Update video display với frame từ current camera"""
        try:
            if self.manager.current_display_camera:
                frame = self.manager.get_display_frame()
                if frame is not None:
                    self._display_frame(frame)
                    self._update_performance_info()
                else:
                    self._display_no_signal()
            else:
                self._display_no_camera()
                
        except Exception as e:
            logger.error(f"❌ Error updating video display: {e}")
    
    def _update_connection_status(self):
        """Update server connection status"""
        try:
            if self.manager.server_sync:
                status = self.manager.server_sync.get_connection_status()
                if status['is_connected']:
                    self.connection_status.config(text="🟢 Connected", foreground='green')
                else:
                    queue_size = status.get('offline_queue_size', 0)
                    self.connection_status.config(
                        text=f"🔴 Offline ({queue_size})", 
                        foreground='red'
                    )
            else:
                self.connection_status.config(text="⚪ Sync Disabled", foreground='gray')
                
        except Exception as e:
            logger.error(f"❌ Error updating connection status: {e}")
    
    def _update_parking_spots(self):
        """Update parking spots status display"""
        try:
            # Clear existing tabs
            for tab in self.spots_notebook.tabs():
                self.spots_notebook.forget(tab)
            
            # Get parking status từ tất cả cameras
            all_parking_status = self.manager.get_all_parking_status()
            
            if not all_parking_status:
                # No cameras running, show message
                empty_frame = ttk.Frame(self.spots_notebook)
                self.spots_notebook.add(empty_frame, text="No Active Cameras")
                
                ttk.Label(empty_frame, text="No cameras are currently running.", 
                         font=('Arial', 12)).pack(expand=True)
                return
            
            # Create tab cho mỗi camera
            for camera_id, camera_data in all_parking_status.items():
                tab_frame = ttk.Frame(self.spots_notebook)
                self.spots_notebook.add(tab_frame, text=camera_data['camera_name'])
                
                # Create treeview for spots
                self._create_parking_spots_tree(tab_frame, camera_data['spots'])
                
        except Exception as e:
            logger.error(f"❌ Error updating parking spots: {e}")
    
    def _update_performance_info(self):
        """Update performance information"""
        try:
            if self.manager.current_display_camera:
                camera_id = self.manager.current_display_camera
                config = self.manager.camera_configs.get(camera_id)
                stats = self.manager.get_camera_stats(camera_id)
                
                if config and stats:
                    # Update camera info
                    self.info_label.config(text=f"{config.name} ({camera_id})")
                    
                    # Update performance info
                    perf_text = (
                        f"FPS: {stats['fps']:.1f} | "
                        f"Frames: {stats['frames_processed']:,} | "
                        f"Events: {stats['events_generated']} | "
                        f"Occupied: {stats['occupied_spots']}/{stats['parking_spots_count']} | "
                        f"Errors: {stats['errors']}"
                    )
                    self.perf_label.config(text=perf_text)
                    
        except Exception as e:
            logger.error(f"❌ Error updating performance info: {e}")
    
    # ==================== DISPLAY METHODS ====================
    
    def _display_frame(self, frame):
        """Display frame trên canvas"""
        try:
            height, width = frame.shape[:2]
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                # Calculate scaling để fit canvas
                scale_w = canvas_width / width
                scale_h = canvas_height / height
                scale = min(scale_w, scale_h, 1.0)  # Don't upscale
                
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                # Resize frame
                if scale != 1.0:
                    frame_resized = cv2.resize(frame, (new_width, new_height))
                else:
                    frame_resized = frame
                
                # Convert BGR to RGB for PIL
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image)
                
                # Update canvas
                self.canvas.delete("all")
                x = (canvas_width - new_width) // 2
                y = (canvas_height - new_height) // 2
                self.canvas.create_image(x, y, anchor=tk.NW, image=photo)
                self.canvas.image = photo  # Keep reference to prevent garbage collection
                
        except Exception as e:
            logger.error(f"❌ Error displaying frame: {e}")
    
    def _display_no_signal(self):
        """Display "No Signal" message"""
        self.canvas.delete("all")
        self.canvas.create_text(
            self.canvas.winfo_width()//2, self.canvas.winfo_height()//2,
            text="📵 No Signal", fill="white", font=('Arial', 24)
        )
    
    def _display_no_camera(self):
        """Display "No Camera Selected" message"""
        self.canvas.delete("all")
        self.canvas.create_text(
            self.canvas.winfo_width()//2, self.canvas.winfo_height()//2,
            text="📹 No Camera Selected", fill="gray", font=('Arial', 18)
        )
    
    # ==================== PARKING SPOTS METHODS ====================
    
    def _create_parking_spots_tree(self, parent, spots_data):
        """Create treeview widget cho parking spots"""
        # Treeview frame với scrollbars
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        
        # Define columns
        columns = ('Spot', 'Status', 'Plate', 'Confidence', 'Enter Time', 'Duration', 'Updates')
        tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=8)
        
        # Configure columns
        column_widths = {'Spot': 80, 'Status': 100, 'Plate': 120, 'Confidence': 80, 
                        'Enter Time': 100, 'Duration': 80, 'Updates': 80}
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=column_widths.get(col, 100), anchor=tk.CENTER)
        
        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Grid layout
        tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Populate với spot data
        self._populate_parking_tree(tree, spots_data)
    
    def _populate_parking_tree(self, tree, spots_data):
        """Populate parking tree with spot data"""
        for spot in spots_data:
            status = "🚗 Occupied" if spot['is_occupied'] else "🅿️ Available"
            plate = spot['plate_text'] or "-"
            conf = f"{spot['plate_confidence']:.3f}" if spot['plate_confidence'] > 0 else "-"
            
            # Format enter time
            enter_time = "-"
            if spot['enter_time']:
                try:
                    dt = datetime.fromisoformat(spot['enter_time'])
                    enter_time = dt.strftime("%H:%M:%S")
                except:
                    pass
            
            # Calculate duration
            duration = self._calculate_duration(spot)
            
            # Insert row
            item = tree.insert('', tk.END, values=(
                spot['spot_name'],
                status,
                plate,
                conf,
                enter_time,
                duration,
                spot['detection_count']
            ))
            
            # Color coding
            if spot['is_occupied']:
                tree.set(item, 'Status', '🚗 Occupied')
            else:
                tree.set(item, 'Status', '🅿️ Available')
    
    def _calculate_duration(self, spot):
        """Calculate parking duration"""
        duration = "-"
        if spot['is_occupied'] and spot['enter_time']:
            try:
                enter_dt = datetime.fromisoformat(spot['enter_time'])
                duration_mins = int((datetime.now() - enter_dt).total_seconds() / 60)
                if duration_mins >= 60:
                    hours = duration_mins // 60
                    mins = duration_mins % 60
                    duration = f"{hours}h {mins}m"
                else:
                    duration = f"{duration_mins}m"
            except:
                pass
        return duration
    
    # ==================== EVENT HANDLERS ====================
    
    def _on_camera_select(self, event):
        """Handle camera selection trong listbox"""
        try:
            selection = self.camera_listbox.curselection()
            if selection:
                index = selection[0]
                camera_ids = list(self.manager.camera_configs.keys())
                if index < len(camera_ids):
                    camera_id = camera_ids[index]
                    if camera_id in self.manager.active_cameras:
                        self.manager.switch_display_camera(camera_id)
                        logger.info(f"🔄 GUI switched display to: {camera_id}")
                        
        except Exception as e:
            logger.error(f"❌ Error handling camera selection: {e}")
    
    def _get_selected_camera_id(self) -> Optional[str]:
        """Get camera ID của selection hiện tại"""
        try:
            selection = self.camera_listbox.curselection()
            if selection:
                index = selection[0]
                camera_ids = list(self.manager.camera_configs.keys())
                if index < len(camera_ids):
                    return camera_ids[index]
        except:
            pass
        return None
    
    # ==================== CONTROL BUTTON HANDLERS ====================
    
    def _start_selected_camera(self):
        """Start camera được select"""
        camera_id = self._get_selected_camera_id()
        if camera_id:
            success = self.manager.start_camera(camera_id)
            if success:
                messagebox.showinfo("Success", f"Started camera: {camera_id}")
            else:
                messagebox.showerror("Error", f"Failed to start camera: {camera_id}")
    
    def _stop_selected_camera(self):
        """Stop camera được select"""
        camera_id = self._get_selected_camera_id()
        if camera_id:
            success = self.manager.stop_camera(camera_id)
            if success:
                messagebox.showinfo("Success", f"Stopped camera: {camera_id}")
            else:
                messagebox.showerror("Error", f"Failed to stop camera: {camera_id}")
    
    def _pause_selected_camera(self):
        """Pause camera được select"""
        camera_id = self._get_selected_camera_id()
        if camera_id:
            success = self.manager.pause_camera(camera_id)
            if success:
                messagebox.showinfo("Success", f"Paused camera: {camera_id}")
    
    def _resume_selected_camera(self):
        """Resume camera được select"""
        camera_id = self._get_selected_camera_id()
        if camera_id:
            success = self.manager.resume_camera(camera_id)
            if success:
                messagebox.showinfo("Success", f"Resumed camera: {camera_id}")
    
    def _start_all_cameras(self):
        """Start tất cả cameras"""
        if messagebox.askyesno("Confirm", "Start all enabled cameras?"):
            results = self.manager.start_all_cameras()
            success_count = sum(results.values())
            messagebox.showinfo("Result", f"Started {success_count}/{len(results)} cameras")
    
    def _stop_all_cameras(self):
        """Stop tất cả cameras"""
        if messagebox.askyesno("Confirm", "Stop all running cameras?"):
            results = self.manager.stop_all_cameras()
            success_count = sum(results.values())
            messagebox.showinfo("Result", f"Stopped {success_count}/{len(results)} cameras")
    
    def _restart_system(self):
        """Restart toàn bộ hệ thống"""
        if messagebox.askyesno("Confirm", "Restart the entire system?\nThis will stop all cameras and restart them."):
            success = self.manager.restart_system()
            if success:
                messagebox.showinfo("Success", "System restarted successfully")
            else:
                messagebox.showerror("Error", "System restart failed")
    
    def _cleanup_resources(self):
        """Force cleanup resources"""
        if messagebox.askyesno("Confirm", "Force cleanup system resources?\nThis may cause temporary performance impact."):
            self.manager.force_cleanup_resources()
            messagebox.showinfo("Success", "Resource cleanup completed")
    
    def _switch_camera_dialog(self):
        """Show dialog để switch camera"""
        active_cameras = list(self.manager.active_cameras)
        if not active_cameras:
            messagebox.showwarning("Warning", "No active cameras to switch to")
            return
        
        # Simple selection dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Switch Camera View")
        dialog.geometry("300x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Select camera to display:", font=('Arial', 10)).pack(pady=10)
        
        camera_var = tk.StringVar()
        for camera_id in active_cameras:
            config = self.manager.camera_configs.get(camera_id)
            name = config.name if config else camera_id
            ttk.Radiobutton(dialog, text=f"{name} ({camera_id})", 
                          variable=camera_var, value=camera_id).pack(anchor=tk.W, padx=20)
        
        if active_cameras:
            camera_var.set(active_cameras[0])
        
        def switch_and_close():
            selected = camera_var.get()
            if selected:
                self.manager.switch_display_camera(selected)
            dialog.destroy()
        
        ttk.Button(dialog, text="Switch", command=switch_and_close).pack(pady=10)
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).pack(pady=5)
    
    def _export_report(self):
        """Export system report"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"parking_system_report_{timestamp}.json"
            
            # Create reports directory if not exists
            import os
            os.makedirs("reports", exist_ok=True)
            filepath = os.path.join("reports", filename)
            
            success = self.manager.export_system_report(filepath)
            if success:
                messagebox.showinfo("Success", f"Report exported to:\n{filepath}")
            else:
                messagebox.showerror("Error", "Failed to export report")
                
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")
    
    # ==================== LIFECYCLE METHODS ====================
    
    def run(self):
        """Chạy GUI main loop"""
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        try:
            logger.info("🎮 Starting GUI main loop...")
            self.root.mainloop()
        except Exception as e:
            logger.error(f"❌ GUI error: {e}")
        finally:
            logger.info("🎮 GUI main loop ended")
    
    def _on_closing(self):
        """Handle window closing event"""
        if messagebox.askokcancel("Quit", "Are you sure you want to quit?\nThis will stop all cameras and cleanup resources."):
            logger.info("🛑 GUI shutdown initiated by user")
            
            # Cancel any scheduled updates
            if hasattr(self, 'root'):
                try:
                    # Cancel all pending after() calls
                    for after_id in self.root.tk.call('after', 'info'):
                        self.root.after_cancel(after_id)
                except:
                    pass
            
            # Cleanup manager
            try:
                self.manager.cleanup()
            except Exception as e:
                logger.error(f"❌ Error during manager cleanup: {e}")
            
            # Destroy GUI
            self.root.destroy()
            logger.info("✅ GUI shutdown completed")


# ==================== CLI INTERFACE CLASS ====================

class MultiCameraCLI:
    """
    Command Line Interface cho hệ thống đa camera
    Alternative cho GUI interface
    """
    
    def __init__(self, manager: MultiCameraManager):
        """
        Khởi tạo CLI interface
        
        Args:
            manager (MultiCameraManager): Manager instance
        """
        self.manager = manager
        self.running = True
        self.commands = {
            'help': self._cmd_help,
            'list': self._cmd_list,
            'status': self._cmd_status,
            'start': self._cmd_start,
            'stop': self._cmd_stop,
            'pause': self._cmd_pause,
            'resume': self._cmd_resume,
            'switch': self._cmd_switch,
            'parking': self._cmd_parking,
            'stats': self._cmd_stats,
            'restart': self._cmd_restart,
            'cleanup': self._cmd_cleanup,
            'export': self._cmd_export,
            'quit': self._cmd_quit,
            'exit': self._cmd_quit
        }
        
        logger.info("🖥️ CLI interface initialized")
    
    # ==================== CLI MAIN METHODS ====================
    
    def run(self):
        """Run CLI main loop"""
        print("\n" + "="*60)
        print("🎮 Multi-Camera Parking System - CLI Interface")
        print("="*60)
        print("Type 'help' for available commands")
        print("Type 'quit' or 'exit' to close")
        
        while self.running:
            try:
                command_line = input("\n> ").strip()
                if not command_line:
                    continue
                
                self._handle_command(command_line)
                
            except KeyboardInterrupt:
                print("\n🛑 Interrupted by user")
                break
            except EOFError:
                print("\n🛑 EOF received")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n👋 Goodbye!")
    
    def _handle_command(self, command_line: str):
        """Handle command input"""
        parts = command_line.lower().split()
        if not parts:
            return
        
        cmd = parts[0]
        args = parts[1:] if len(parts) > 1 else []
        
        if cmd in self.commands:
            try:
                self.commands[cmd](args)
            except Exception as e:
                print(f"❌ Command error: {e}")
        else:
            print(f"❌ Unknown command: {cmd}. Type 'help' for available commands.")
    
    # ==================== CLI COMMAND METHODS ====================
    
    def _cmd_help(self, args):
        """Show help message"""
        print("\n📖 Available Commands:")
        print("="*40)
        commands_help = [
            ("help", "Show this help message"),
            ("list", "List all cameras"),
            ("status", "Show system status"),
            ("start <id>|all", "Start camera(s)"),
            ("stop <id>|all", "Stop camera(s)"),
            ("pause <id>", "Pause camera"),
            ("resume <id>", "Resume camera"),
            ("switch <id>", "Switch display camera"),
            ("parking", "Show parking spots status"),
            ("stats [<id>]", "Show detailed statistics"),
            ("restart", "Restart entire system"),
            ("cleanup", "Cleanup resources"),
            ("export", "Export system report"),
            ("quit/exit", "Exit application")
        ]
        
        for cmd, desc in commands_help:
            print(f"  {cmd:20} - {desc}")
    
    def _cmd_list(self, args):
        """List all cameras"""
        print("\n📹 Camera List:")
        print("-" * 80)
        
        for camera_id, config in self.manager.camera_configs.items():
            status = "🟢 Active" if camera_id in self.manager.active_cameras else "🔴 Inactive"
            display = "📺 " if camera_id == self.manager.current_display_camera else "   "
            priority = "⭐" * config.priority
            enabled = "✅" if config.enabled else "❌"
            
            print(f"  {display}{status} {enabled} {camera_id:12} {config.name:25} {priority}")
    
    def _cmd_status(self, args):
        """Show system status"""
        try:
            system_status = self.manager.get_system_status()
            
            print("\n📊 System Status:")
            print("-" * 50)
            
            info = system_status['system_info']
            print(f"  ⏱️  Uptime: {info['uptime_seconds']/3600:.1f}h")
            print(f"  📹 Cameras: {info['active_cameras']}/{info['configured_cameras']}")
            print(f"  📺 Display: {info['current_display'] or 'None'}")
            print(f"  🎯 Max Concurrent: {info['max_concurrent']}")
            
            stats = system_status['aggregated_stats']
            print(f"\n📈 Processing Stats:")
            print(f"  📼 Frames: {stats['total_frames_processed']:,}")
            print(f"  📤 Events: {stats['total_events_generated']:,}")
            print(f"  🚗 Plates: {stats['total_plates_detected']:,}")
            print(f"  📊 Occupancy: {stats['occupancy_rate']}%")
            
            resources = system_status['resource_usage']
            print(f"\n💾 Resources:")
            print(f"  🧠 Memory: {resources['memory_usage_mb']:.0f} MB")
            print(f"  🖥️  GPU: {resources['gpu_memory_mb']:.0f} MB")
            print(f"  🧵 Threads: {resources['active_threads']}")
            
            if 'server_connection' in system_status:
                server = system_status['server_connection']
                conn_status = "🟢 Connected" if server.get('is_connected') else "🔴 Offline"
                print(f"\n🌐 Server: {conn_status}")
                if not server.get('is_connected'):
                    print(f"  📡 Queue: {server.get('offline_queue_size', 0)}")
                    
        except Exception as e:
            print(f"❌ Error getting status: {e}")
    
    def _cmd_start(self, args):
        """Start camera(s)"""
        if not args:
            print("❌ Usage: start <camera_id> or start all")
            return
        
        target = args[0].lower()
        
        if target == 'all':
            print("🚀 Starting all cameras...")
            results = self.manager.start_all_cameras()
            success_count = sum(results.values())
            print(f"✅ Started {success_count}/{len(results)} cameras")
            
            for camera_id, success in results.items():
                status = "✅" if success else "❌"
                print(f"  {status} {camera_id}")
        else:
            camera_id = self._normalize_camera_id(args[0])
            success = self.manager.start_camera(camera_id)
            if success:
                print(f"✅ Started camera: {camera_id}")
            else:
                print(f"❌ Failed to start camera: {camera_id}")
    
    def _cmd_stop(self, args):
        """Stop camera(s)"""
        if not args:
            print("❌ Usage: stop <camera_id> or stop all")
            return
        
        target = args[0].lower()
        
        if target == 'all':
            print("🛑 Stopping all cameras...")
            results = self.manager.stop_all_cameras()
            success_count = sum(results.values())
            print(f"✅ Stopped {success_count}/{len(results)} cameras")
        else:
            camera_id = self._normalize_camera_id(args[0])
            success = self.manager.stop_camera(camera_id)
            if success:
                print(f"✅ Stopped camera: {camera_id}")
            else:
                print(f"❌ Failed to stop camera: {camera_id}")
    
    def _cmd_pause(self, args):
        """Pause camera"""
        if not args:
            print("❌ Usage: pause <camera_id>")
            return
        
        camera_id = self._normalize_camera_id(args[0])
        success = self.manager.pause_camera(camera_id)
        if success:
            print(f"⏸️ Paused camera: {camera_id}")
        else:
            print(f"❌ Failed to pause camera: {camera_id}")
    
    def _cmd_resume(self, args):
        """Resume camera"""
        if not args:
            print("❌ Usage: resume <camera_id>")
            return
        
        camera_id = self._normalize_camera_id(args[0])
        success = self.manager.resume_camera(camera_id)
        if success:
            print(f"▶️ Resumed camera: {camera_id}")
        else:
            print(f"❌ Failed to resume camera: {camera_id}")
    
    def _cmd_switch(self, args):
        """Switch display camera"""
        if not args:
            print("❌ Usage: switch <camera_id>")
            return
        
        camera_id = self._normalize_camera_id(args[0])
        success = self.manager.switch_display_camera(camera_id)
        if success:
            print(f"🔄 Switched display to: {camera_id}")
        else:
            print(f"❌ Failed to switch to camera: {camera_id}")
    
    def _cmd_parking(self, args):
        """Show parking spots status"""
        try:
            all_parking_status = self.manager.get_all_parking_status()
            
            if not all_parking_status:
                print("📍 No active cameras with parking data")
                return
            
            print("\n🅿️  Parking Spots Status:")
            print("=" * 80)
            
            for camera_id, camera_data in all_parking_status.items():
                print(f"\n📹 {camera_data['camera_name']} ({camera_id}):")
                print("-" * 60)
                
                for spot in camera_data['spots']:
                    status_icon = "🚗" if spot['is_occupied'] else "🅿️"
                    plate_info = f" ({spot['plate_text']})" if spot['plate_text'] else ""
                    
                    # Calculate duration
                    duration = self._calculate_parking_duration(spot)
                    conf_info = f" ({spot['plate_confidence']:.3f})" if spot['plate_confidence'] > 0 else ""
                    
                    print(f"  {status_icon} {spot['spot_name']:10}{plate_info}{conf_info}{duration}")
                    
        except Exception as e:
            print(f"❌ Error getting parking status: {e}")
    
    def _cmd_stats(self, args):
        """Show detailed statistics"""
        try:
            if args:
                # Show stats for specific camera
                camera_id = self._normalize_camera_id(args[0])
                stats = self.manager.get_camera_stats(camera_id)
                if stats:
                    self._print_camera_stats(camera_id, stats)
                else:
                    print(f"❌ No stats available for camera: {camera_id}")
            else:
                # Show stats for all cameras
                all_stats = self.manager.get_all_camera_stats()
                
                if not all_stats:
                    print("📊 No camera statistics available")
                    return
                
                print("\n📈 Detailed Statistics:")
                print("=" * 80)
                
                for camera_id, stats in all_stats.items():
                    if stats:
                        self._print_camera_stats(camera_id, stats)
                        print()  # Empty line between cameras
                        
        except Exception as e:
            print(f"❌ Error getting statistics: {e}")
    
    def _cmd_restart(self, args):
        """Restart system"""
        print("🔄 Restarting system...")
        success = self.manager.restart_system()
        if success:
            print("✅ System restart completed")
        else:
            print("❌ System restart failed")
    
    def _cmd_cleanup(self, args):
        """Cleanup resources"""
        print("🧹 Cleaning up resources...")
        self.manager.force_cleanup_resources()
        print("✅ Resource cleanup completed")
    
    def _cmd_export(self, args):
        """Export system report"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reports/cli_report_{timestamp}.json"
            
            success = self.manager.export_system_report(filename)
            if success:
                print(f"📄 Report exported to: {filename}")
            else:
                print("❌ Failed to export report")
                
        except Exception as e:
            print(f"❌ Export error: {e}")
    
    def _cmd_quit(self, args):
        """Quit application"""
        print("🛑 Shutting down...")
        self.running = False
        self.manager.cleanup()
    
    # ==================== CLI HELPER METHODS ====================
    
    def _normalize_camera_id(self, camera_id: str) -> str:
        """Normalize camera ID format"""
        camera_id = camera_id.upper()
        if not camera_id.startswith('CAM_'):
            camera_id = f"CAM_{camera_id}"
        return camera_id
    
    def _calculate_parking_duration(self, spot):
        """Calculate parking duration for CLI display"""
        duration = ""
        if spot['is_occupied'] and spot['enter_time']:
            try:
                enter_dt = datetime.fromisoformat(spot['enter_time'])
                duration_mins = int((datetime.now() - enter_dt).total_seconds() / 60)
                duration = f" - {duration_mins}m"
            except:
                pass
        return duration
    
    def _print_camera_stats(self, camera_id: str, stats: Dict[str, Any]):
        """Print statistics for a camera"""
        config = self.manager.camera_configs.get(camera_id)
        camera_name = config.name if config else camera_id
        
        status = "🟢 Running" if stats['is_running'] else "🔴 Stopped"
        if stats.get('is_paused'):
            status += " (⏸️ Paused)"
        
        print(f"📹 {camera_name} ({camera_id}) - {status}")
        print(f"  ⚡ FPS: {stats['fps']:.1f}")
        print(f"  📼 Frames: {stats['frames_processed']:,}")
        print(f"  📤 Events: {stats['events_generated']}")
        print(f"  🚗 Plates: {stats['plates_detected']}")
        print(f"  🅿️  Spots: {stats['occupied_spots']}/{stats['parking_spots_count']}")
        print(f"  ⏱️  Avg Time: {stats['avg_processing_time']:.3f}s")
        print(f"  📊 Queues: Frame={stats['queue_sizes']['frame_queue']}, Result={stats['queue_sizes']['result_queue']}")
        print(f"  ❌ Errors: {stats['errors']}")
        print(f"  ⏰ Uptime: {stats['uptime_seconds']/3600:.1f}h")