"""
3D Slicer Python Bridge for Surgical Physics & Autonomous Robotic Ultrasound.
Connects directly to the PyBullet & Acoustic Simulation Server via OpenIGTLink (pyigtl),
subscribing to live ultrasound images, probe transforms, and 3D reconstructed bone points.

Compatible with Slicer 5.0+ and embedded Python (pyigtl).

Features:
- Side-by-side clinical layout (2D live ultrasound on left, 3D anatomy & robot probe on right).
- In-place memory buffer mutation of vtkImageData and vtkPoints to prevent memory leaks and UI lag.
- Physical 3D Ultrasound Transducer Model dynamically moving with the robot end-effector.
- Automatic 2D Slice Viewer background binding with calibrated 8-bit window/level.
- Automatic 3D camera auto-centering and focal point tracking.
- Continuous auto-reconnection loop resilient to process start order and server restarts.
"""

import sys
import time
import numpy as np

try:
    import slicer
    from slicer.util import VTKObservationMixin
    import vtk
    import vtk.util.numpy_support as vtk_np
except ImportError:
    print("[ERROR] This script must be run within 3D Slicer Python environment.")
    sys.exit(1)

try:
    import pyigtl
except ImportError:
    slicer.util.pip_install("pyigtl")
    import pyigtl


class SlicerSurgicalPhysicsBridge:
    def __init__(self, host="127.0.0.1", port=18944, update_interval_ms=33):
        self.host = host
        self.port = port
        self.update_interval_ms = update_interval_ms
        self.client = None
        self.is_connected = False
        self.last_connect_attempt = 0.0
        self.frames_received = 0
        self.camera_initialized = False

        # MRML Nodes
        self.us_volume_node = None
        self.probe_transform_node = None
        self.probe_model_node = None
        self.bone_model_node = None
        self.bone_points = None
        self.bone_polydata = None

        self._setup_layout_and_views()
        self._setup_mrml_nodes()

        # Qt Timer for continuous non-blocking polling and auto-reconnection
        from qt import QTimer
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_timer_tick)
        self.timer.start(self.update_interval_ms)
        print(f"[*] Slicer Surgical Physics Bridge initialized. Polling {self.host}:{self.port} at {1000//self.update_interval_ms} Hz...")

    def _setup_layout_and_views(self):
        """Sets up clinical side-by-side view (2D Red slice viewer + 3D View)."""
        layoutManager = slicer.app.layoutManager()
        if layoutManager:
            # Side-by-side: Left = Red Slice (Ultrasound), Right = 3D Patient Space
            layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutSideBySideView)
            print("[+] Configured Slicer UI layout: Side-by-Side (2D Ultrasound | 3D Scene)")

    def _setup_mrml_nodes(self):
        """Creates or retrieves MRML nodes in Slicer scene."""
        scene = slicer.mrmlScene

        # 1. Transform Node for Probe
        self.probe_transform_node = scene.GetFirstNodeByName("UltrasoundProbe")
        if not self.probe_transform_node:
            self.probe_transform_node = scene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "UltrasoundProbe")
            print("[+] Created vtkMRMLLinearTransformNode: UltrasoundProbe")

        # 2. 3D Visual Mesh for Ultrasound Probe Transducer
        self.probe_model_node = scene.GetFirstNodeByName("UltrasoundProbeModel")
        if not self.probe_model_node:
            self.probe_model_node = self._create_probe_3d_model(scene)
            if self.probe_model_node:
                self.probe_model_node.SetAndObserveTransformNodeID(self.probe_transform_node.GetID())
                print("[+] Attached 3D Transducer Model to UltrasoundProbe transform")

        # 3. Volume Node for Ultrasound Image
        self.us_volume_node = scene.GetFirstNodeByName("US_Image")
        if not self.us_volume_node:
            self.us_volume_node = scene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "US_Image")
            init_img = np.zeros((1, 160, 100), dtype=np.uint8)
            slicer.util.updateVolumeFromArray(self.us_volume_node, init_img)
            self.us_volume_node.SetAndObserveTransformNodeID(self.probe_transform_node.GetID())

            # Calibrated Window/Level for 8-bit B-mode ultrasound
            dnode = self.us_volume_node.GetDisplayNode()
            if dnode:
                dnode.SetAutoWindowLevel(False)
                dnode.SetWindow(255.0)
                dnode.SetLevel(127.5)

            # Bind immediately to slice viewer background
            slicer.util.setSliceViewerLayers(background=self.us_volume_node, fit=True)
            print("[+] Created vtkMRMLScalarVolumeNode: US_Image and bound to 2D Slice Viewers")

        # 4. Model Node for Reconstructed Bone Points
        self.bone_model_node = scene.GetFirstNodeByName("ReconstructedBone")
        if not self.bone_model_node:
            self.bone_model_node = scene.AddNewNodeByClass("vtkMRMLModelNode", "ReconstructedBone")
            self.bone_polydata = vtk.vtkPolyData()
            self.bone_points = vtk.vtkPoints()
            self.bone_polydata.SetPoints(self.bone_points)

            verts = vtk.vtkCellArray()
            self.bone_polydata.SetVerts(verts)
            self.bone_model_node.SetAndObservePolyData(self.bone_polydata)

            # High-visibility glowing green/gold point styling
            dnode = self.bone_model_node.GetDisplayNode()
            if not dnode:
                dnode = scene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
                self.bone_model_node.SetAndObserveDisplayNodeID(dnode.GetID())
            dnode.SetColor(0.15, 0.95, 0.35)  # Glowing emerald green
            dnode.SetPointSize(6.0)
            dnode.SetLighting(False)  # Unshaded for maximum point brilliance
            dnode.SetVisibility(True)
            print("[+] Created vtkMRMLModelNode: ReconstructedBone (Emerald Point Cloud)")

    def _create_probe_3d_model(self, scene):
        """Builds a realistic 3D ultrasound probe transducer geometry using VTK."""
        try:
            # Handle: cylindrical ergonomic shaft
            cyl = vtk.vtkCylinderSource()
            cyl.SetRadius(10.0)
            cyl.SetHeight(75.0)
            cyl.SetResolution(32)
            cyl.Update()

            # Rotate handle along Z-axis (pointing away from skin)
            trans_cyl = vtk.vtkTransform()
            trans_cyl.RotateX(90)
            trans_cyl.Translate(0, 0, 37.5)
            f_cyl = vtk.vtkTransformPolyDataFilter()
            f_cyl.SetInputConnection(cyl.GetOutputPort())
            f_cyl.SetTransform(trans_cyl)
            f_cyl.Update()

            # Transducer Head: wedge housing
            head = vtk.vtkCubeSource()
            head.SetXLength(38.0)  # Lateral width
            head.SetYLength(16.0)  # Elevation thickness
            head.SetZLength(20.0)  # Head height
            head.SetCenter(0.0, 0.0, 10.0)
            head.Update()

            # Acoustic Lens: transducer contact face
            lens = vtk.vtkCubeSource()
            lens.SetXLength(36.0)
            lens.SetYLength(14.0)
            lens.SetZLength(3.0)
            lens.SetCenter(0.0, 0.0, -1.5)
            lens.Update()

            # Combine geometries into single polydata
            append = vtk.vtkAppendPolyData()
            append.AddInputData(f_cyl.GetOutput())
            append.AddInputData(head.GetOutput())
            append.AddInputData(lens.GetOutput())
            append.Update()

            model_node = scene.AddNewNodeByClass("vtkMRMLModelNode", "UltrasoundProbeModel")
            model_node.SetAndObservePolyData(append.GetOutput())

            dnode = scene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
            model_node.SetAndObserveDisplayNodeID(dnode.GetID())
            dnode.SetColor(0.2, 0.5, 0.85)  # Medical blue
            dnode.SetOpacity(0.85)
            dnode.SetVisibility(True)
            return model_node
        except Exception as e:
            print(f"[!] Warning: Could not create probe 3D model: {e}")
            return None

    def connect_server(self):
        """Attempts connection to OpenIGTLink server."""
        try:
            self.client = pyigtl.OpenIGTLinkClient(host=self.host, port=self.port)
            self.is_connected = True
            print(f"[✓] Successfully connected to OpenIGTLink Server at {self.host}:{self.port}!")
            return True
        except Exception as e:
            self.is_connected = False
            return False

    def on_timer_tick(self):
        """Non-blocking periodic update for packet streaming and auto-reconnect."""
        now = time.time()

        # Auto-reconnection logic
        if not self.is_connected:
            if now - self.last_connect_attempt >= 1.0:
                self.last_connect_attempt = now
                self.connect_server()
            return

        # Connected: process incoming messages
        try:
            messages = self.client.get_latest_messages()
        except Exception as e:
            print(f"[!] Error reading OpenIGTLink socket: {e}. Reconnecting...")
            self.is_connected = False
            if self.client:
                try:
                    self.client.stop()
                except Exception:
                    pass
                self.client = None
            return

        # Coalesce messages to retain only latest payload per type (prevents redundant VTK updates)
        latest_probe_msg = None
        latest_image_msg = None
        latest_points_msg = None

        for msg in messages:
            if isinstance(msg, pyigtl.TransformMessage) and msg.device_name == "UltrasoundProbe":
                latest_probe_msg = msg
            elif isinstance(msg, pyigtl.ImageMessage) and msg.device_name == "US_Image":
                latest_image_msg = msg
            elif isinstance(msg, pyigtl.PointMessage) and msg.device_name == "ReconstructedBone":
                latest_points_msg = msg

        # A. Apply Latest Probe Transform
        if latest_probe_msg is not None:
            vtk_mat = vtk.vtkMatrix4x4()
            for r in range(4):
                for c in range(4):
                    vtk_mat.SetElement(r, c, latest_probe_msg.matrix[r, c])
            self.probe_transform_node.SetMatrixTransformToParent(vtk_mat)

        # B. Apply Latest 2D Ultrasound B-mode Slice
        if latest_image_msg is not None:
            img_data = latest_image_msg.image
            if img_data.ndim == 3 and img_data.shape[2] == 1:
                img_data = img_data[:, :, 0]

            # Format to (1, H, W) for Slicer scalar volume
            vol_slice = img_data[np.newaxis, :, :]
            slicer.util.updateVolumeFromArray(self.us_volume_node, vol_slice)

            if self.frames_received % 10 == 0:
                slicer.util.setSliceViewerLayers(background=self.us_volume_node, fit=True)
            self.frames_received += 1

        # C. Apply Latest 3D Reconstructed Bone Points
        if latest_points_msg is not None:
            pts = latest_points_msg.positions
            if pts is not None and len(pts) > 0:
                n_pts = len(pts)
                pts_array = np.ascontiguousarray(pts, dtype=np.float32)
                vtk_data_array = vtk_np.numpy_to_vtk(pts_array, deep=False)
                self.bone_points.SetData(vtk_data_array)
                self.bone_points.Modified()

                # Fast vectorized cell array construction for 3D point cloud
                cell_data = np.empty((n_pts, 2), dtype=np.int64)
                cell_data[:, 0] = 1
                cell_data[:, 1] = np.arange(n_pts, dtype=np.int64)
                cell_array = vtk_np.numpy_to_vtkIdTypeArray(cell_data.ravel(), deep=False)
                verts = vtk.vtkCellArray()
                verts.SetCells(n_pts, cell_array)
                self.bone_polydata.SetVerts(verts)
                self.bone_polydata.Modified()

                # Auto-reset camera focal point to center onto patient spine on first batch of points
                if not self.camera_initialized and n_pts >= 50:
                    layoutManager = slicer.app.layoutManager()
                    if layoutManager:
                        threeDWidget = layoutManager.threeDWidget(0)
                        if threeDWidget:
                            threeDWidget.threeDView().resetFocalPoint()
                            threeDWidget.threeDView().resetCamera()
                            self.camera_initialized = True
                            print(f"[+] 3D Camera centered on reconstructed bone ({n_pts} points)")

    def disconnect(self):
        self.timer.stop()
        if self.client:
            try:
                self.client.stop()
            except Exception:
                pass
            self.client = None
        self.is_connected = False
        print("[*] Slicer bridge disconnected.")


# Instantiate bridge in Slicer Python runtime
if __name__ == "__main__":
    bridge = SlicerSurgicalPhysicsBridge()
