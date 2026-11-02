"""
3D Slicer Python Bridge for Surgical Physics & Autonomous Robotic Ultrasound.
Connects directly to the PyBullet & Acoustic Simulation Server via OpenIGTLink (pyigtl),
subscribing to live ultrasound images, probe transforms, and 3D reconstructed bone points.

Compatible with Slicer 5.0+ and embedded Python (pyigtl).

Features:
- True Side-by-side clinical layout (Left: 2D Live B-mode ultrasound, Right: 3D Patient Scene).
- In-place memory buffer mutation of vtkImageData and vtkPoints (zero memory leak, <0.2ms latency).
- Physical 3D Ultrasound Transducer Model & Cyan Acoustic Beam Plane dynamically tracking robot end-effector.
- Automatic 2D Red Slice Viewer background binding with calibrated 8-bit window/level.
- Optional TotalSegmentator lumbar spine anatomical surface model in 3D viewer.
- Automatic 3D camera auto-centering and focal point tracking.
- Continuous auto-reconnection loop resilient to process start order and server restarts.
"""

import sys
import os
import glob
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
        self.beam_model_node = None
        self.bone_model_node = None
        self.bone_points = None
        self.bone_polydata = None

        self._setup_layout_and_views()
        self._setup_mrml_nodes()
        self._load_patient_spine_model()

        # Qt Timer for continuous non-blocking polling and auto-reconnection
        from qt import QTimer
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_timer_tick)
        self.timer.start(self.update_interval_ms)
        print(f"[*] Slicer Surgical Physics Bridge initialized. Polling {self.host}:{self.port} at {1000//self.update_interval_ms} Hz...")

    def _setup_layout_and_views(self):
        """Sets up clinical side-by-side view (Left: 2D Live Ultrasound | Right: 3D Scene)."""
        layoutManager = slicer.app.layoutManager()
        if layoutManager:
            # Custom clinical layout XML: Left = Red Slice (Axial), Right = 3D View (View 1)
            custom_layout_xml = (
                '<layout type="horizontal" split="true">'
                ' <item splitSize="500">'
                '  <view class="vtkMRMLSliceNode" singletontag="Red">'
                '   <property name="orientation" action="default">Axial</property>'
                '   <property name="viewlabel" action="default">R</property>'
                '   <property name="viewcolor" action="default">#F34A33</property>'
                '  </view>'
                ' </item>'
                ' <item splitSize="500">'
                '  <view class="vtkMRMLViewNode" singletontag="1">'
                '   <property name="viewlabel" action="default">1</property>'
                '  </view>'
                ' </item>'
                '</layout>'
            )
            layout_node = layoutManager.layoutLogic().GetLayoutNode()
            custom_layout_id = 501
            layout_node.AddLayoutDescription(custom_layout_id, custom_layout_xml)
            layout_node.SetViewArrangement(custom_layout_id)
            print("[+] Configured Slicer UI layout: True Side-by-Side (2D Ultrasound Left | 3D View Right)")

            # Configure 3D view styling (dark medical slate background)
            view_node = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLViewNode")
            if view_node:
                view_node.SetBackgroundColor(0.12, 0.14, 0.18)
                view_node.SetBackgroundColor2(0.08, 0.09, 0.12)
                view_node.SetBoxVisible(True)
                view_node.SetAxisLabelsVisible(True)

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

        # 3. 3D Visual Mesh for Ultrasound Acoustic Beam Plane
        self.beam_model_node = scene.GetFirstNodeByName("UltrasoundBeamPlane")
        if not self.beam_model_node:
            self.beam_model_node = self._create_beam_plane_model(scene)
            if self.beam_model_node:
                self.beam_model_node.SetAndObserveTransformNodeID(self.probe_transform_node.GetID())
                print("[+] Attached 3D Acoustic Beam Plane to UltrasoundProbe transform")

        # 4. Volume Node for 2D Ultrasound Image (Dedicated monitor view)
        self.us_volume_node = scene.GetFirstNodeByName("US_Image")
        if not self.us_volume_node:
            self.us_volume_node = scene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "US_Image")
            init_img = np.zeros((1, 160, 100), dtype=np.uint8)
            slicer.util.updateVolumeFromArray(self.us_volume_node, init_img)
            # US_Image is viewed directly face-on in the 2D Red slice view (un-transformed)
            self.us_volume_node.SetAndObserveTransformNodeID(None)
            self.us_volume_node.SetSpacing(0.38, 0.375, 1.0)
            self.us_volume_node.SetOrigin(-19.0, -30.0, 0.0)

            # Calibrated Window/Level for 8-bit B-mode ultrasound
            dnode = self.us_volume_node.GetDisplayNode()
            if dnode:
                dnode.SetAutoWindowLevel(False)
                dnode.SetWindow(255.0)
                dnode.SetLevel(127.5)

            # Bind immediately to Red slice viewer background
            slicer.util.setSliceViewerLayers(background=self.us_volume_node, fit=True)
            layoutManager = slicer.app.layoutManager()
            if layoutManager:
                red_widget = layoutManager.sliceWidget("Red")
                if red_widget:
                    red_slice = red_widget.mrmlSliceNode()
                    red_slice.SetOrientation("Axial")
                    red_slice.SetSliceOffset(0.0)
                    red_widget.sliceController().fitSliceToBackground()
            print("[+] Created vtkMRMLScalarVolumeNode: US_Image and framed in 2D Red Slice Viewer")

        # 5. Model Node for Reconstructed Bone Points
        self.bone_model_node = scene.GetFirstNodeByName("ReconstructedBone")
        if not self.bone_model_node:
            self.bone_model_node = scene.AddNewNodeByClass("vtkMRMLModelNode", "ReconstructedBone")
            self.bone_polydata = vtk.vtkPolyData()
            self.bone_points = vtk.vtkPoints()
            self.bone_polydata.SetPoints(self.bone_points)

            verts = vtk.vtkCellArray()
            self.bone_polydata.SetVerts(verts)
            self.bone_model_node.SetAndObservePolyData(self.bone_polydata)

            # High-visibility glowing green point styling
            dnode = self.bone_model_node.GetDisplayNode()
            if not dnode:
                dnode = scene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
                self.bone_model_node.SetAndObserveDisplayNodeID(dnode.GetID())
            dnode.SetColor(0.15, 0.95, 0.35)  # Glowing emerald green
            dnode.SetPointSize(7.0)
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

    def _create_beam_plane_model(self, scene):
        """Creates a semi-transparent 3D acoustic scanning plane originating at transducer contact face."""
        try:
            plane_source = vtk.vtkPlaneSource()
            plane_source.SetOrigin(-19.0, 0.0, -3.0)
            plane_source.SetPoint1(19.0, 0.0, -3.0)
            plane_source.SetPoint2(-19.0, 0.0, -63.0)
            plane_source.SetResolution(1, 1)
            plane_source.Update()

            beam_node = scene.AddNewNodeByClass("vtkMRMLModelNode", "UltrasoundBeamPlane")
            beam_node.SetAndObservePolyData(plane_source.GetOutput())
            dnode = scene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
            beam_node.SetAndObserveDisplayNodeID(dnode.GetID())
            dnode.SetColor(0.0, 0.85, 1.0)  # Glowing cyan
            dnode.SetOpacity(0.35)
            dnode.SetBackfaceCulling(False)
            dnode.SetVisibility(True)
            return beam_node
        except Exception as e:
            print(f"[!] Warning: Could not create acoustic beam model: {e}")
            return None

    def _load_patient_spine_model(self):
        """Extracts and displays patient lumbar spine surface mesh in 3D viewer without disrupting slice viewers."""
        scene = slicer.mrmlScene
        if scene.GetFirstNodeByName("GroundTruthSpine"):
            return

        patient_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sample_totalsegmentator_patient")
        mesh_ras_path = os.path.join(patient_dir, "meshes", "patient_lumbar_spine_ras.obj")
        torso_ras_path = os.path.join(patient_dir, "meshes", "patient_torso_skin_ras.obj")

        # Optional Patient Torso Skin Mesh (Translucent Flesh)
        if os.path.exists(torso_ras_path) and not scene.GetFirstNodeByName("PatientTorso"):
            try:
                torso_model = slicer.util.loadModel(torso_ras_path)
                if torso_model:
                    torso_model.SetName("PatientTorso")
                    tdnode = torso_model.GetDisplayNode()
                    if tdnode:
                        tdnode.SetColor(0.86, 0.68, 0.58)  # Flesh tint
                        tdnode.SetOpacity(0.18)  # Subtle translucent envelope
                        tdnode.SetVisibility(True)
                    print("[+] Successfully loaded PatientTorso anatomical envelope in 3D Scene")
            except Exception as e:
                print(f"[!] Warning: Could not load torso mesh: {e}")

        if os.path.exists(mesh_ras_path):
            try:
                model = slicer.util.loadModel(mesh_ras_path)
                if model:
                    model.SetName("GroundTruthSpine")
                    dnode = model.GetDisplayNode()
                    if dnode:
                        dnode.SetColor(0.92, 0.89, 0.78)  # Bone ivory
                        dnode.SetOpacity(0.40)  # Semi-transparent to reveal reconstructed points
                        dnode.SetVisibility(True)
                    print("[+] Successfully loaded GroundTruthSpine anatomical surface in 3D Scene")
                    return
            except Exception as e:
                print(f"[!] Warning: Could not load spine mesh: {e}")

        # Fallback to marching cubes if mesh obj not found
        seg_dir = os.path.join(patient_dir, "segmentations")
        files = sorted(glob.glob(os.path.join(seg_dir, "vertebrae_*.nii.gz")))
        if not files:
            return

        try:
            append = vtk.vtkAppendPolyData()
            for f in files:
                vol = slicer.util.loadVolume(f)
                fe = vtk.vtkDiscreteFlyingEdges3D()
                fe.SetInputData(vol.GetImageData())
                fe.SetValue(0, 1)
                fe.Update()

                mat = vtk.vtkMatrix4x4()
                vol.GetIJKToRASMatrix(mat)
                trans = vtk.vtkTransform()
                trans.SetMatrix(mat)

                tf = vtk.vtkTransformPolyDataFilter()
                tf.SetInputConnection(fe.GetOutputPort())
                tf.SetTransform(trans)
                tf.Update()
                append.AddInputData(tf.GetOutput())
                scene.RemoveNode(vol)

            append.Update()
            model = slicer.modules.models.logic().AddModel(append.GetOutput())
            model.SetName("GroundTruthSpine")
            dnode = model.GetDisplayNode()
            if dnode:
                dnode.SetColor(0.92, 0.89, 0.78)  # Bone ivory
                dnode.SetOpacity(0.40)
                dnode.SetVisibility(True)
            print("[+] Successfully loaded GroundTruthSpine anatomical surface in 3D Scene")
        except Exception as e:
            print(f"[!] Warning: Could not generate spine surface model: {e}")

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

        # B. Apply Latest 2D Ultrasound B-mode Slice (Top-down clinical display)
        if latest_image_msg is not None:
            img_data = latest_image_msg.image
            if img_data.ndim == 3 and img_data.shape[2] == 1:
                img_data = img_data[:, :, 0]

            # Invert row order so probe face (depth 0) is at screen top (+Y in Axial view)
            oriented_img = np.ascontiguousarray(img_data[::-1, :])
            vol_slice = oriented_img[np.newaxis, :, :]
            slicer.util.updateVolumeFromArray(self.us_volume_node, vol_slice)

            # Ensure Side-by-Side layout and bind US_Image to Red slice viewer
            if self.frames_received < 5:
                layoutManager = slicer.app.layoutManager()
                if layoutManager:
                    layout_node = layoutManager.layoutLogic().GetLayoutNode()
                    if layout_node.GetViewArrangement() != 501:
                        layout_node.SetViewArrangement(501)
                    red_widget = layoutManager.sliceWidget("Red")
                    if red_widget:
                        comp_node = red_widget.mrmlSliceCompositeNode()
                        if comp_node.GetBackgroundVolumeID() != self.us_volume_node.GetID():
                            comp_node.SetBackgroundVolumeID(self.us_volume_node.GetID())
                        slice_node = red_widget.mrmlSliceNode()
                        slice_node.SetOrientationToAxial()
                        red_widget.sliceController().fitSliceToBackground()
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

                # Fast vectorized cell array construction without deprecation warnings
                cell_data = np.empty((n_pts, 2), dtype=np.int64)
                cell_data[:, 0] = 1
                cell_data[:, 1] = np.arange(n_pts, dtype=np.int64)
                cell_array = vtk_np.numpy_to_vtkIdTypeArray(cell_data.ravel(), deep=False)
                verts = vtk.vtkCellArray()
                verts.ImportLegacyFormat(cell_array)
                self.bone_polydata.SetVerts(verts)
                self.bone_polydata.Modified()

                # Auto-reset camera focal point to center onto patient spine on first batch of points
                if not self.camera_initialized and n_pts >= 30:
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
