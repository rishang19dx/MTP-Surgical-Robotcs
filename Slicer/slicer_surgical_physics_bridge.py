"""
3D Slicer Python bridge for the Surgical Physics & Autonomous Robotic Ultrasound server.

Run inside Slicer (5.0+):  Slicer --python-script Slicer/slicer_surgical_physics_bridge.py
It connects to ``surgical_physics_bridge/src/unified_server.py`` with ``pyigtl`` and keeps
reconnecting until the server is up.

Frames
------
All patient content (meshes, probe pose, reconstructed bone points, skin deformation) arrives and
is stored in the simulation patient frame **P** (mm). P is an LPS frame (+X left, +Y posterior =
toward the dorsal skin, +Z superior), *not* RAS, although ``ct.nii.gz`` and the ``*_ras.obj`` mesh
files label it RAS (contract C7; see ``surgical_physics_bridge/src/coordinates.py``). Every patient
node is parented under one ``vtkMRMLLinearTransformNode`` named ``PatientToRAS`` that holds
``P_TO_RAS = diag(-1, -1, 1)`` (det +1). Nothing is flipped in the data itself. Meshes are read
with the order-preserving ``obj_io.read_obj`` (not ``slicer.util.loadModel``, which would assume
the OBJ is LPS and flip it a second time, and may reorder vertices).

Messages consumed (device name / type / payload)
------------------------------------------------
``UltrasoundProbe``   TransformMessage  4x4 T_P<-probe, mm (indented probe face = image plane, +Y = beam).
``US_Image``          ImageMessage      uint8 (H, W, 1) B-mode frame, row 0 = probe face. Shown in the Red view.
``ReconstructedBone`` ImageMessage      float32 (1, N, 3) bone points in P (mm).
``DeformedSkin``      ImageMessage      float32 (1, K, 4) rows ``[vertex_index, dx, dy, dz]``: displacement (mm, P)
                                        of vertex ``vertex_index`` of ``meshes/patient_torso_skin_ras.obj`` (file
                                        vertex order) from its undeformed position, computed by the server's tissue
                                        model. Vertices not listed are undeformed. A single row with index -1 resets
                                        the skin. Alternatively float32 (1, N, 3) = all N deformed vertex positions.
``Telemetry``         StringMessage     JSON (force, indentation, coupling, physics-rate stats); logged periodically.

Skin updates are in place: one persistent float32 array is bound to the torso ``vtkPoints`` once
(``numpy_to_vtk(deep=False)``); each frame writes ``arr[:] = base; arr[idx] += disp`` and calls
``Modified()``. No per-frame VTK arrays or copies are allocated.
"""

import json
import os
import sys
import time

import numpy as np

try:
    import slicer
    import vtk
    import vtk.util.numpy_support as vtk_np
except ImportError:
    print("[ERROR] This script must be run within the 3D Slicer Python environment.")
    sys.exit(1)

try:
    import pyigtl
except ImportError:
    slicer.util.pip_install("pyigtl")
    import pyigtl

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from surgical_physics_bridge.src.coordinates import P_TO_RAS  # noqa: E402
from surgical_physics_bridge.src.obj_io import read_obj  # noqa: E402

PATIENT_DIR = os.path.join(PROJECT_ROOT, "data", "sample_totalsegmentator_patient")
LAYOUT_ID = 501


def _array_payload(msg, ncols):
    """Image-message array transport -> (rows, ncols) float32, or None if the shape does not fit."""
    arr = np.asarray(msg.image, dtype=np.float32)
    if arr.size == 0 or arr.size % ncols != 0:
        return None
    return arr.reshape(-1, ncols)


class SlicerSurgicalPhysicsBridge:
    def __init__(self, host="127.0.0.1", port=18944, update_interval_ms=33):
        self.host = host
        self.port = port
        self.client = None
        self.is_connected = False
        self.last_connect_attempt = 0.0
        self.frames_received = 0
        self.camera_initialized = False
        self.last_telemetry_log = 0.0
        self._warned = set()

        self.patient_to_ras_node = None
        self.us_volume_node = None
        self.probe_transform_node = None
        self.bone_points = None
        self.bone_polydata = None
        self.torso_polydata = None
        self.skin_base = None      # (N, 3) float32 undeformed vertices (P)
        self.skin_live = None      # (N, 3) float32 array bound to the torso vtkPoints (deep=False)
        self.skin_deformed = False

        self._setup_layout()
        self._setup_nodes()
        self._load_patient_models()

        from qt import QTimer
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_timer_tick)
        self.timer.start(update_interval_ms)
        print(f"[*] Slicer bridge polling {self.host}:{self.port} every {update_interval_ms} ms")

    def _warn_once(self, key, text):
        if key not in self._warned:
            self._warned.add(key)
            print(f"[!] {text}")

    # ------------------------------------------------------------------ scene
    def _setup_layout(self):
        lm = slicer.app.layoutManager()
        if not lm:
            return
        xml = (
            '<layout type="horizontal" split="true">'
            ' <item splitSize="500"><view class="vtkMRMLSliceNode" singletontag="Red">'
            '  <property name="orientation" action="default">Axial</property>'
            '  <property name="viewlabel" action="default">R</property>'
            '  <property name="viewcolor" action="default">#F34A33</property></view></item>'
            ' <item splitSize="500"><view class="vtkMRMLViewNode" singletontag="1">'
            '  <property name="viewlabel" action="default">1</property></view></item>'
            '</layout>'
        )
        layout_node = lm.layoutLogic().GetLayoutNode()
        layout_node.AddLayoutDescription(LAYOUT_ID, xml)
        layout_node.SetViewArrangement(LAYOUT_ID)
        view_node = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLViewNode")
        if view_node:
            view_node.SetBackgroundColor(0.12, 0.14, 0.18)
            view_node.SetBackgroundColor2(0.08, 0.09, 0.12)

    def _get_or_add(self, cls, name):
        node = slicer.mrmlScene.GetFirstNodeByName(name)
        return node if node else slicer.mrmlScene.AddNewNodeByClass(cls, name)

    def _model_node(self, name, polydata, color, opacity, parent=None):
        node = self._get_or_add("vtkMRMLModelNode", name)
        node.SetAndObservePolyData(polydata)
        if not node.GetDisplayNode():
            node.CreateDefaultDisplayNodes()
        d = node.GetDisplayNode()
        d.SetColor(*color)
        d.SetOpacity(opacity)
        d.SetVisibility(True)
        node.SetAndObserveTransformNodeID((parent or self.patient_to_ras_node).GetID())
        return node

    def _setup_nodes(self):
        # Fixed P -> RAS transform: parent of every patient-frame node.
        self.patient_to_ras_node = self._get_or_add("vtkMRMLLinearTransformNode", "PatientToRAS")
        m = vtk.vtkMatrix4x4()
        for r in range(4):
            for c in range(4):
                m.SetElement(r, c, float(P_TO_RAS[r, c]))
        self.patient_to_ras_node.SetMatrixTransformToParent(m)

        # Probe pose (T_P<-probe) lives in P, so it is a child of PatientToRAS.
        self.probe_transform_node = self._get_or_add("vtkMRMLLinearTransformNode", "UltrasoundProbe")
        self.probe_transform_node.SetAndObserveTransformNodeID(self.patient_to_ras_node.GetID())
        self._model_node("UltrasoundProbeModel", self._probe_polydata(), (0.2, 0.5, 0.85), 0.85,
                         parent=self.probe_transform_node)
        beam = vtk.vtkPlaneSource()  # in probe frame: x lateral, +y = beam (depth)
        beam.SetOrigin(-19.0, 0.0, 0.0)
        beam.SetPoint1(19.0, 0.0, 0.0)
        beam.SetPoint2(-19.0, 60.0, 0.0)
        beam.Update()
        beam_node = self._model_node("UltrasoundBeamPlane", beam.GetOutput(), (0.0, 0.85, 1.0), 0.35,
                                     parent=self.probe_transform_node)
        beam_node.GetDisplayNode().SetBackfaceCulling(False)

        # 2D B-mode image, shown face-on in the Red slice view (not in patient space).
        self.us_volume_node = self._get_or_add("vtkMRMLScalarVolumeNode", "US_Image")
        slicer.util.updateVolumeFromArray(self.us_volume_node, np.zeros((1, 160, 100), dtype=np.uint8))
        self.us_volume_node.SetSpacing(0.38, 0.375, 1.0)
        self.us_volume_node.SetOrigin(-19.0, -30.0, 0.0)
        dnode = self.us_volume_node.GetDisplayNode()
        if dnode:
            dnode.SetAutoWindowLevel(False)
            dnode.SetWindow(255.0)
            dnode.SetLevel(127.5)
        slicer.util.setSliceViewerLayers(background=self.us_volume_node, fit=True)

        # Reconstructed bone point cloud (P).
        self.bone_polydata = vtk.vtkPolyData()
        self.bone_points = vtk.vtkPoints()
        self.bone_polydata.SetPoints(self.bone_points)
        self.bone_polydata.SetVerts(vtk.vtkCellArray())
        bone_node = self._model_node("ReconstructedBone", self.bone_polydata, (0.15, 0.95, 0.35), 1.0)
        bone_node.GetDisplayNode().SetPointSize(7.0)
        bone_node.GetDisplayNode().SetLighting(False)

    @staticmethod
    def _probe_polydata():
        """Probe geometry in the probe frame: face at y=0, beam along +y, handle along -y."""
        append = vtk.vtkAppendPolyData()
        handle = vtk.vtkCylinderSource()  # cylinder axis is y
        handle.SetRadius(10.0)
        handle.SetHeight(75.0)
        handle.SetCenter(0.0, -57.5, 0.0)
        handle.SetResolution(32)
        head = vtk.vtkCubeSource()
        head.SetXLength(38.0)
        head.SetYLength(20.0)
        head.SetZLength(16.0)
        head.SetCenter(0.0, -10.0, 0.0)
        for src in (handle, head):
            src.Update()
            append.AddInputData(src.GetOutput())
        append.Update()
        return append.GetOutput()

    @staticmethod
    def _polydata_from_obj(path):
        verts, faces = read_obj(path)
        poly = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        points.SetData(vtk_np.numpy_to_vtk(verts, deep=True))
        poly.SetPoints(points)
        cells = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces]).ravel()
        polys = vtk.vtkCellArray()
        polys.ImportLegacyFormat(vtk_np.numpy_to_vtkIdTypeArray(cells, deep=True))
        poly.SetPolys(polys)
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(poly)
        normals.SplittingOff()  # keep vertex count/order identical to the file
        normals.Update()
        out = vtk.vtkPolyData()
        out.DeepCopy(normals.GetOutput())
        return out, verts

    def _load_patient_models(self):
        mesh_dir = os.path.join(PATIENT_DIR, "meshes")
        specs = [
            ("GroundTruthSpine", "patient_lumbar_spine_ras.obj", (0.92, 0.89, 0.78), 0.40),
            ("GroundTruthL4", "patient_l4_vertebra_ras.obj", (1.0, 0.88, 0.45), 0.40),
            ("PatientTorso", "patient_torso_skin_ras.obj", (0.86, 0.68, 0.58), 0.24),
        ]
        for name, fname, color, opacity in specs:
            path = os.path.join(mesh_dir, fname)
            if not os.path.exists(path):
                print(f"[!] Missing {path}; run ./.venv/bin/python -m ReconstructionPipeline.data.prepare_patient_assets")
                continue
            poly, verts = self._polydata_from_obj(path)
            self._model_node(name, poly, color, opacity)
            if name == "PatientTorso":
                self.torso_polydata = poly
                n = poly.GetNumberOfPoints()
                if n != len(verts):
                    self._warn_once("skin_count", f"Torso point count {n} != OBJ vertex count {len(verts)}; "
                                                  "skin deformation disabled.")
                    continue
                # Bind ONE persistent float32 buffer to the vtkPoints; per-frame updates write into it.
                self.skin_base = np.ascontiguousarray(verts, dtype=np.float32)
                self.skin_live = self.skin_base.copy()
                self._skin_vtk_array = vtk_np.numpy_to_vtk(self.skin_live, deep=False)  # keep a reference
                poly.GetPoints().SetData(self._skin_vtk_array)

    # ------------------------------------------------------------- networking
    def connect_server(self):
        try:
            self.client = pyigtl.OpenIGTLinkClient(host=self.host, port=self.port)
            self.is_connected = True
            print(f"[+] Connected to OpenIGTLink server at {self.host}:{self.port}")
        except Exception as e:
            self.is_connected = False
            self._warn_once("connect", f"Waiting for server at {self.host}:{self.port} ({e})")

    def on_timer_tick(self):
        now = time.time()
        if not self.is_connected:
            if now - self.last_connect_attempt >= 1.0:
                self.last_connect_attempt = now
                self.connect_server()
            return
        try:
            messages = self.client.get_latest_messages()
        except Exception as e:
            print(f"[!] OpenIGTLink read error: {e}. Reconnecting...")
            self.disconnect_client()
            return

        latest = {}  # get_latest_messages already keeps one message per device; keep the last anyway
        for msg in messages:
            latest[msg.device_name] = msg

        msg = latest.get("UltrasoundProbe")
        if isinstance(msg, pyigtl.TransformMessage):
            m = vtk.vtkMatrix4x4()
            for r in range(4):
                for c in range(4):
                    m.SetElement(r, c, float(msg.matrix[r, c]))
            self.probe_transform_node.SetMatrixTransformToParent(m)

        msg = latest.get("US_Image")
        if isinstance(msg, pyigtl.ImageMessage):
            self._update_us_image(msg)

        msg = latest.get("ReconstructedBone")
        if isinstance(msg, pyigtl.ImageMessage):
            pts = _array_payload(msg, 3)
            if pts is not None:
                self._update_bone_points(pts)

        msg = latest.get("DeformedSkin")
        if isinstance(msg, pyigtl.ImageMessage):
            self._update_skin(msg)

        msg = latest.get("Telemetry")
        if isinstance(msg, pyigtl.StringMessage) and now - self.last_telemetry_log >= 5.0:
            self.last_telemetry_log = now
            try:
                t = json.loads(msg.string)
                phys = t.get("physics", {})
                print(f"[telemetry] F={t.get('normal_force_n', float('nan')):.2f} N, "
                      f"indent={t.get('indentation_mm', float('nan')):.2f} mm, points={t.get('num_points')}, "
                      f"physics={phys.get('achieved_hz', float('nan')):.0f} Hz")
            except (ValueError, TypeError) as e:
                self._warn_once("telemetry", f"Bad Telemetry payload: {e}")

    def _update_us_image(self, msg):
        img = np.asarray(msg.image)
        if img.ndim == 3 and img.shape[2] == 1:
            img = img[:, :, 0]
        # Row 0 = probe face; flip so the face is at the top of the axial view.
        slicer.util.updateVolumeFromArray(self.us_volume_node, np.ascontiguousarray(img[::-1, :])[np.newaxis])
        if self.frames_received < 5:
            lm = slicer.app.layoutManager()
            if lm:
                red = lm.sliceWidget("Red")
                if red:
                    red.mrmlSliceCompositeNode().SetBackgroundVolumeID(self.us_volume_node.GetID())
                    red.sliceController().fitSliceToBackground()
        self.frames_received += 1

    def _update_bone_points(self, pts):
        n = len(pts)
        self._bone_array = np.ascontiguousarray(pts, dtype=np.float32)  # referenced while VTK uses it
        self.bone_points.SetData(vtk_np.numpy_to_vtk(self._bone_array, deep=False))
        cells = np.empty((n, 2), dtype=np.int64)
        cells[:, 0] = 1
        cells[:, 1] = np.arange(n, dtype=np.int64)
        verts = vtk.vtkCellArray()
        verts.ImportLegacyFormat(vtk_np.numpy_to_vtkIdTypeArray(cells.ravel(), deep=True))
        self.bone_polydata.SetVerts(verts)
        self.bone_points.Modified()
        self.bone_polydata.Modified()
        if not self.camera_initialized and n >= 30:
            lm = slicer.app.layoutManager()
            if lm and lm.threeDWidget(0):
                lm.threeDWidget(0).threeDView().resetFocalPoint()
                self.camera_initialized = True

    def _update_skin(self, msg):
        if self.skin_live is None:
            return
        arr = np.asarray(msg.image, dtype=np.float32)
        if arr.ndim >= 2 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        n = len(self.skin_base)
        ncols = arr.shape[-1] if arr.ndim >= 2 else 0
        if ncols == 3 and arr.size == n * 3:  # full deformed vertex positions
            self.skin_live[:] = arr.reshape(n, 3)
            self.skin_deformed = True
        elif ncols == 4:  # sparse [index, dx, dy, dz]; index -1 = reset
            rows = arr.reshape(-1, 4)
            self.skin_live[:] = self.skin_base
            if rows[0, 0] >= 0:
                idx = rows[:, 0].astype(np.int64)
                if idx.max() >= n:
                    self._warn_once("skin_idx", f"DeformedSkin index {idx.max()} out of range ({n} vertices)")
                    return
                self.skin_live[idx] += rows[:, 1:]
            self.skin_deformed = bool(rows[0, 0] >= 0)
        else:
            self._warn_once("skin_shape", f"Unexpected DeformedSkin shape {np.asarray(msg.image).shape}")
            return
        self.torso_polydata.GetPoints().Modified()
        self.torso_polydata.Modified()

    def disconnect_client(self):
        if self.client:
            try:
                self.client.stop()
            except Exception as e:
                self._warn_once("stop", f"Error stopping client: {e}")
            self.client = None
        self.is_connected = False

    def disconnect(self):
        self.timer.stop()
        self.disconnect_client()
        print("[*] Slicer bridge disconnected.")


if __name__ == "__main__":
    bridge = SlicerSurgicalPhysicsBridge()
