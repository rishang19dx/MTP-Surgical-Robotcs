"""
3D Slicer client for the deformation demos (run inside Slicer):

    MTP_DEMO_SCENE=demos/scenes/liver Slicer --python-script demos/slicer_demo.py

It connects to `demos/run_demo.py` over OpenIGTLink (port 18945) and shows
  * 3D view: the whole body (skin, bones, organs), cut open at the probe's image plane. Deforming meshes keep
    their own colour where they do not move and turn yellow -> orange -> red -> purple with |u| (mm); the probe;
    and the live B-mode drawn in its true plane inside the body;
  * Red view: the B-mode, which changes with the pressure;
  * Plot: force (N) and indentation (mm) over the last frames;
  * corner text: scene, force, indentation, max displacement, magnification.
Vertex updates are in place (numpy_to_vtk(deep=False) once, write + Modified(); AGENTS.md invariant 3).

Environment: MTP_DEMO_SCENE (required), MTP_DEMO_PORT (18945), MTP_DEMO_MAGNIFY (1), MTP_DEMO_SNAPSHOT_DIR,
MTP_DEMO_SNAPSHOT_S (0.2), MTP_DEMO_EXIT_AFTER_S (0 = never), MTP_DEMO_DISP_RANGE (3 mm), MTP_DEMO_CAM_SIDE (1),
MTP_DEMO_CAM_DIST (1.0, camera distance scale).
Offline mode: MTP_DEMO_OFFLINE=<frames.npz> (written by `run_demo.py --cache`) renders every cached frame in turn,
saves MTP_DEMO_SNAPSHOT_DIR/slicer_<frame>.png and exits; this gives frame-exact video frames.
Messages: US (uint8 image), Probe (transform scene <- face), V_<mesh> (float32 (1,N,3) deformed vertices),
D_<mesh> (float32 (1,N) |u| mm), Telemetry (JSON string with "frame", "force_n", "indent_mm", "max_u_mm", ...).
"""

import json
import os
import sys
import time

import numpy as np

import qt
import slicer
import vtk
import vtk.util.numpy_support as vtk_np

try:
    import pyigtl
except ImportError:
    slicer.util.pip_install("pyigtl")
    import pyigtl

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SCENE = os.path.join(ROOT, os.environ.get("MTP_DEMO_SCENE", "demos/scenes/liver"))
PORT = int(os.environ.get("MTP_DEMO_PORT", "18945"))
MAG = float(os.environ.get("MTP_DEMO_MAGNIFY", "1"))
SNAP_DIR = os.environ.get("MTP_DEMO_SNAPSHOT_DIR", "")
SNAP_S = float(os.environ.get("MTP_DEMO_SNAPSHOT_S", "0.2"))
EXIT_AFTER = float(os.environ.get("MTP_DEMO_EXIT_AFTER_S", "0"))
US_HW, US_MM = (200, 150), (75.0, 50.0)
LAYOUT_ID = 611
DISP_RANGE_MM = float(os.environ.get("MTP_DEMO_DISP_RANGE", "3.0"))
OFFLINE = os.environ.get("MTP_DEMO_OFFLINE", "")
CAM_DIST = float(os.environ.get("MTP_DEMO_CAM_DIST", "1.0"))
RAMP = [(0.0, None), (0.06, None), (0.25, (1.0, 0.92, 0.35)), (0.5, (1.0, 0.55, 0.12)), (0.75, (0.88, 0.15, 0.15)),
        (1.0, (0.55, 0.0, 0.55))]                                   # fraction of DISP_RANGE_MM -> colour (None = base)


def ramp_table(name, base):
    """Colour table: the tissue's own colour below 6 % of the range (no visible motion), then a heat ramp."""
    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode", f"disp_{name}")
    node.SetTypeToUser()
    n = 256
    node.SetNumberOfColors(n)
    stops = [(f, base if c is None else c) for f, c in RAMP]
    for i in range(n):
        x = i / (n - 1)
        for (f0, c0), (f1, c1) in zip(stops, stops[1:]):
            if x <= f1:
                w = 0.0 if f1 == f0 else (x - f0) / (f1 - f0)
                c = [c0[k] + w * (c1[k] - c0[k]) for k in range(3)]
                break
        node.SetColor(i, f"{i}", c[0], c[1], c[2], 1.0)
    node.SetNamesInitialised(True)
    return node


def read_obj(path):
    v, f = [], []
    for line in open(path):
        if line.startswith("v "):
            v.append([float(a) for a in line.split()[1:4]])
        elif line.startswith("f "):
            f.append([int(a.split("/")[0]) - 1 for a in line.split()[1:4]])
    return np.asarray(v, np.float32), np.asarray(f, np.int64)


def polydata(v, f):
    pd = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    arr = np.ascontiguousarray(v, dtype=np.float32)
    pts.SetData(vtk_np.numpy_to_vtk(arr, deep=False))
    pd.SetPoints(pts)
    cells = vtk.vtkCellArray()
    conn = np.hstack([np.full((len(f), 1), 3, np.int64), f]).ravel()
    cells.SetCells(len(f), vtk_np.numpy_to_vtkIdTypeArray(conn, deep=True))
    pd.SetPolys(cells)
    return pd, arr


class DemoClient:
    def __init__(self):
        self.sc = json.load(open(os.path.join(SCENE, "scene.json")))
        self.client, self.connected, self.t0 = None, False, time.time()
        self.last_try = self.last_snap = 0.0
        self.snap_i, self.frames = 0, 0
        self.tel = {}
        self.hist = []
        slicer.mrmlScene.Clear(0)
        mw = slicer.util.mainWindow()
        for fn in ("setToolbarsVisible", "setMenuBarsVisible", "setPythonConsoleVisible", "setStatusBarVisible",
                   "setDataProbeVisible"):
            try:
                getattr(slicer.util, fn)(False)
            except Exception:
                pass
        try:
            mw.findChild(qt.QDockWidget, "PanelDockWidget").hide()
        except Exception:
            pass
        mw.resize(1600, 900)
        mw.showMaximized()
        self.camera_set = False
        self._layout()
        self._scene_transform()
        self._meshes()
        self._probe_and_us()
        self._plot()
        self.timer = vtk.vtkTimerLog()
        self.qt_timer = qt.QTimer()
        if OFFLINE:
            self._load_offline()
            self.qt_timer.timeout.connect(self.offline_tick)
            self.qt_timer.start(10)
        else:
            self.qt_timer.timeout.connect(self.tick)
            self.qt_timer.start(33)

    # ------------------------------------------------------------ setup
    def _layout(self):
        xml = ('<layout type="horizontal" split="true">'
               ' <item splitSize="640"><view class="vtkMRMLViewNode" singletontag="1">'
               '  <property name="viewlabel" action="default">3D</property></view></item>'
               ' <item splitSize="360"><layout type="vertical" split="true">'
               '  <item splitSize="600"><view class="vtkMRMLSliceNode" singletontag="Red">'
               '   <property name="orientation" action="default">Axial</property>'
               '   <property name="viewlabel" action="default">US</property>'
               '   <property name="viewcolor" action="default">#2F9BD6</property></view></item>'
               '  <item splitSize="300"><view class="vtkMRMLPlotViewNode" singletontag="PlotView1">'
               '   <property name="viewlabel" action="default">P</property></view></item>'
               ' </layout></item></layout>')
        lm = slicer.app.layoutManager()
        ln = lm.layoutLogic().GetLayoutNode()
        ln.AddLayoutDescription(LAYOUT_ID, xml)
        ln.SetViewArrangement(LAYOUT_ID)
        vn = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLViewNode")
        vn.SetBackgroundColor(0.05, 0.06, 0.08)
        vn.SetBackgroundColor2(0.13, 0.15, 0.19)
        vn.SetBoxVisible(False)
        vn.SetAxisLabelsVisible(False)

    def _scene_transform(self):
        tr = self.sc["to_ras"]
        R, c, t = np.asarray(tr["R"]), np.asarray(tr["c_ras"]), np.asarray(tr["t"])
        M = np.eye(4)
        M[:3, :3] = R.T
        M[:3, 3] = c - R.T @ t                                      # x_ras = R^T (x_scene - t) + c
        self.scene_to_ras = M
        self.tf = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "SceneToRAS")
        self.tf.SetMatrixTransformToParent(slicer.util.vtkMatrixFromArray(M))

    def _model(self, name, pd, color, opacity, parent):
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        node.SetAndObservePolyData(pd)
        node.CreateDefaultDisplayNodes()
        d = node.GetDisplayNode()
        d.SetColor(*color)
        d.SetOpacity(opacity)
        d.SetBackfaceCulling(False)
        d.SetAmbient(0.25)
        d.SetDiffuse(0.85)
        d.SetSpecular(0.15)
        node.SetAndObserveTransformNodeID(parent.GetID())
        return node

    def _meshes(self):
        self.live = {}
        self._clipped = []
        for m in self.sc["meshes"]:
            v, f = read_obj(os.path.join(SCENE, m["file"]))
            pd, arr = polydata(v, f)
            op = 0.75 if m["name"] == "skin" else (1.0 if m.get("focus") or m["category"] == "bone" else 0.7)
            node = self._model(m["name"], pd, m["color"], op, self.tf)
            self._clipped.append(node.GetDisplayNode())             # live cutaway at the image plane
            if m.get("deforms"):
                disp = np.zeros(len(v), np.float32)
                da = vtk_np.numpy_to_vtk(disp, deep=False)
                da.SetName("disp_mm")
                pd.GetPointData().AddArray(da)
                pd.GetPointData().SetActiveScalars("disp_mm")
                d = node.GetDisplayNode()
                d.SetActiveScalarName("disp_mm")
                d.SetAndObserveColorNodeID(ramp_table(m["name"], m["color"]).GetID())
                d.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseManualScalarRange)
                d.SetScalarRange(0.0, DISP_RANGE_MM)
                d.SetScalarVisibility(True)
                self.live[m["name"]] = {"pd": pd, "arr": arr, "base": v.copy(), "disp": disp}
                if m["name"] == "skin" and os.environ.get("MTP_DEMO_GHOST", "0") == "1":   # undeformed ghost
                    gpd, _ = polydata(v.copy(), f)
                    g = self._model("skin (undeformed)", gpd, (0.8, 0.85, 0.9), 0.12, self.tf)
                    g.GetDisplayNode().SetRepresentation(1)        # wireframe
        try:                                                         # small legend on the skin's colour table
            node = slicer.util.getNode("skin")
            leg = slicer.modules.colors.logic().AddDefaultColorLegendDisplayNode(node.GetDisplayNode())
            leg.SetTitleText("|u| mm")
            leg.SetLabelFormat("%.1f")
            leg.SetNumberOfLabels(4)
            leg.SetMaxNumberOfColors(64)
            leg.SetSize(0.06, 0.32)
            leg.SetPosition(0.97, 0.06)
            leg.GetTitleTextProperty().SetFontSize(14)
            leg.GetLabelTextProperty().SetFontSize(12)
            leg.SetViewNodeIDs([slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLViewNode").GetID()])
        except Exception as e:
            print("[slicer] legend:", e, flush=True)
        lm = slicer.app.layoutManager()
        lm.threeDWidget(0).threeDView().resetFocalPoint()
        # cutaway: a hidden plane that follows the probe's image plane clips skin and soft tissue (bones stay
        # whole); the cut faces are capped so the cross-section reads as solid tissue
        self.cut = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsPlaneNode", "image plane cut")
        self.cut.CreateDefaultDisplayNodes()
        self.cut.GetDisplayNode().SetVisibility(False)
        self.cut.SetLocked(True)
        clip = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLClipNode", "cutaway")
        clip.SetClipType(clip.ClipUnion)
        clip.AddAndObserveClippingNodeID(self.cut.GetID())
        side = os.environ.get("MTP_DEMO_CAM_SIDE", "1") != "-1"
        clip.SetNthClippingNodeState(0, clip.ClipNegativeSpace if side else clip.ClipPositiveSpace)   # remove the camera half
        for d in self._clipped:
            d.SetAndObserveClipNodeID(clip.GetID())
            d.SetClipping(True)
            d.SetClippingCapSurface(True)
            d.SetClippingOutline(True)
            d.SetClippingCapOpacity(0.18 if d.GetDisplayableNode().GetName() == "skin" else 1.0)   # skin cap = fat
        self.clip = clip

    def _probe_and_us(self):
        self.probe_tf = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "ProbeToScene")
        self.probe_tf.SetAndObserveTransformNodeID(self.tf.GetID())
        cube = vtk.vtkCubeSource()
        cube.SetXLength(45)
        cube.SetYLength(60)
        cube.SetZLength(10)
        cube.SetCenter(0, -30, 0)                                   # face at y = 0, body above it (-beam)
        cube.Update()
        pr = self._model("probe", cube.GetOutput(), (0.15, 0.55, 0.95), 0.45, self.probe_tf)   # see the dent
        pr.GetDisplayNode().SetEdgeVisibility(True)
        # (1) 2D B-mode in the Red view: its own volume, not transformed; depth points down on screen
        vol = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "B-mode")
        lat, ax = US_MM[1] / (US_HW[1] - 1), US_MM[0] / (US_HW[0] - 1)
        vol.SetSpacing(lat, ax, 1.0)
        vol.SetOrigin(-0.5 * US_MM[1], 0.0, 0.0)
        M = vtk.vtkMatrix4x4()
        M.SetElement(1, 1, -1.0)                                     # +J (depth) -> -A, i.e. down in the axial view
        vol.SetIJKToRASDirectionMatrix(M)
        slicer.util.updateVolumeFromArray(vol, np.zeros((1, US_HW[0], US_HW[1]), np.uint8))
        vol.CreateDefaultDisplayNodes()
        dn = vol.GetDisplayNode()
        dn.SetAutoWindowLevel(False)
        dn.SetWindowLevel(255, 127)
        dn.SetInterpolate(True)
        self.us = vol
        red = slicer.app.layoutManager().sliceWidget("Red")
        red.mrmlSliceCompositeNode().SetBackgroundVolumeID(vol.GetID())
        self.red_widget = red
        self.red = red.mrmlSliceNode()
        self.red.SetOrientationToAxial()
        self.red.SetSliceVisible(False)
        red.sliceController().fitSliceToBackground()
        # (2) the same image on a textured plane at the probe's image plane, inside the body in the 3D view
        plane = vtk.vtkPlaneSource()
        zoff = 0.8 * float(os.environ.get("MTP_DEMO_CAM_SIDE", "1"))   # just off the cut plane, on the camera side
        plane.SetOrigin(-0.5 * US_MM[1], 0.0, zoff)
        plane.SetPoint1(0.5 * US_MM[1], 0.0, zoff)
        plane.SetPoint2(-0.5 * US_MM[1], US_MM[0], zoff)
        plane.Update()
        self.tex = vtk.vtkImageData()
        self.tex.SetDimensions(US_HW[1], US_HW[0], 1)
        self.tex.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        self.tex_arr = vtk_np.vtk_to_numpy(self.tex.GetPointData().GetScalars())
        self.tex_prod = vtk.vtkTrivialProducer()
        self.tex_prod.SetOutput(self.tex)
        pn = self._model("B-mode plane", plane.GetOutput(), (1, 1, 1), 1.0, self.probe_tf)
        d = pn.GetDisplayNode()
        d.SetTextureImageDataConnection(self.tex_prod.GetOutputPort())
        d.SetAmbient(1.0)
        d.SetDiffuse(0.0)
        d.SetScalarVisibility(False)

    def _plot(self):
        self.table = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "telemetry")
        t = self.table.GetTable()
        for name in ("frame", "force (N)", "indentation (mm)", "max tissue displacement (mm)"):
            a = vtk.vtkFloatArray()
            a.SetName(name)
            t.AddColumn(a)
        chart = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode", "pressure")
        for col, rgb in (("force (N)", (1.0, 0.75, 0.2)), ("indentation (mm)", (0.3, 0.75, 1.0)),
                         ("max tissue displacement (mm)", (0.95, 0.35, 0.3))):
            s = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", col)
            s.SetAndObserveTableNodeID(self.table.GetID())
            s.SetXColumnName("frame")
            s.SetYColumnName(col)
            s.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeLine)
            s.SetColor(*rgb)
            s.SetLineWidth(3)
            s.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleNone)
            chart.AddAndObservePlotSeriesNodeID(s.GetID())
        chart.SetTitle("pressure, indentation and tissue displacement")
        chart.SetXAxisTitle("frame")
        pv = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLPlotViewNode")
        pv.SetPlotChartNodeID(chart.GetID())
        self.chart = chart

    # ------------------------------------------------------------ loop
    def tick(self):
        now = time.time()
        if EXIT_AFTER > 0 and now - self.t0 > EXIT_AFTER:
            slicer.util.exit(0)
            return
        if not self.connected:
            if now - self.last_try > 1.0:
                self.last_try = now
                try:
                    self.client = pyigtl.OpenIGTLinkClient(host="127.0.0.1", port=PORT)
                    self.connected = True
                except Exception:
                    pass
            return
        try:
            msgs = {m.device_name: m for m in self.client.get_latest_messages()}
        except Exception:
            self.connected = False
            return
        if not msgs:
            return
        m = msgs.get("Probe")
        if m is not None:
            self.probe_tf.SetMatrixTransformToParent(slicer.util.vtkMatrixFromArray(np.asarray(m.matrix, float)))
            self._cutplane(np.asarray(m.matrix, float))
            if not self.camera_set:
                self._camera(np.asarray(m.matrix, float))
        m = msgs.get("US")
        if m is not None:
            img = np.asarray(m.image)
            img = np.ascontiguousarray(img[..., 0] if img.ndim == 3 else img).astype(np.uint8)
            slicer.util.updateVolumeFromArray(self.us, img[np.newaxis])
            self.tex_arr[:] = img.ravel()                                # row 0 = probe face = t 0
            self.tex.Modified()
            if self.frames < 3:
                self.red_widget.sliceController().fitSliceToBackground()
        for name, L in self.live.items():
            mv = msgs.get(f"V_{name}")
            if mv is not None:
                a = np.asarray(mv.image, np.float32).reshape(-1, 3)
                if len(a) == len(L["base"]):
                    L["arr"][:] = L["base"] + MAG * (a - L["base"])
                    L["pd"].GetPoints().Modified()
            md = msgs.get(f"D_{name}")
            if md is not None:
                d = np.asarray(md.image, np.float32).ravel()
                if len(d) == len(L["disp"]):
                    L["disp"][:] = d
                    L["pd"].GetPointData().GetArray("disp_mm").Modified()
            L["pd"].Modified()
        m = msgs.get("Telemetry")
        if m is not None:
            try:
                self.tel = json.loads(m.string)
                self._telemetry()
            except (ValueError, TypeError):
                pass
        self.frames += 1
        if SNAP_DIR and now - self.last_snap >= SNAP_S:
            self.last_snap = now
            os.makedirs(SNAP_DIR, exist_ok=True)
            f = self.tel.get("frame", -1)
            slicer.app.layoutManager().viewport().grab().save(os.path.join(SNAP_DIR, f"slicer_{self.snap_i:05d}_f{f}.png"))
            self.snap_i += 1

    # ------------------------------------------------------------ offline (frame-exact) rendering
    def _load_offline(self):
        z = np.load(OFFLINE)
        self.off = {k: z[k] for k in z.files}
        self.off_meta = json.loads(str(self.off["meta"]))
        sel = os.environ.get("MTP_DEMO_OFFLINE_FRAMES", "")
        self.off_list = [int(v) for v in sel.split(",")] if sel else list(range(len(self.off["t"])))
        self.off_k, self.off_warm = 0, 0
        print(f"[slicer] offline: {len(self.off['t'])} frames from {OFFLINE}", flush=True)

    def offline_tick(self):
        o = self.off
        if self.off_k >= len(self.off_list):
            self.qt_timer.stop()
            slicer.util.exit(0)
            return
        k = self.off_list[self.off_k]
        T = o["pose_face"][k]
        self.probe_tf.SetMatrixTransformToParent(slicer.util.vtkMatrixFromArray(T))
        self._cutplane(T)
        if not self.camera_set:
            self._camera(T)
        img = np.ascontiguousarray(o["bmode"][k])
        slicer.util.updateVolumeFromArray(self.us, img[np.newaxis])
        self.tex_arr[:] = img.ravel()
        self.tex.Modified()
        for name, L in self.live.items():
            if f"idx__{name}" not in o:
                continue
            idx = o[f"idx__{name}"]
            L["arr"][idx] = L["base"][idx] + MAG * (o[f"v__{name}"][k] - L["base"][idx])
            L["disp"][idx] = o[f"d__{name}"][k].astype(np.float32)
            L["pd"].GetPoints().Modified()
            L["pd"].GetPointData().GetArray("disp_mm").Modified()
            L["pd"].Modified()
        self.tel = {"frame": k, "t_s": float(o["t"][k]), "force_n": float(o["force_n"][k]),
                    "indent_mm": float(o["indent_mm"][k]), "max_u_mm": float(o["max_u_mm"][k])}
        self._telemetry()
        if self.off_warm < 3:                                            # let the first frames settle the layout
            self.red_widget.sliceController().fitSliceToBackground()
            self.off_warm += 1
            return
        slicer.app.processEvents()
        slicer.app.layoutManager().threeDWidget(0).threeDView().forceRender()
        slicer.app.layoutManager().sliceWidget("Red").sliceView().forceRender()
        if SNAP_DIR:
            os.makedirs(SNAP_DIR, exist_ok=True)
            slicer.app.layoutManager().viewport().grab().save(os.path.join(SNAP_DIR, f"slicer_{k:04d}.png"))
        self.off_k += 1

    def _cutplane(self, T):
        M = self.scene_to_ras @ T
        self.cut.SetOriginWorld(M[:3, 3])
        self.cut.SetNormalWorld(M[:3, 2] / np.linalg.norm(M[:3, 2]))

    def _camera(self, T):
        """Oblique close-up from the probe's side: focal point 30 mm under the face, camera back along -beam and
        off to the side, so the skin colour map and the B-mode plane are both visible."""
        M = self.scene_to_ras @ T
        lat, beam, elev = M[:3, 0], M[:3, 1], M[:3, 2]
        focal = (M @ np.array([0.0, 30.0, 0.0, 1.0]))[:3]
        side = float(os.environ.get("MTP_DEMO_CAM_SIDE", "1"))      # which side of the cut the camera looks from
        pos = focal + CAM_DIST * (-150 * beam + side * 300 * elev + 90 * lat)
        cam = slicer.app.layoutManager().threeDWidget(0).threeDView().cameraNode()
        cam.SetFocalPoint(*focal)
        cam.SetPosition(*pos)
        cam.SetViewUp(*(-beam))
        cam.GetCamera().SetViewAngle(40)
        cam.ResetClippingRange()
        self.camera_set = True

    def _align_red(self, T):
        M = self.scene_to_ras @ T
        n, t = M[:3, 2], M[:3, 0]
        p = (M @ np.array([0.0, US_MM[0] / 2, 0.0, 1.0]))[:3]
        self.red.SetSliceToRASByNTP(*n, *t, *p, 0)

    def _telemetry(self):
        t = self.tel
        txt = (f"{self.sc['title']}\nforce {t.get('force_n', 0):.1f} N   indentation {t.get('indent_mm', 0):.1f} mm\n"
               f"max tissue displacement {t.get('max_u_mm', 0):.2f} mm" + (f"   (shown x{MAG:g})" if MAG != 1 else ""))
        v = slicer.app.layoutManager().threeDWidget(0).threeDView()
        v.cornerAnnotation().SetText(2, txt)
        v.cornerAnnotation().GetTextProperty().SetColor(1, 1, 1)
        v.forceRender()
        tb = self.table.GetTable()
        self.hist.append((t.get("frame", 0), t.get("force_n", 0), t.get("indent_mm", 0), t.get("max_u_mm", 0)))
        self.hist = self.hist[-150:]
        tb.SetNumberOfRows(len(self.hist))
        for i, row in enumerate(self.hist):
            for j, val in enumerate(row):
                tb.SetValue(i, j, float(val))
        tb.Modified()
        self.table.Modified()
        f = self.hist[-1][0]
        self.chart.SetXAxisRangeAuto(False)
        self.chart.SetXAxisRange(max(0, f - 150), max(150, f))
        self.chart.SetYAxisRangeAuto(False)
        self.chart.SetYAxisRange(0, max(7.0, 1.1 * max(max(r[1:]) for r in self.hist)))
        self.chart.SetXAxisTitle("frame (20 per second)")


demo = DemoClient()
