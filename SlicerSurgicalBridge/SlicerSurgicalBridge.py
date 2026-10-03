import os
import qt
import vtk
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import json
import logging

class SlicerSurgicalBridge(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Surgical Physics Bridge"
        self.parent.categories = ["Simulation"]
        self.parent.dependencies = []
        self.parent.contributors = ["Antigravity"]
        self.parent.helpText = "Connects Slicer to an external physics engine (PyBullet/MuJoCo) via ZeroMQ."
        self.parent.acknowledgementText = "MTP Project."

class SlicerSurgicalBridgeWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.layout.addWidget(qt.QLabel("Surgical Physics Bridge Settings"))
        
        self.connectButton = qt.QPushButton("Connect to Physics Server")
        self.connectButton.toolTip = "Start receiving state from ZMQ server."
        self.layout.addWidget(self.connectButton)
        self.connectButton.connect('clicked(bool)', self.onConnectButtonClicked)
        
        # Command UI
        commandLayout = qt.QHBoxLayout()
        self.qDesInput = qt.QLineEdit("0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0")
        self.qDesInput.toolTip = "Comma-separated target joint positions (q_des)"
        commandLayout.addWidget(qt.QLabel("q_des:"))
        commandLayout.addWidget(self.qDesInput)
        
        self.sendCmdButton = qt.QPushButton("Send Command")
        commandLayout.addWidget(self.sendCmdButton)
        self.sendCmdButton.connect('clicked(bool)', self.onSendCommandClicked)
        
        self.layout.addLayout(commandLayout)
        
        self.layout.addStretch(1)
        self.logic = SlicerSurgicalBridgeLogic()

    def onSendCommandClicked(self):
        if self.logic and self.logic.is_connected():
            try:
                q_des_str = self.qDesInput.text
                q_des = [float(x.strip()) for x in q_des_str.split(',')]
                self.logic.send_command(q_des)
            except Exception as e:
                logging.error(f"Invalid q_des format: {e}")
        else:
            logging.warning("Not connected to physics server.")

    def onConnectButtonClicked(self):
        if not self.logic.is_connected():
            self.logic.connect_zmq("tcp://127.0.0.1:5555")
            self.connectButton.text = "Disconnect"
        else:
            self.logic.disconnect_zmq()
            self.connectButton.text = "Connect to Physics Server"

class SlicerSurgicalBridgeLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.zmq_context = None
        self.zmq_socket = None
        self.zmq_cmd_socket = None
        self.timer = qt.QTimer()
        self.timer.setInterval(20) # 50 Hz UI refresh polling
        self.timer.connect('timeout()', self.poll_zmq)
        
        self.transform_nodes = {} # maps link name to MRML LinearTransformNode

    def is_connected(self):
        return self.zmq_socket is not None

    def connect_zmq(self, address="tcp://127.0.0.1:5555"):
        try:
            import zmq
        except ImportError:
            slicer.util.pip_install('pyzmq')
            import zmq
            
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_socket.connect(address)
        self.zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "STATE")
        
        # Don't block the UI
        self.zmq_socket.setsockopt(zmq.RCVTIMEO, 0) 
        
        # Command socket
        base_address, port_str = address.rsplit(':', 1)
        cmd_address = f"{base_address}:{int(port_str) + 1}"
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PUB)
        self.zmq_cmd_socket.connect(cmd_address)
        
        self.timer.start()
        logging.info(f"Connected to physics server at {address}")

    def disconnect_zmq(self):
        self.timer.stop()
        if self.zmq_socket:
            self.zmq_socket.close()
            self.zmq_socket = None
        if self.zmq_cmd_socket:
            self.zmq_cmd_socket.close()
            self.zmq_cmd_socket = None
        if self.zmq_context:
            self.zmq_context.term()
            self.zmq_context = None
        logging.info("Disconnected from physics server")

    def send_command(self, q_des):
        if self.zmq_cmd_socket:
            msg = {"q_des": q_des}
            self.zmq_cmd_socket.send_string("CMD " + json.dumps(msg))

    def poll_zmq(self):
        if not self.zmq_socket:
            return
            
        import zmq
        try:
            # Drain the queue to get the latest state
            latest_msg = None
            while True:
                try:
                    msg = self.zmq_socket.recv_string(flags=zmq.NOBLOCK)
                    latest_msg = msg
                except zmq.Again:
                    break # Queue is empty
                    
            if latest_msg:
                # Format is "STATE {\"timestamp\": ..., \"poses\": ...}"
                json_str = latest_msg[6:]
                state = json.loads(json_str)
                self.update_mrml_transforms(state["poses"])
                
        except Exception as e:
            logging.error(f"Error polling ZMQ: {e}")

    def update_mrml_transforms(self, poses):
        for link_name, pose_matrix in poses.items():
            if link_name not in self.transform_nodes:
                # Create a new transform node
                transformNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLinearTransformNode', link_name)
                self.transform_nodes[link_name] = transformNode
            else:
                transformNode = self.transform_nodes[link_name]
                
            # pose_matrix is a 4x4 nested list
            vtk_matrix = vtk.vtkMatrix4x4()
            for i in range(4):
                for j in range(4):
                    vtk_matrix.SetElement(i, j, pose_matrix[i][j])
                    
            transformNode.SetMatrixTransformToParent(vtk_matrix)
