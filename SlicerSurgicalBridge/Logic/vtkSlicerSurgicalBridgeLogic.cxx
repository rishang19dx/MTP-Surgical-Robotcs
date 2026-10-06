#include "vtkSlicerSurgicalBridgeLogic.h"
#include "vtkMRMLIGTLConnectorNode.h"
#include "vtkMRMLScene.h"
#include <vtkObjectFactory.h>

vtkStandardNewMacro(vtkSlicerSurgicalBridgeLogic);

vtkSlicerSurgicalBridgeLogic::vtkSlicerSurgicalBridgeLogic()
{
  this->ConnectorNode = nullptr;
}

vtkSlicerSurgicalBridgeLogic::~vtkSlicerSurgicalBridgeLogic()
{
}

void vtkSlicerSurgicalBridgeLogic::SetMRMLSceneInternal(vtkMRMLScene* newScene)
{
  vtkNew<vtkIntArray> events;
  events->InsertNextValue(vtkMRMLScene::NodeAddedEvent);
  events->InsertNextValue(vtkMRMLScene::NodeRemovedEvent);
  events->InsertNextValue(vtkMRMLScene::EndBatchProcessEvent);
  this->SetAndObserveMRMLSceneEventsInternal(newScene, events.GetPointer());
}

void vtkSlicerSurgicalBridgeLogic::RegisterNodes()
{
  if (!this->GetMRMLScene()) return;
}

void vtkSlicerSurgicalBridgeLogic::UpdateFromMRMLScene()
{
}

void vtkSlicerSurgicalBridgeLogic::ConnectToPhysicsServer(const std::string& host, int port)
{
  if (!this->GetMRMLScene()) return;
  
  if (!this->ConnectorNode) {
    this->ConnectorNode = vtkMRMLIGTLConnectorNode::SafeDownCast(
      this->GetMRMLScene()->AddNewNodeByClass("vtkMRMLIGTLConnectorNode", "PhysicsServerConnector")
    );
  }
  
  if (this->ConnectorNode) {
    this->ConnectorNode->SetTypeClient(host.c_str(), port);
    this->ConnectorNode->Start();
  }
}

void vtkSlicerSurgicalBridgeLogic::SendRobotCommand(const std::string& q_des)
{
  // Implement OpenIGTLink message sending logic here (e.g. String message or NDArray message)
  // For sending IGTL string messages through SlicerOpenIGTLinkIF, we can push to an outgoing queue.
  // vtkMRMLIGTLConnectorNode has methods to register outgoing messages.
}
