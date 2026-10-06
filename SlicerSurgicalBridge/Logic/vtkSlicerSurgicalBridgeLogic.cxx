#include "vtkSlicerSurgicalBridgeLogic.h"
#include "vtkMRMLIGTLConnectorNode.h"
#include "vtkMRMLScene.h"
#include "vtkMRMLModelNode.h"
#include "vtkMRMLTableNode.h"
#include "vtkTable.h"
#include "vtkDoubleArray.h"
#include "vtkPolyData.h"
#include "vtkPoints.h"
#include "vtkDataArray.h"
#include <vtkObjectFactory.h>

vtkStandardNewMacro(vtkSlicerSurgicalBridgeLogic);

vtkSlicerSurgicalBridgeLogic::vtkSlicerSurgicalBridgeLogic()
{
  this->ConnectorNode = nullptr;
  this->TargetModelNode = nullptr;
  this->DeformedAnatomyNode = nullptr;
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

void vtkSlicerSurgicalBridgeLogic::SetTargetModelNode(vtkMRMLModelNode* node)
{
  this->TargetModelNode = node;
}

void vtkSlicerSurgicalBridgeLogic::OnMRMLSceneNodeAdded(vtkMRMLNode* node)
{
  if (!node) return;
  std::string name = node->GetName() ? node->GetName() : "";
  if (name == "DeformedAnatomy") {
    vtkSetAndObserveMRMLNodeMacro(this->DeformedAnatomyNode, node);
  }
}

void vtkSlicerSurgicalBridgeLogic::ProcessMRMLNodesEvents(vtkObject* caller, unsigned long event, void* callData)
{
  if (caller == this->DeformedAnatomyNode && event == vtkCommand::ModifiedEvent) {
    if (!this->TargetModelNode) return;
    vtkPolyData* polyData = this->TargetModelNode->GetPolyData();
    if (!polyData) return;
    vtkPoints* points = polyData->GetPoints();
    if (!points) return;
    
    vtkDoubleArray* dataArray = nullptr;
    if (auto tableNode = vtkMRMLTableNode::SafeDownCast(this->DeformedAnatomyNode)) {
      if (tableNode->GetTable() && tableNode->GetTable()->GetNumberOfColumns() > 0) {
        dataArray = vtkDoubleArray::SafeDownCast(tableNode->GetTable()->GetColumn(0));
      }
    }
    
    if (dataArray && points->GetNumberOfPoints() * 3 == dataArray->GetNumberOfValues()) {
      vtkDataArray* ptsData = points->GetData();
      for (vtkIdType i = 0; i < points->GetNumberOfPoints(); ++i) {
        ptsData->SetComponent(i, 0, dataArray->GetValue(i*3 + 0));
        ptsData->SetComponent(i, 1, dataArray->GetValue(i*3 + 1));
        ptsData->SetComponent(i, 2, dataArray->GetValue(i*3 + 2));
      }
      points->Modified();
      polyData->Modified();
    }
  }
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
