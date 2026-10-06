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
#include "vtkCellArray.h"
#include "vtkLine.h"
#include <vtkObjectFactory.h>

vtkStandardNewMacro(vtkSlicerSurgicalBridgeLogic);

vtkSlicerSurgicalBridgeLogic::vtkSlicerSurgicalBridgeLogic()
{
  this->ConnectorNode = nullptr;
  this->TargetModelNode = nullptr;
  this->ForceModelNode = nullptr;
  this->DeformedAnatomyNode = nullptr;
  this->ContactForcesNode = nullptr;
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

void vtkSlicerSurgicalBridgeLogic::SetForceModelNode(vtkMRMLModelNode* node)
{
  this->ForceModelNode = node;
}

void vtkSlicerSurgicalBridgeLogic::OnMRMLSceneNodeAdded(vtkMRMLNode* node)
{
  if (!node) return;
  std::string name = node->GetName() ? node->GetName() : "";
  if (name == "DeformedAnatomy") {
    vtkSetAndObserveMRMLNodeMacro(this->DeformedAnatomyNode, node);
  } else if (name == "ContactForces") {
    vtkSetAndObserveMRMLNodeMacro(this->ContactForcesNode, node);
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
  } else if (caller == this->ContactForcesNode && event == vtkCommand::ModifiedEvent) {
    if (!this->ForceModelNode) return;
    vtkDoubleArray* dataArray = nullptr;
    if (auto tableNode = vtkMRMLTableNode::SafeDownCast(this->ContactForcesNode)) {
      if (tableNode->GetTable() && tableNode->GetTable()->GetNumberOfColumns() > 0) {
        dataArray = vtkDoubleArray::SafeDownCast(tableNode->GetTable()->GetColumn(0));
      }
    }
    
    if (dataArray) {
      vtkNew<vtkPolyData> polyData;
      vtkNew<vtkPoints> points;
      vtkNew<vtkCellArray> lines;
      
      int numContacts = dataArray->GetNumberOfValues() / 7;
      for (int i = 0; i < numContacts; ++i) {
        double p[3] = { dataArray->GetValue(i*7 + 0), dataArray->GetValue(i*7 + 1), dataArray->GetValue(i*7 + 2) };
        double n[3] = { dataArray->GetValue(i*7 + 3), dataArray->GetValue(i*7 + 4), dataArray->GetValue(i*7 + 5) };
        double f = dataArray->GetValue(i*7 + 6);
        
        // Only draw non-zero forces
        if (f > 1e-4) {
          // Scale factor for visualization, assuming PyBullet in meters, Slicer in mm
          // So forces might need significant scaling to be visible. Let's use 10.0 * f
          double scale = 10.0 * f; 
          double p2[3] = { p[0] + n[0]*scale, p[1] + n[1]*scale, p[2] + n[2]*scale };
          
          vtkIdType pid1 = points->InsertNextPoint(p);
          vtkIdType pid2 = points->InsertNextPoint(p2);
          
          vtkNew<vtkLine> line;
          line->GetPointIds()->SetId(0, pid1);
          line->GetPointIds()->SetId(1, pid2);
          lines->InsertNextCell(line);
        }
      }
      
      polyData->SetPoints(points);
      polyData->SetLines(lines);
      this->ForceModelNode->SetAndObservePolyData(polyData);
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
