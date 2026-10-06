#ifndef __vtkSlicerSurgicalBridgeLogic_h
#define __vtkSlicerSurgicalBridgeLogic_h

#include "vtkSlicerModuleLogic.h"
#include <string>

class vtkMRMLIGTLConnectorNode;

class vtkSlicerSurgicalBridgeLogic : public vtkSlicerModuleLogic
{
public:
  static vtkSlicerSurgicalBridgeLogic *New();
  vtkTypeMacro(vtkSlicerSurgicalBridgeLogic, vtkSlicerModuleLogic);
  
  void ConnectToPhysicsServer(const std::string& host, int port);
  void SendRobotCommand(const std::string& q_des);
  
protected:
  vtkSlicerSurgicalBridgeLogic();
  ~vtkSlicerSurgicalBridgeLogic() override;
  
  void SetMRMLSceneInternal(vtkMRMLScene* newScene) override;
  void RegisterNodes() override;
  void UpdateFromMRMLScene() override;
  
private:
  vtkSlicerSurgicalBridgeLogic(const vtkSlicerSurgicalBridgeLogic&);
  void operator=(const vtkSlicerSurgicalBridgeLogic&);
  
  vtkMRMLIGTLConnectorNode* ConnectorNode;
};
#endif
