#include "qSlicerSurgicalBridgeModule.h"
#include "qSlicerSurgicalBridgeModuleWidget.h"
#include "vtkSlicerSurgicalBridgeLogic.h"

class qSlicerSurgicalBridgeModulePrivate
{
public:
  qSlicerSurgicalBridgeModulePrivate();
};

qSlicerSurgicalBridgeModulePrivate::qSlicerSurgicalBridgeModulePrivate() {}

qSlicerSurgicalBridgeModule::qSlicerSurgicalBridgeModule(QObject* _parent)
  : Superclass(_parent), d_ptr(new qSlicerSurgicalBridgeModulePrivate) {}

qSlicerSurgicalBridgeModule::~qSlicerSurgicalBridgeModule() {}

QString qSlicerSurgicalBridgeModule::helpText() const { return "Surgical Physics Bridge via OpenIGTLink."; }
QString qSlicerSurgicalBridgeModule::acknowledgementText() const { return "MTP Project."; }
QStringList qSlicerSurgicalBridgeModule::contributors() const { return QStringList() << "Antigravity"; }
QStringList qSlicerSurgicalBridgeModule::categories() const { return QStringList() << "Simulation"; }
QStringList qSlicerSurgicalBridgeModule::dependencies() const { return QStringList() << "OpenIGTLinkIF"; }

void qSlicerSurgicalBridgeModule::setup() {
  this->Superclass::setup();
}

qSlicerAbstractModuleRepresentation* qSlicerSurgicalBridgeModule::createWidgetRepresentation() {
  return new qSlicerSurgicalBridgeModuleWidget;
}

vtkMRMLAbstractLogic* qSlicerSurgicalBridgeModule::createLogic() {
  return vtkSlicerSurgicalBridgeLogic::New();
}
