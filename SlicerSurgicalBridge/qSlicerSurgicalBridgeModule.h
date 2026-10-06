#ifndef __qSlicerSurgicalBridgeModule_h
#define __qSlicerSurgicalBridgeModule_h

#include "qSlicerLoadableModule.h"

class qSlicerSurgicalBridgeModulePrivate;

class qSlicerSurgicalBridgeModule : public qSlicerLoadableModule
{
  Q_OBJECT
  Q_PLUGIN_METADATA(IID "org.slicer.modules.loadable.qSlicerLoadableModule/1.0")
  Q_INTERFACES(qSlicerLoadableModule)
public:
  typedef qSlicerLoadableModule Superclass;
  explicit qSlicerSurgicalBridgeModule(QObject *parent=0);
  ~qSlicerSurgicalBridgeModule() override;
  qSlicerGetTitleMacro(QTMODULE_TITLE);
  QString helpText() const override;
  QString acknowledgementText() const override;
  QStringList contributors() const override;
  QStringList categories() const override;
  QStringList dependencies() const override;
protected:
  void setup() override;
  qSlicerAbstractModuleRepresentation * createWidgetRepresentation() override;
  vtkMRMLAbstractLogic* createLogic() override;
protected:
  QScopedPointer<qSlicerSurgicalBridgeModulePrivate> d_ptr;
private:
  Q_DECLARE_PRIVATE(qSlicerSurgicalBridgeModule);
  Q_DISABLE_COPY(qSlicerSurgicalBridgeModule);
};
#endif
