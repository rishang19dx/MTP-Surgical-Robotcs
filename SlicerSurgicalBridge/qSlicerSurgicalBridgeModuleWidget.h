#ifndef __qSlicerSurgicalBridgeModuleWidget_h
#define __qSlicerSurgicalBridgeModuleWidget_h

#include "qSlicerAbstractModuleWidget.h"

class qSlicerSurgicalBridgeModuleWidgetPrivate;
class vtkMRMLNode;

class qSlicerSurgicalBridgeModuleWidget : public qSlicerAbstractModuleWidget
{
  Q_OBJECT
public:
  typedef qSlicerAbstractModuleWidget Superclass;
  qSlicerSurgicalBridgeModuleWidget(QWidget *parent=0);
  ~qSlicerSurgicalBridgeModuleWidget() override;
public slots:
  void setMRMLScene(vtkMRMLScene *scene) override;
protected slots:
  void onConnectClicked();
  void onSendCommandClicked();
  void onTargetModelNodeChanged(vtkMRMLNode* node);
protected:
  void setup() override;
  QScopedPointer<qSlicerSurgicalBridgeModuleWidgetPrivate> d_ptr;
private:
  Q_DECLARE_PRIVATE(qSlicerSurgicalBridgeModuleWidget);
  Q_DISABLE_COPY(qSlicerSurgicalBridgeModuleWidget);
};
#endif
