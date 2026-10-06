#include "qSlicerSurgicalBridgeModuleWidget.h"
#include "ui_qSlicerSurgicalBridgeModuleWidget.h"
#include "vtkSlicerSurgicalBridgeLogic.h"
#include "vtkMRMLModelNode.h"

class qSlicerSurgicalBridgeModuleWidgetPrivate : public Ui_qSlicerSurgicalBridgeModuleWidget
{
  Q_DECLARE_PUBLIC(qSlicerSurgicalBridgeModuleWidget);
protected:
  qSlicerSurgicalBridgeModuleWidget* const q_ptr;
public:
  qSlicerSurgicalBridgeModuleWidgetPrivate(qSlicerSurgicalBridgeModuleWidget& object);
  vtkSlicerSurgicalBridgeLogic* logic() const;
};

qSlicerSurgicalBridgeModuleWidgetPrivate::qSlicerSurgicalBridgeModuleWidgetPrivate(qSlicerSurgicalBridgeModuleWidget& object)
  : q_ptr(&object) {}

vtkSlicerSurgicalBridgeLogic* qSlicerSurgicalBridgeModuleWidgetPrivate::logic() const {
  Q_Q(const qSlicerSurgicalBridgeModuleWidget);
  return vtkSlicerSurgicalBridgeLogic::SafeDownCast(q->logic());
}

qSlicerSurgicalBridgeModuleWidget::qSlicerSurgicalBridgeModuleWidget(QWidget* _parent)
  : Superclass(_parent), d_ptr(new qSlicerSurgicalBridgeModuleWidgetPrivate(*this)) {}

qSlicerSurgicalBridgeModuleWidget::~qSlicerSurgicalBridgeModuleWidget() {}

void qSlicerSurgicalBridgeModuleWidget::setup() {
  Q_D(qSlicerSurgicalBridgeModuleWidget);
  d->setupUi(this);
  this->Superclass::setup();
  
  connect(d->connectButton, SIGNAL(clicked()), this, SLOT(onConnectClicked()));
  connect(d->sendCommandButton, SIGNAL(clicked()), this, SLOT(onSendCommandClicked()));
  connect(d->targetModelSelector, SIGNAL(currentNodeChanged(vtkMRMLNode*)), this, SLOT(onTargetModelNodeChanged(vtkMRMLNode*)));
  connect(d->forceModelSelector, SIGNAL(currentNodeChanged(vtkMRMLNode*)), this, SLOT(onForceModelNodeChanged(vtkMRMLNode*)));
  connect(d->sendVFButton, SIGNAL(clicked()), this, SLOT(onSendVFClicked()));
  connect(d->vfModelSelector, SIGNAL(currentNodeChanged(vtkMRMLNode*)), this, SLOT(onVFModelNodeChanged(vtkMRMLNode*)));
}

void qSlicerSurgicalBridgeModuleWidget::setMRMLScene(vtkMRMLScene* scene) {
  this->Superclass::setMRMLScene(scene);
}

void qSlicerSurgicalBridgeModuleWidget::onConnectClicked() {
  Q_D(qSlicerSurgicalBridgeModuleWidget);
  if (d->logic()) {
    d->logic()->ConnectToPhysicsServer("localhost", 18944);
  }
}

void qSlicerSurgicalBridgeModuleWidget::onSendCommandClicked() {
  Q_D(qSlicerSurgicalBridgeModuleWidget);
  if (d->logic()) {
    std::string q_des = d->qDesLineEdit->text().toStdString();
    d->logic()->SendRobotCommand(q_des);
  }
}

void qSlicerSurgicalBridgeModuleWidget::onTargetModelNodeChanged(vtkMRMLNode* node) {
  Q_D(qSlicerSurgicalBridgeModuleWidget);
  if (d->logic()) {
    d->logic()->SetTargetModelNode(vtkMRMLModelNode::SafeDownCast(node));
  }
}

void qSlicerSurgicalBridgeModuleWidget::onForceModelNodeChanged(vtkMRMLNode* node) {
  Q_D(qSlicerSurgicalBridgeModuleWidget);
  if (d->logic()) {
    d->logic()->SetForceModelNode(vtkMRMLModelNode::SafeDownCast(node));
  }
}

void qSlicerSurgicalBridgeModuleWidget::onSendVFClicked() {
  Q_D(qSlicerSurgicalBridgeModuleWidget);
  if (d->logic()) {
    d->logic()->SendVirtualFixture();
  }
}

void qSlicerSurgicalBridgeModuleWidget::onVFModelNodeChanged(vtkMRMLNode* node) {
  Q_D(qSlicerSurgicalBridgeModuleWidget);
  if (d->logic()) {
    d->logic()->SetVFModelNode(vtkMRMLModelNode::SafeDownCast(node));
  }
}
