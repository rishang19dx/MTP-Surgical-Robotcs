cmake_minimum_required(VERSION 3.16.3...3.19.7)
project(SurgicalBridge)

# -----------------------------------------------------------------------------
# Extension meta-data
set(EXTENSION_HOMEPAGE "https://github.com/Antigravity/MTP")
set(EXTENSION_CATEGORY "Simulation")
set(EXTENSION_CONTRIBUTORS "Antigravity")
set(EXTENSION_DESCRIPTION "Connects Slicer to an external physics engine via OpenIGTLink.")
set(EXTENSION_ICONURL "https://github.com/Antigravity/MTP/icon.png")
set(EXTENSION_DEPENDS "OpenIGTLinkIF")

find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

add_subdirectory(Logic)

# -----------------------------------------------------------------------------
# Module
set(MODULE_NAME ${PROJECT_NAME})

set(MODULE_EXPORT_DIRECTIVE "Q_SLICER_QTMODULES_${MODULE_NAME}_EXPORT")

set(MODULE_INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/Logic
  ${CMAKE_CURRENT_BINARY_DIR}/Logic
  )

set(MODULE_SRCS
  qSlicer${MODULE_NAME}Module.cxx
  qSlicer${MODULE_NAME}Module.h
  qSlicer${MODULE_NAME}ModuleWidget.cxx
  qSlicer${MODULE_NAME}ModuleWidget.h
  )

set(MODULE_MOC_SRCS
  qSlicer${MODULE_NAME}Module.h
  qSlicer${MODULE_NAME}ModuleWidget.h
  )

set(MODULE_UI_SRCS
  qSlicer${MODULE_NAME}ModuleWidget.ui
  )

set(MODULE_TARGET_LIBRARIES
  vtkSlicer${MODULE_NAME}ModuleLogic
  )

slicerMacroBuildQtModule(
  NAME ${MODULE_NAME}
  EXPORT_DIRECTIVE ${MODULE_EXPORT_DIRECTIVE}
  INCLUDE_DIRECTORIES ${MODULE_INCLUDE_DIRECTORIES}
  SRCS ${MODULE_SRCS}
  MOC_SRCS ${MODULE_MOC_SRCS}
  UI_SRCS ${MODULE_UI_SRCS}
  TARGET_LIBRARIES ${MODULE_TARGET_LIBRARIES}
  )
