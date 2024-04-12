#ifndef TOPMAP_PANEL_H
#define TOPMAP_PANEL_H

#include <cstdio>

#include <rviz_common/panel.hpp>
#include <rviz_common/properties/property_tree_widget.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/parameter_client.hpp>


class QComboBox;
class QMessageBox;
class QModelIndex;
class QPushButton;
class QInputDialog;
class QStyleOptionFrame;
class QStyleOptionViewItem;

namespace bayesian_topological_localisation_rviz_tools {
/**
 * @brief Panel for choosing the view controller and saving and restoring
 * viewpoints.
 */
class BayesianLocalisationPanel : public rviz_common::Panel {
public:
  BayesianLocalisationPanel(QWidget* parent = 0);
  virtual ~BayesianLocalisationPanel() {}

  /** @brief Overridden from BayesianLocalisationPanel.  Just calls setMan() with vis_manager_->getTopmapManager(). */
  virtual void onInitialize();

private Q_SLOTS:
  void onAddAgentClicked();
  void onRemoveAgentClicked();

private:
  rclcpp::Node::SharedPtr nh_;
  rclcpp::SyncParametersClient::SharedPtr param_client_;
  rviz_common::properties::PropertyTreeWidget* properties_view_;
  rclcpp::Logger logger_{rclcpp::get_logger("rviz2")};
};

} // namespace topological_rviz_tools

#endif // TOPMAP_PANEL_H
