#include "bayesian_topological_localisation_rviz_tools/bayesian_localisation_panel.hpp"

#include <QLabel>
#include <QDebug>
#include <QDialogButtonBox>
#include <QListWidget>
#include <QComboBox>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QMessageBox>
#include <QComboBox>

using namespace std;

namespace bayesian_topological_localisation_rviz_tools {

BayesianLocalisationPanel::BayesianLocalisationPanel(QWidget* parent)
  : rviz_common::Panel(parent)//,
    //property_model_(new rviz_common::properties::PropertyTreeModel(root_property_))
    {
  properties_view_ = new rviz_common::properties::PropertyTreeWidget();
  nh_ = make_shared<rclcpp::Node>("bayesian_topological_localisation_panel");
  param_client_ = make_shared<rclcpp::SyncParametersClient>(nh_, "~");

  QPushButton* button_add_agent = new QPushButton("Add Agent");
  QPushButton* button_remove_agent = new QPushButton("Remove Agent");
  QHBoxLayout* button_layout = new QHBoxLayout;
  button_layout->addWidget(button_add_agent);
  button_layout->addWidget(button_remove_agent);
  button_layout->setContentsMargins(2, 0, 2, 2);

  QVBoxLayout* main_layout = new QVBoxLayout;
  main_layout->setContentsMargins(0, 0, 0, 0);
  main_layout->addWidget(properties_view_);
  main_layout->addLayout(button_layout);
  setLayout(main_layout);
  
  connect(button_remove_agent, &QPushButton::clicked, this, &BayesianLocalisationPanel::onRemoveAgentClicked);
  connect(button_add_agent, &QPushButton::clicked, this, &BayesianLocalisationPanel::onAddAgentClicked);
}

void BayesianLocalisationPanel::onInitialize() {
  RCLCPP_INFO(logger_, "BayesianLocalisationPanel::OnInitialise");
}

void BayesianLocalisationPanel::onRemoveAgentClicked() {
  RCLCPP_INFO(logger_, "BayesianLocalisationPanel::onRemoveAgentClicked");
}

void BayesianLocalisationPanel::onAddAgentClicked() {
  QDialog *dialog = new QDialog();
  QVBoxLayout *layout_dialog = new QVBoxLayout();

  // Title
  QLabel *label_title = new QLabel("Add new localisation agent:");

  // Data input fields
  QHBoxLayout *layout_datafields = new QHBoxLayout();

  QVBoxLayout *layout_labels = new QVBoxLayout();
  QLabel *label_name = new QLabel("Name: ");
  QLabel *label_n_particles = new QLabel("Particles: ");
  QLabel *label_do_prediction = new QLabel("Do Prediction: ");
  QLabel *label_prediction_rate = new QLabel("Prediction Rate: ");
  layout_labels->addWidget(label_name);
  layout_labels->addWidget(label_n_particles);
  layout_labels->addWidget(label_do_prediction);
  layout_labels->addWidget(label_prediction_rate);

  QVBoxLayout *layout_input = new QVBoxLayout();
  QLineEdit *lineedit_name = new QLineEdit();
  QLineEdit *lineedit_n_particles = new QLineEdit();
  QComboBox *combobox_do_predicition = new QComboBox();
  combobox_do_predicition->addItems(QStringList() << "True" << "False");
  QLineEdit *lineedit_prediction_rate = new QLineEdit();
  layout_input->addWidget(lineedit_name);
  layout_input->addWidget(lineedit_n_particles);
  layout_input->addWidget(combobox_do_predicition);
  layout_input->addWidget(lineedit_prediction_rate);

  layout_datafields->addLayout(layout_labels);
  layout_datafields->addLayout(layout_input);

  // Ok / Cancel fields
  QDialogButtonBox *button_box = new QDialogButtonBox(QDialogButtonBox::Ok
                                                      | QDialogButtonBox::Cancel);
  QObject::connect(button_box, SIGNAL(accepted()), dialog, SLOT(accept()));
  QObject::connect(button_box, SIGNAL(rejected()), dialog, SLOT(reject()));

  layout_dialog->addWidget(label_title);
  layout_dialog->addLayout(layout_datafields);
  layout_dialog->addWidget(button_box);

  dialog->setLayout(layout_dialog);
  int result = dialog->exec();

}

} // namespace bayesian_topological_localisation_rviz_tools

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(bayesian_topological_localisation_rviz_tools::BayesianLocalisationPanel, rviz_common::Panel)
