from PySide6.QtCore import Slot
from PySide6.QtWidgets import QFrame, QComboBox, QPushButton

from envs.gui_env.src.utils.alert_dialogs import WarningDialog
from envs.gui_env.src.utils.utils import SignalHandler

CAR_MODELS = ["Car A", "Car B", "Car C"]  # pragma: no cover

TIRE_VARIANTS = ["18 Inch", "19 Inch", "20 Inch", "22 Inch"]  # pragma: no cover

INTERIOR_VARIANTS = ["Modern", "Vintage", "Sport"]  # pragma: no cover

PROPULSION_SYSTEMS = ["Combustion Engine A", "Combustion Engine B", "Combustion Engine C", "Electric Motor A",
                      "Electric Motor B"]  # pragma: no cover

"""
Allowed configurations:
Car A:
    - Tires: 20 and 22 Inch
    - Interior: Modern, Vintage
    - Propulsion System: Combustion Engine A, C
    
Car B:
    - Tires: 18, 19 and 20 Inch
    - Interior: Modern, Sport
    - Propulsion System: Electric Motor A, B
    
Car C:
    - Tires: 19, 20 and 22 Inch
    - Interior: Vintage, Sport
    - Propulsion System: Combustion Engine B, C and Electric Motor A, B
"""  # pragma: no cover


class CarConfigurator:

    def __init__(self,
                 car_model_selection_frame: QFrame, car_model_selection_combobox: QComboBox,
                 tire_selection_frame: QFrame, tire_selection_combobox: QComboBox,
                 interior_design_frame: QFrame, interior_design_combobox: QComboBox,
                 propulsion_system_frame: QFrame, propulsion_system_combobox: QComboBox,
                 show_configuration_button: QPushButton):  # pragma: no cover

        self.car_model_selection_frame = car_model_selection_frame
        self.car_model_selection_combobox = car_model_selection_combobox
        self.tire_selection_frame = tire_selection_frame
        self.tire_selection_combobox = tire_selection_combobox
        self.interior_design_frame = interior_design_frame
        self.interior_design_combobox = interior_design_combobox
        self.propulsion_system_frame = propulsion_system_frame
        self.propulsion_system_combobox = propulsion_system_combobox
        self.show_configuration_button = show_configuration_button
        self.signal_handler = SignalHandler()

        self.car_a = True
        self.car_b = True
        self.car_c = True

        self.tire_18_inch = True
        self.tire_19_inch = True
        self.tire_20_inch = True
        self.tire_22_inch = True

        self.modern_interior = True
        self.vintage_interior = True
        self.sport_interior = True

        self.combustion_engine_a = True
        self.combustion_engine_b = True
        self.combustion_engine_c = True
        self.electric_motor_a = True
        self.electric_motor_b = True

        self.selected_car = None
        self.selected_tire = None
        self.selected_interior = None
        self.selected_propulsion_system = None

        self._initialize()
        self._connect()

    def _initialize(self):  # pragma: no cover
        # This keeps the space in the layout, even when the widget is not visible
        default_size_policy = self.tire_selection_frame.sizePolicy()
        default_size_policy.setRetainSizeWhenHidden(True)
        self.tire_selection_frame.setSizePolicy(default_size_policy)
        self.interior_design_frame.setSizePolicy(default_size_policy)
        self.propulsion_system_frame.setSizePolicy(default_size_policy)

        # Use the size policy from the button because I do not know if it is different to the one from the QFrame
        button_size_policy = self.show_configuration_button.sizePolicy()
        button_size_policy.setRetainSizeWhenHidden(True)
        self.show_configuration_button.setSizePolicy(button_size_policy)

        self._reset()

    def _connect(self):  # pragma: no cover
        self.car_model_selection_combobox.activated.connect(self.change_selected_car)
        self.tire_selection_combobox.activated.connect(self.change_selected_tire)
        self.interior_design_combobox.activated.connect(self.change_selected_interior)
        self.propulsion_system_combobox.activated.connect(self.change_selected_propulsion_system)

        self.show_configuration_button.clicked.connect(self.print_selected_configuration)

    def _hide_all_boxes(self):
        # Make the configurations invisible
        self.tire_selection_frame.setVisible(False)
        self.interior_design_frame.setVisible(False)
        self.propulsion_system_frame.setVisible(False)
        self.show_configuration_button.setVisible(False)

        self.signal_handler.changed_active_car_configurator_widgets.emit([
            (False, self.tire_selection_frame),
            (False, self.interior_design_frame),
            (False, self.propulsion_system_frame),
            (False, self.show_configuration_button)
        ])

    def _reset(self):
        cars = []
        if self.car_a:
            cars.append("Car A")
        if self.car_b:
            cars.append("Car B")
        if self.car_c:
            cars.append("Car C")

        self.car_model_selection_combobox.clear()
        self.car_model_selection_combobox.addItems(cars)
        self._hide_all_boxes()

    def update_cars_by_tire(self):
        car_a_disabled = True
        car_b_disabled = True
        car_c_disabled = True

        if self.tire_18_inch:
            car_b_disabled = False
        if self.tire_19_inch:
            car_b_disabled = False
            car_c_disabled = False
        if self.tire_20_inch:
            car_a_disabled = False
            car_b_disabled = False
            car_c_disabled = False
        if self.tire_22_inch:
            car_a_disabled = False
            car_c_disabled = False

        return car_a_disabled, car_b_disabled, car_c_disabled

    def update_cars_by_interior(self):
        car_a_disabled = True
        car_b_disabled = True
        car_c_disabled = True

        if self.modern_interior:
            car_a_disabled = False
            car_b_disabled = False
        if self.vintage_interior:
            car_a_disabled = False
            car_c_disabled = False
        if self.sport_interior:
            car_b_disabled = False
            car_c_disabled = False

        return car_a_disabled, car_b_disabled, car_c_disabled

    def update_cars_by_propulsion_system(self):
        car_a_disabled = True
        car_b_disabled = True
        car_c_disabled = True

        if self.combustion_engine_a:
            car_a_disabled = False
        if self.combustion_engine_b:
            car_c_disabled = False
        if self.combustion_engine_c:
            car_a_disabled = False
            car_c_disabled = False
        if self.electric_motor_a:
            car_b_disabled = False
            car_c_disabled = False
        if self.electric_motor_b:
            car_b_disabled = False
            car_c_disabled = False

        return car_a_disabled, car_b_disabled, car_c_disabled

    def update_comboboxes(self):
        disabled_cars = []

        car_a_disabled_tire, car_b_disabled_tire, car_c_disabled_tire = self.update_cars_by_tire()
        car_a_disabled_interior, car_b_disabled_interior, car_c_disabled_interior = self.update_cars_by_interior()
        car_a_disabled_propulsion, car_b_disabled_propulsion, car_c_disabled_propulsion = self.update_cars_by_propulsion_system()

        if not car_a_disabled_tire and not car_a_disabled_interior and not car_a_disabled_propulsion:
            self.car_a = True
        else:
            if self.car_a:
                disabled_cars.append("Car A")
            self.car_a = False

        if not car_b_disabled_tire and not car_b_disabled_interior and not car_b_disabled_propulsion:
            self.car_b = True
        else:
            if self.car_b:
                disabled_cars.append("Car B")
            self.car_b = False

        if not car_c_disabled_tire and not car_c_disabled_interior and not car_c_disabled_propulsion:
            self.car_c = True
        else:
            if self.car_c:
                disabled_cars.append("Car C")
            self.car_c = False

        if len(disabled_cars) > 0:
            self.signal_handler.disabled_cars.emit(', '.join(car for car in disabled_cars))

        self._reset()

    def initialize_car_a(self):
        tires = []
        if self.tire_20_inch:
            tires.append("20 Inch")
        if self.tire_22_inch:
            tires.append("22 Inch")
        self.tire_selection_combobox.addItems(tires)

        interiors = []
        if self.modern_interior:
            interiors.append("Modern")
        if self.vintage_interior:
            interiors.append("Vintage")
        self.interior_design_combobox.addItems(interiors)

        propulsion_systems = []
        if self.combustion_engine_a:
            propulsion_systems.append("Combustion Engine A")
        if self.combustion_engine_c:
            propulsion_systems.append("Combustion Engine C")
        self.propulsion_system_combobox.addItems(propulsion_systems)

    def initialize_car_b(self):
        tires = []
        if self.tire_18_inch:
            tires.append("18 Inch")
        if self.tire_19_inch:
            tires.append("19 Inch")
        if self.tire_20_inch:
            tires.append("20 Inch")
        self.tire_selection_combobox.addItems(tires)

        interiors = []
        if self.modern_interior:
            interiors.append("Modern")
        if self.sport_interior:
            interiors.append("Sport")
        self.interior_design_combobox.addItems(interiors)

        propulsion_systems = []
        if self.electric_motor_a:
            propulsion_systems.append("Electric Motor A")
        if self.electric_motor_b:
            propulsion_systems.append("Electric Motor B")
        self.propulsion_system_combobox.addItems(propulsion_systems)

    def initialize_car_c(self):
        tires = []
        if self.tire_19_inch:
            tires.append("19 Inch")
        if self.tire_20_inch:
            tires.append("20 Inch")
        if self.tire_22_inch:
            tires.append("22 Inch")
        self.tire_selection_combobox.addItems(tires)

        interiors = []
        if self.vintage_interior:
            interiors.append("Vintage")
        if self.sport_interior:
            interiors.append("Sport")
        self.interior_design_combobox.addItems(interiors)

        propulsion_systems = []
        if self.combustion_engine_b:
            propulsion_systems.append("Combustion Engine B")
        if self.combustion_engine_c:
            propulsion_systems.append("Combustion Engine C")
        if self.electric_motor_a:
            propulsion_systems.append("Electric Motor A")
        if self.electric_motor_b:
            propulsion_systems.append("Electric Motor B")
        self.propulsion_system_combobox.addItems(propulsion_systems)

    @Slot()
    def change_selected_car(self):
        selected_car = self.car_model_selection_combobox.currentText()
        initialize_function = None
        if selected_car == "Car A":
            self.selected_car = "Car A"
            initialize_function = self.initialize_car_a
        elif selected_car == "Car B":
            self.selected_car = "Car B"
            initialize_function = self.initialize_car_b
        elif selected_car == "Car C":
            self.selected_car = "Car C"
            initialize_function = self.initialize_car_c
        assert self.selected_car in CAR_MODELS

        self.tire_selection_combobox.clear()
        self.interior_design_combobox.clear()
        self.propulsion_system_combobox.clear()

        # Initialize the content of the comboboxes depending on the selected cars as the cars can not have all possible
        # parts
        initialize_function()

        self.tire_selection_frame.setVisible(True)
        self.interior_design_frame.setVisible(False)
        self.propulsion_system_frame.setVisible(False)
        self.show_configuration_button.setVisible(False)

        self.signal_handler.changed_active_car_configurator_widgets.emit([
            (True, self.tire_selection_frame),
            (False, self.interior_design_frame),
            (False, self.propulsion_system_frame),
            (False, self.show_configuration_button)
        ])

    @Slot()
    def change_selected_tire(self):
        selected_tire = self.tire_selection_combobox.currentText()
        if selected_tire == "18 Inch":
            self.selected_tire = "18 Inch"
        elif selected_tire == "19 Inch":
            self.selected_tire = "19 Inch"
        elif selected_tire == "20 Inch":
            self.selected_tire = "20 Inch"
        elif selected_tire == "22 Inch":
            self.selected_tire = "22 Inch"
        assert self.selected_tire in TIRE_VARIANTS

        self.interior_design_frame.setVisible(True)
        self.propulsion_system_frame.setVisible(False)
        self.show_configuration_button.setVisible(False)

        self.signal_handler.changed_active_car_configurator_widgets.emit([
            (True, self.tire_selection_frame),
            (True, self.interior_design_frame),
            (False, self.propulsion_system_frame),
            (False, self.show_configuration_button)
        ])

    @Slot()
    def change_selected_interior(self):
        selected_interior = self.interior_design_combobox.currentText()
        if selected_interior == "Modern":
            self.selected_interior = "Modern"
        elif selected_interior == "Vintage":
            self.selected_interior = "Vintage"
        elif selected_interior == "Sport":
            self.selected_interior = "Sport"
        assert self.selected_interior in INTERIOR_VARIANTS

        self.propulsion_system_frame.setVisible(True)
        self.show_configuration_button.setVisible(False)

        self.signal_handler.changed_active_car_configurator_widgets.emit([
            (True, self.tire_selection_frame),
            (True, self.interior_design_frame),
            (True, self.propulsion_system_frame),
            (False, self.show_configuration_button)
        ])

    @Slot()
    def change_selected_propulsion_system(self):
        selected_propulsion_system = self.propulsion_system_combobox.currentText()
        if selected_propulsion_system == "Combustion Engine A":
            self.selected_propulsion_system = "Combustion Engine A"
        elif selected_propulsion_system == "Combustion Engine B":
            self.selected_propulsion_system = "Combustion Engine B"
        elif selected_propulsion_system == "Combustion Engine C":
            self.selected_propulsion_system = "Combustion Engine C"
        elif selected_propulsion_system == "Electric Motor A":
            self.selected_propulsion_system = "Electric Motor A"
        elif selected_propulsion_system == "Electric Motor B":
            self.selected_propulsion_system = "Electric Motor B"
        assert self.selected_propulsion_system in PROPULSION_SYSTEMS

        self.show_configuration_button.setVisible(True)

        self.signal_handler.changed_active_car_configurator_widgets.emit([
            (True, self.tire_selection_frame),
            (True, self.interior_design_frame),
            (True, self.propulsion_system_frame),
            (True, self.show_configuration_button)
        ])

    @Slot()
    def print_selected_configuration(self):
        selected_car = None
        if self.selected_car == "Car A":
            selected_car = "Car A"
        elif self.selected_car == "Car B":
            selected_car = "Car B"
        elif self.selected_car == "Car C":
            selected_car = "Car C"
        assert selected_car in CAR_MODELS

        selected_tire = None
        if self.selected_tire == "18 Inch":
            selected_tire = "18 Inch"
        elif self.selected_tire == "19 Inch":
            selected_tire = "19 Inch"
        elif self.selected_tire == "20 Inch":
            selected_tire = "20 Inch"
        elif self.selected_tire == "22 Inch":
            selected_tire = "22 Inch"
        assert selected_tire in TIRE_VARIANTS

        selected_interior = None
        if self.selected_interior == "Modern":
            selected_interior = "Modern"
        elif self.selected_interior == "Vintage":
            selected_interior = "Vintage"
        elif self.selected_interior == "Sport":
            selected_interior = "Sport"
        assert selected_interior in INTERIOR_VARIANTS

        selected_propulsion_system = None
        if self.selected_propulsion_system == "Combustion Engine A":
            selected_propulsion_system = "Combustion Engine A"
        elif self.selected_propulsion_system == "Combustion Engine B":
            selected_propulsion_system = "Combustion Engine B"
        elif self.selected_propulsion_system == "Combustion Engine C":
            selected_propulsion_system = "Combustion Engine C"
        elif self.selected_propulsion_system == "Electric Motor A":
            selected_propulsion_system = "Electric Motor A"
        elif self.selected_propulsion_system == "Electric Motor B":
            selected_propulsion_system = "Electric Motor B"
        assert selected_propulsion_system in PROPULSION_SYSTEMS

        selected_configuration_message = (f"Selected {selected_car} with the following configuration:\n" +
                                          f"Tires: {selected_tire}\n" +
                                          f"Interior: {selected_interior}\n" +
                                          f"Propulsion System: {selected_propulsion_system}")
        self.signal_handler.car_configured.emit(selected_configuration_message)
        self._reset()

    # Methods for changing the settings

    @Slot(bool)
    def change_18_inch_tire(self, checked: bool):
        if checked:
            self.tire_18_inch = True
        else:
            self.tire_18_inch = False
        self.update_comboboxes()

    @Slot(bool)
    def change_19_inch_tire(self, checked: bool):
        if checked:
            self.tire_19_inch = True
        else:
            self.tire_19_inch = False
        self.update_comboboxes()

    @Slot(bool)
    def change_20_inch_tire(self, checked: bool):
        if checked:
            self.tire_20_inch = True
        else:
            self.tire_20_inch = False
        self.update_comboboxes()

    @Slot(bool)
    def change_22_inch_tire(self, checked: bool):
        if checked:
            self.tire_22_inch = True
        else:
            self.tire_22_inch = False
        self.update_comboboxes()

    @Slot(bool)
    def change_modern_interior(self, checked: bool):
        if checked:
            self.modern_interior = True
        else:
            self.modern_interior = False
        self.update_comboboxes()

    @Slot(bool)
    def change_vintage_interior(self, checked: bool):
        if checked:
            self.vintage_interior = True
        else:
            self.vintage_interior = False
        self.update_comboboxes()

    @Slot(bool)
    def change_sport_interior(self, checked: bool):
        if checked:
            self.sport_interior = True
        else:
            self.sport_interior = False
        self.update_comboboxes()

    @Slot(bool)
    def change_combustion_engine_a(self, checked: bool):
        if checked:
            self.combustion_engine_a = True
        else:
            self.combustion_engine_a = False
        self.update_comboboxes()

    @Slot(bool)
    def change_combustion_engine_b(self, checked: bool):
        if checked:
            self.combustion_engine_b = True
        else:
            self.combustion_engine_b = False
        self.update_comboboxes()

    @Slot(bool)
    def change_combustion_engine_c(self, checked: bool):
        if checked:
            self.combustion_engine_c = True
        else:
            self.combustion_engine_c = False
        self.update_comboboxes()

    @Slot(bool)
    def change_electric_motor_a(self, checked: bool):
        if checked:
            self.electric_motor_a = True
        else:
            self.electric_motor_a = False
        self.update_comboboxes()

    @Slot(bool)
    def change_electric_motor_b(self, checked: bool):
        if checked:
            self.electric_motor_b = True
        else:
            self.electric_motor_b = False
        self.update_comboboxes()


@Slot(str)
def show_disabled_cars_error_dialog(settings_dialog, disabled_cars: str):
    disabled_cars_dialog = WarningDialog(
        warning_text=f"Disabled the following car(s): {disabled_cars}",
        parent=settings_dialog.settings_dialog
    )
    disabled_cars_dialog.show()


@Slot(str)
def show_car_configuration_dialog(settings_dialog, car_configuration: str):
    car_configuration_dialog = WarningDialog(warning_text=car_configuration, parent=settings_dialog.settings_dialog)
    car_configuration_dialog.show()
