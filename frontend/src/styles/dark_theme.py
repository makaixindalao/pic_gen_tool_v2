"""
深色主题样式定义
"""

DARK_STYLE = """
QWidget {
    color: #eff0f1;
    background-color: #31363b;
    font-size: 11pt;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
}

QMainWindow {
    background-color: #232629;
}

QTabWidget::pane {
    border: 1px solid #454a4f;
    border-radius: 6px;
    margin-top: 5px;
}

QTabBar::tab {
    background: #3c4146;
    border: 1px solid #454a4f;
    border-bottom-color: #31363b;
    padding: 12px 30px;
    margin-right: 2px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-weight: bold;
}

QTabBar::tab:selected {
    background: #555e66;
    border-color: #555e66;
    border-bottom-color: #555e66;
}

QTabBar::tab:hover {
    background: #4a5157;
}

QGroupBox {
    background-color: #2b2f34;
    border: 1px solid #454a4f;
    border-radius: 8px;
    margin-top: 15px;
    padding-top: 10px;
    font-weight: bold;
    font-size: 12pt;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 5px 10px;
    background-color: #2b2f34;
    border-radius: 6px;
    color: #ffffff;
}

QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #2b2f34;
    padding: 8px;
    border: 1px solid #454a4f;
    border-radius: 6px;
    font-size: 10pt;
}

QLineEdit:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 2px solid #0078d7;
}

QPushButton {
    background-color: #0078d7;
    color: white;
    font-weight: bold;
    border: none;
    border-radius: 6px;
    padding: 12px 20px;
    font-size: 11pt;
}

QPushButton:hover {
    background-color: #008bf2;
}

QPushButton:pressed {
    background-color: #006ac1;
}

QPushButton:disabled {
    background-color: #454a4f;
    color: #888;
}

QRadioButton {
    spacing: 8px;
    font-size: 10pt;
    padding: 4px;
}

QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border: 2px solid #454a4f;
    border-radius: 10px;
    background-color: #31363b;
}

QRadioButton::indicator:checked {
    background-color: #0078d7;
    border-color: #0078d7;
}

QCheckBox {
    spacing: 8px;
    font-size: 10pt;
    padding: 4px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 2px solid #454a4f;
    border-radius: 3px;
    background-color: #31363b;
}

QCheckBox::indicator:checked {
    background-color: #0078d7;
    border-color: #0078d7;
}

QComboBox::drop-down {
    border: none;
}

QFormLayout {
    spacing: 12px;
}

QProgressBar {
    border: 1px solid #454a4f;
    border-radius: 6px;
    text-align: center;
    background-color: #2b2f34;
    font-size: 10pt;
}

QProgressBar::chunk {
    background-color: #0078d7;
    border-radius: 5px;
}

QScrollArea {
    border: 1px solid #454a4f;
    border-radius: 6px;
    background-color: #2b2f34;
}

QScrollBar:vertical {
    background-color: #2b2f34;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background-color: #454a4f;
    border-radius: 6px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #555e66;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    border: none;
    background: none;
}

QFrame {
    border: none;
}

QSplitter::handle {
    background-color: #454a4f;
}

QSplitter::handle:horizontal {
    width: 3px;
}

QSplitter::handle:vertical {
    height: 3px;
}
""" 