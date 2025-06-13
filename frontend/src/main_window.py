import sys
import os
import asyncio
from PyQt6.QtGui import QIcon, QFont, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QGroupBox, QFormLayout, QLineEdit, QSpinBox, QTextEdit, QPushButton,
    QGridLayout, QFrame, QSplitter, QLabel, QProgressBar, QMessageBox,
    QRadioButton, QDoubleSpinBox, QComboBox, QStyle, QCheckBox, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer

from services.ai_service import AIServiceClient
from styles.dark_theme import DARK_STYLE

class AIGenerationWorker(QThread):
    """AI生成工作线程"""
    progress = pyqtSignal(str)  # 进度信息
    finished = pyqtSignal(dict)  # 完成信号
    error = pyqtSignal(str)  # 错误信号
    
    def __init__(self, ai_client, generation_params):
        super().__init__()
        self.ai_client = ai_client
        self.generation_params = generation_params
    
    def run(self):
        try:
            self.progress.emit("正在生成图像...")
            result = self.ai_client.generate_images(**self.generation_params)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

class PromptWorker(QThread):
    """提示词构建工作线程"""
    finished = pyqtSignal(str, str)  # positive_prompt, negative_prompt
    error = pyqtSignal(str)  # 错误信息
    
    def __init__(self, ai_client, target, weather, scene, custom_prompt=""):
        super().__init__()
        self.ai_client = ai_client
        self.target = target
        self.weather = weather
        self.scene = scene
        self.custom_prompt = custom_prompt
    
    def run(self):
        try:
            positive_prompt, negative_prompt = self.ai_client.build_prompt(
                military_target=self.target,
                weather=self.weather,
                scene=self.scene,
                custom_prompt=self.custom_prompt
            )
            self.finished.emit(positive_prompt, negative_prompt)
        except Exception as e:
            self.error.emit(str(e))

class BatchGenerationWorker(QThread):
    """批量生成工作线程"""
    progress = pyqtSignal(str, int, int)  # 进度信息, 当前数量, 总数量
    batch_progress = pyqtSignal(int, int)  # 批次进度, 当前批次, 总批次
    finished = pyqtSignal(dict)  # 完成信号
    error = pyqtSignal(str)  # 错误信号
    image_generated = pyqtSignal(dict)  # 单张图片生成完成信号
    
    def __init__(self, ai_client, batch_configs):
        super().__init__()
        self.ai_client = ai_client
        self.batch_configs = batch_configs
    
    def run(self):
        try:
            self.progress.emit("开始批量生成...", 0, len(self.batch_configs))
            
            all_results = []
            successful_count = 0
            
            for i, config in enumerate(self.batch_configs):
                if self.isInterruptionRequested():
                    break
                    
                self.batch_progress.emit(i + 1, len(self.batch_configs))
                self.progress.emit(f"正在生成第 {i+1}/{len(self.batch_configs)} 组图像...", i, len(self.batch_configs))
                
                try:
                    result = self.ai_client.generate_images(**config)
                    all_results.append(result)
                    successful_count += 1
                    self.image_generated.emit(result)
                except Exception as e:
                    error_result = {"error": str(e), "config": config, "success": False}
                    all_results.append(error_result)
                    self.progress.emit(f"第 {i+1} 组生成失败: {str(e)}", i, len(self.batch_configs))
            
            final_result = {
                "results": all_results,
                "total_configs": len(self.batch_configs),
                "successful_count": successful_count,
                "failed_count": len(self.batch_configs) - successful_count
            }
            
            self.finished.emit(final_result)
            
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("军事目标数据集生成平台")
        self.setGeometry(100, 100, 1200, 800)
        self.setWindowIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon))

        # 初始化AI服务客户端
        self.ai_client = AIServiceClient()
        self.generation_worker = None
        self.batch_worker = None
        
        # 用于AI prompt自动生成
        self.ai_target_radios = []
        self.ai_weather_radios = []
        self.ai_scene_radios = []
        
        # 图片显示相关
        self.current_images = []  # 当前显示的图片列表
        self.current_image_index = 0  # 当前显示的图片索引

        self._create_main_widget()
        self._initialize_ai_service()

    def _initialize_ai_service(self):
        """初始化AI服务"""
        self.status_label.setText("正在初始化AI服务...")
        
        # 使用定时器异步检查服务状态
        self.init_timer = QTimer()
        self.init_timer.timeout.connect(self._check_service_status)
        self.init_timer.start(1000)  # 每秒检查一次
        
        # 尝试初始化服务
        try:
            success = self.ai_client.initialize_service()
            if success:
                self.status_label.setText("AI服务初始化成功")
                self._load_ai_options()
            else:
                self.status_label.setText("AI服务初始化失败，请检查后端服务")
        except Exception as e:
            self.status_label.setText(f"连接AI服务失败: {str(e)}")

    def _check_service_status(self):
        """检查服务状态"""
        try:
            status = self.ai_client.get_service_status()
            if status.get('service_status') == 'running':
                self.status_label.setText("AI服务运行中")
                self.ai_generate_btn.setEnabled(True)
                self.init_timer.stop()
            else:
                self.status_label.setText("AI服务未就绪...")
                self.ai_generate_btn.setEnabled(False)
        except:
            self.status_label.setText("无法连接到AI服务")
            self.ai_generate_btn.setEnabled(False)

    def _load_ai_options(self):
        """加载AI服务选项"""
        try:
            # 加载采样器列表
            scheduler_info = self.ai_client.get_available_schedulers()
            schedulers = scheduler_info.get('schedulers', [])
            if schedulers and hasattr(self, 'sampler_combo'):
                self.sampler_combo.clear()
                self.sampler_combo.addItems(schedulers)
                
        except Exception as e:
            print(f"加载AI选项失败: {str(e)}")

    def _create_main_widget(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 使用分割器来分隔左右面板
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左侧：功能区
        left_panel = self._create_left_panel()
        
        # 右侧：图像预览区
        right_panel = self._create_right_panel()
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([600, 400])  # 设置初始大小比例
        
        main_layout.addWidget(splitter)

    def _create_left_panel(self):
        """创建功能面板"""
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(10)

        tab_widget = QTabWidget()
        
        traditional_tab = self._create_traditional_generation_tab()
        ai_tab = self._create_ai_generation_tab()
        
        tab_widget.addTab(traditional_tab, "传统生成")
        tab_widget.addTab(ai_tab, "AI 生成")
        
        # 添加图标
        style = self.style()
        tab_widget.setTabIcon(0, style.standardIcon(QStyle.StandardPixmap.SP_FileIcon))
        tab_widget.setTabIcon(1, style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

        left_layout.addWidget(tab_widget)
        return left_panel

    def _create_right_panel(self):
        """创建右侧图像预览面板"""
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(12)
        
        # 图像预览区
        preview_group = QGroupBox("生成图像预览")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setSpacing(8)
        preview_layout.setContentsMargins(10, 15, 10, 10)
        
        # 图片显示区域
        self.preview_label = QLabel("生成的图像将在此处显示")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(380, 320)
        self.preview_label.setScaledContents(False)
        self.preview_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #454a4f;
                border-radius: 8px;
                background-color: #2b2f34;
                color: #888;
                font-size: 11pt;
                padding: 15px;
            }
        """)
        preview_layout.addWidget(self.preview_label)
        
        # 图片导航控制
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("◀ 上一张")
        self.next_btn = QPushButton("下一张 ▶")
        self.image_info_label = QLabel("0/0")
        
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.prev_btn.clicked.connect(self._show_previous_image)
        self.next_btn.clicked.connect(self._show_next_image)
        
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addStretch()
        nav_layout.addWidget(self.image_info_label)
        nav_layout.addStretch()
        nav_layout.addWidget(self.next_btn)
        
        # 添加测试按钮
        test_btn = QPushButton("测试显示")
        test_btn.clicked.connect(self._test_image_display)
        nav_layout.addWidget(test_btn)
        
        preview_layout.addLayout(nav_layout)
        
        # 生成状态信息
        status_group = QGroupBox("生成状态")
        status_layout = QVBoxLayout(status_group)
        status_layout.setSpacing(6)
        status_layout.setContentsMargins(10, 15, 10, 10)
        
        self.status_label = QLabel("等待开始生成...")
        self.status_label.setStyleSheet("color: #eff0f1; font-size: 10pt; padding: 5px;")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        self.progress_label = QLabel("进度: 0/0")
        self.progress_label.setStyleSheet("color: #0078d7; font-size: 10pt; padding: 5px;")
        
        # 批量生成进度
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setVisible(False)
        self.batch_progress_label = QLabel("批次进度: 0/0")
        self.batch_progress_label.setStyleSheet("color: #0078d7; font-size: 10pt; padding: 5px;")
        
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)
        status_layout.addWidget(self.progress_label)
        status_layout.addWidget(self.batch_progress_bar)
        status_layout.addWidget(self.batch_progress_label)
        
        right_layout.addWidget(preview_group, 3)
        right_layout.addWidget(status_group, 1)
        
        return right_panel

    def _create_radio_button_group(self, title, options):
        """创建带有单选框的通用分组"""
        group_box = QGroupBox(title)
        layout = QGridLayout(group_box)
        layout.setSpacing(12)
        layout.setContentsMargins(15, 20, 15, 15)
        
        radio_buttons = []
        for i, option in enumerate(options):
            rb = QRadioButton(option)
            radio_buttons.append(rb)
            layout.addWidget(rb, i // 2, i % 2)

        if radio_buttons:
            radio_buttons[0].setChecked(True)

        return group_box, radio_buttons

    def _create_traditional_generation_tab(self):
        """创建传统生成选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(20)
        layout.setContentsMargins(15, 25, 15, 15)

        targets = ["坦克", "战机", "舰艇"]
        target_group, _ = self._create_radio_button_group("军事目标", targets)
        
        weathers = ["雨天", "雪天", "大雾", "夜间"]
        weather_group, _ = self._create_radio_button_group("天气", weathers)
        
        scenes = ["城市", "岛屿", "乡村"]
        scene_group, _ = self._create_radio_button_group("地形场景", scenes)

        start_button = QPushButton("开始生成")
        start_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOkButton))
        start_button.setMinimumHeight(45)

        layout.addWidget(target_group)
        layout.addWidget(weather_group)
        layout.addWidget(scene_group)
        layout.addStretch()
        layout.addWidget(start_button)
        
        return tab

    def _create_ai_generation_tab(self):
        """创建AI生成标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)
        layout.setContentsMargins(8, 8, 8, 8)

        # 目标选择组
        selection_group = QGroupBox("目标与场景选择")
        selection_layout = QVBoxLayout(selection_group)
        selection_layout.setSpacing(8)
        selection_layout.setContentsMargins(10, 15, 10, 10)

        # 军事目标选择
        target_group_widget, self.ai_target_radios = self._create_radio_button_group("军事目标", ["坦克", "战机", "舰艇"])
        selection_layout.addWidget(target_group_widget)

        # 天气条件选择
        weather_group_widget, self.ai_weather_radios = self._create_radio_button_group("天气条件", ["雨天", "雪天", "大雾", "夜间"])
        selection_layout.addWidget(weather_group_widget)

        # 场景环境选择
        scene_group_widget, self.ai_scene_radios = self._create_radio_button_group("场景环境", ["城市", "岛屿", "乡村"])
        selection_layout.addWidget(scene_group_widget)

        # 为所有单选按钮添加事件监听
        for radio in self.ai_target_radios + self.ai_weather_radios + self.ai_scene_radios:
            radio.toggled.connect(self._update_ai_prompt)

        # 提示词组
        prompt_group = QGroupBox("提示词设置")
        prompt_layout = QVBoxLayout(prompt_group)
        prompt_layout.setSpacing(8)
        prompt_layout.setContentsMargins(10, 15, 10, 10)

        self.ai_prompt_text = QTextEdit()
        self.ai_prompt_text.setMaximumHeight(80)
        self.ai_prompt_text.setMinimumHeight(60)
        self.ai_prompt_text.setPlaceholderText("在此输入自定义提示词，或通过上方选择自动生成...")
        prompt_layout.addWidget(self.ai_prompt_text)

        # 提示词操作按钮
        prompt_btn_layout = QHBoxLayout()
        build_prompt_btn = QPushButton("构建提示词")
        optimize_prompt_btn = QPushButton("优化提示词")
        
        build_prompt_btn.clicked.connect(self._build_prompt)
        optimize_prompt_btn.clicked.connect(self._optimize_prompt)
        
        prompt_btn_layout.addWidget(build_prompt_btn)
        prompt_btn_layout.addWidget(optimize_prompt_btn)
        prompt_btn_layout.addStretch()
        
        prompt_layout.addLayout(prompt_btn_layout)

        # 生成配置组
        config_group = QGroupBox("生成配置")
        config_layout = QFormLayout(config_group)
        config_layout.setSpacing(6)
        config_layout.setContentsMargins(10, 15, 10, 10)

        # 生成模式选择
        mode_layout = QHBoxLayout()
        self.single_mode_radio = QRadioButton("单张生成")
        self.batch_mode_radio = QRadioButton("批量生成")
        self.single_mode_radio.setChecked(True)
        
        self.single_mode_radio.toggled.connect(self._on_generation_mode_changed)
        self.batch_mode_radio.toggled.connect(self._on_generation_mode_changed)
        
        mode_layout.addWidget(self.single_mode_radio)
        mode_layout.addWidget(self.batch_mode_radio)
        mode_layout.addStretch()
        config_layout.addRow("生成模式:", mode_layout)

        # 生成数量（批量模式）
        self.batch_count_spin = QSpinBox()
        self.batch_count_spin.setRange(1, 100)
        self.batch_count_spin.setValue(5)
        self.batch_count_spin.setEnabled(False)
        config_layout.addRow("生成数量:", self.batch_count_spin)

        # 每张图片数量
        self.num_images_spin = QSpinBox()
        self.num_images_spin.setRange(1, 10)
        self.num_images_spin.setValue(1)
        config_layout.addRow("每组图片数:", self.num_images_spin)

        # 采样步数
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(10, 100)
        self.steps_spin.setValue(30)
        config_layout.addRow("采样步数:", self.steps_spin)

        # CFG引导强度
        self.cfg_spin = QDoubleSpinBox()
        self.cfg_spin.setRange(1.0, 20.0)
        self.cfg_spin.setValue(7.5)
        self.cfg_spin.setSingleStep(0.5)
        config_layout.addRow("CFG强度:", self.cfg_spin)

        # 随机种子
        seed_layout = QHBoxLayout()
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-1, 2147483647)
        self.seed_spin.setValue(-1)
        self.seed_spin.setSpecialValueText("随机")
        
        random_seed_btn = QPushButton("随机")
        random_seed_btn.clicked.connect(lambda: self.seed_spin.setValue(-1))
        
        seed_layout.addWidget(self.seed_spin)
        seed_layout.addWidget(random_seed_btn)
        config_layout.addRow("种子:", seed_layout)

        # 采样器
        self.sampler_combo = QComboBox()
        self.sampler_combo.addItems([
            "DPM++ 2M Karras", "Euler a", "Euler", "LMS", 
            "Heun", "DPM2", "DPM2 a", "DPM++ 2S a", "DDIM"
        ])
        config_layout.addRow("采样器:", self.sampler_combo)

        # 批量生成选项
        batch_options_group = QGroupBox("批量生成选项")
        batch_options_layout = QVBoxLayout(batch_options_group)
        batch_options_layout.setSpacing(6)
        batch_options_layout.setContentsMargins(10, 15, 10, 10)
        
        self.random_target_cb = QCheckBox("随机军事目标")
        self.random_weather_cb = QCheckBox("随机天气条件")
        self.random_scene_cb = QCheckBox("随机场景环境")
        self.random_seed_cb = QCheckBox("每次使用随机种子")
        self.random_seed_cb.setChecked(True)
        
        batch_options_layout.addWidget(self.random_target_cb)
        batch_options_layout.addWidget(self.random_weather_cb)
        batch_options_layout.addWidget(self.random_scene_cb)
        batch_options_layout.addWidget(self.random_seed_cb)
        
        batch_options_group.setEnabled(False)
        self.batch_options_group = batch_options_group

        # 生成按钮
        self.ai_generate_btn = QPushButton("生成图像")
        self.ai_generate_btn.clicked.connect(self._generate_ai_images)
        self.ai_generate_btn.setEnabled(False)  # 初始禁用，等待服务就绪
        self.ai_generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #454a4f;
                color: #888;
            }
        """)

        layout.addWidget(selection_group)
        layout.addWidget(prompt_group)
        layout.addWidget(config_group)
        layout.addWidget(batch_options_group)
        layout.addStretch()
        layout.addWidget(self.ai_generate_btn)

        # 初始化提示词
        self._update_ai_prompt()

        return tab

    def _on_generation_mode_changed(self):
        """生成模式改变时的处理"""
        is_batch_mode = self.batch_mode_radio.isChecked()
        self.batch_count_spin.setEnabled(is_batch_mode)
        self.batch_options_group.setEnabled(is_batch_mode)
        
        if is_batch_mode:
            self.ai_generate_btn.setText("批量生成图像")
        else:
            self.ai_generate_btn.setText("生成图像")

    def _get_selected_option(self, radio_buttons):
        """获取选中的单选按钮文本"""
        for rb in radio_buttons:
            if rb.isChecked():
                return rb.text()
        return ""

    def _update_ai_prompt(self):
        """根据选择自动更新提示词"""
        if not hasattr(self, 'ai_target_radios') or not self.ai_target_radios:
            return
            
        target = self._get_selected_option(self.ai_target_radios)
        weather = self._get_selected_option(self.ai_weather_radios)
        scene = self._get_selected_option(self.ai_scene_radios)
        
        if target and weather and scene:
            try:
                positive_prompt, negative_prompt = self.ai_client.build_prompt(
                    military_target=target,
                    weather=weather,
                    scene=scene
                )
                if positive_prompt:
                    self.ai_prompt_text.setPlaceholderText(positive_prompt[:200] + "...")
            except Exception as e:
                print(f"自动生成提示词失败: {str(e)}")

    def _build_prompt(self):
        """手动构建提示词"""
        target = self._get_selected_option(self.ai_target_radios)
        weather = self._get_selected_option(self.ai_weather_radios)
        scene = self._get_selected_option(self.ai_scene_radios)
        
        if not all([target, weather, scene]):
            QMessageBox.warning(self, "警告", "请先选择军事目标、天气和场景")
            return
        
        # 显示加载状态
        self.status_label.setText("正在构建提示词...")
        
        # 使用线程来避免UI卡住
        self.prompt_worker = PromptWorker(self.ai_client, target, weather, scene, self.ai_prompt_text.toPlainText())
        self.prompt_worker.finished.connect(self._on_prompt_built)
        self.prompt_worker.error.connect(self._on_prompt_error)
        self.prompt_worker.start()

    def _on_prompt_built(self, positive_prompt, negative_prompt):
        """提示词构建完成"""
        if positive_prompt:
            self.ai_prompt_text.setPlainText(positive_prompt)
            self.status_label.setText("提示词已生成")
        else:
            self.status_label.setText("提示词生成失败")

    def _on_prompt_error(self, error_message):
        """提示词构建错误"""
        self.status_label.setText(f"提示词生成失败: {error_message}")
        QMessageBox.critical(self, "错误", f"提示词生成失败: {error_message}")

    def _optimize_prompt(self):
        """优化提示词"""
        current_prompt = self.ai_prompt_text.toPlainText().strip()
        if not current_prompt:
            QMessageBox.warning(self, "警告", "请先输入提示词")
            return
        
        try:
            optimized_prompt = self.ai_client.optimize_prompt(current_prompt)
            if optimized_prompt and optimized_prompt != current_prompt:
                self.ai_prompt_text.setPlainText(optimized_prompt)
                self.status_label.setText("提示词已优化")
            else:
                self.status_label.setText("提示词无需优化")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"提示词优化失败: {str(e)}")

    def _generate_ai_images(self):
        """生成AI图像"""
        target = self._get_selected_option(self.ai_target_radios)
        weather = self._get_selected_option(self.ai_weather_radios)
        scene = self._get_selected_option(self.ai_scene_radios)
        
        if not all([target, weather, scene]) and not self._is_batch_mode_with_random():
            QMessageBox.warning(self, "警告", "请先选择军事目标、天气和场景，或在批量模式下启用随机选项")
            return
        
        if self.batch_mode_radio.isChecked():
            self._start_batch_generation()
        else:
            self._start_single_generation()

    def _is_batch_mode_with_random(self):
        """检查是否为批量模式且启用了随机选项"""
        if not self.batch_mode_radio.isChecked():
            return False
        return (self.random_target_cb.isChecked() or 
                self.random_weather_cb.isChecked() or 
                self.random_scene_cb.isChecked())

    def _start_single_generation(self):
        """开始单张生成"""
        target = self._get_selected_option(self.ai_target_radios)
        weather = self._get_selected_option(self.ai_weather_radios)
        scene = self._get_selected_option(self.ai_scene_radios)
        
        generation_params = {
            'military_target': target,
            'weather': weather,
            'scene': scene,
            'custom_prompt': self.ai_prompt_text.toPlainText().strip(),
            'num_images': self.num_images_spin.value(),
            'steps': self.steps_spin.value(),
            'cfg_scale': self.cfg_spin.value(),
            'seed': self.seed_spin.value(),
            'width': 512,
            'height': 512,
            'scheduler_name': self.sampler_combo.currentText(),
            'save_images': True,
            'generate_annotations': True
        }
        
        # 禁用生成按钮
        self.ai_generate_btn.setEnabled(False)
        self.ai_generate_btn.setText("生成中...")
        
        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 不确定进度
        
        # 启动生成工作线程
        self.generation_worker = AIGenerationWorker(self.ai_client, generation_params)
        self.generation_worker.progress.connect(self._on_generation_progress)
        self.generation_worker.finished.connect(self._on_generation_finished)
        self.generation_worker.error.connect(self._on_generation_error)
        self.generation_worker.start()

    def _start_batch_generation(self):
        """开始批量生成"""
        import random
        
        # 准备批量配置
        batch_configs = []
        batch_count = self.batch_count_spin.value()
        
        # 可选项列表
        targets = ["坦克", "战机", "舰艇"]
        weathers = ["雨天", "雪天", "大雾", "夜间"]
        scenes = ["城市", "岛屿", "乡村"]
        
        # 获取当前选择
        current_target = self._get_selected_option(self.ai_target_radios)
        current_weather = self._get_selected_option(self.ai_weather_radios)
        current_scene = self._get_selected_option(self.ai_scene_radios)
        
        for i in range(batch_count):
            # 确定本次生成的参数
            if self.random_target_cb.isChecked():
                target = random.choice(targets)
            else:
                target = current_target
                
            if self.random_weather_cb.isChecked():
                weather = random.choice(weathers)
            else:
                weather = current_weather
                
            if self.random_scene_cb.isChecked():
                scene = random.choice(scenes)
            else:
                scene = current_scene
            
            # 种子处理
            if self.random_seed_cb.isChecked():
                seed = -1  # 随机种子
            else:
                seed = self.seed_spin.value()
            
            config = {
                'military_target': target,
                'weather': weather,
                'scene': scene,
                'custom_prompt': self.ai_prompt_text.toPlainText().strip(),
                'num_images': self.num_images_spin.value(),
                'steps': self.steps_spin.value(),
                'cfg_scale': self.cfg_spin.value(),
                'seed': seed,
                'width': 512,
                'height': 512,
                'scheduler_name': self.sampler_combo.currentText(),
                'save_images': True,
                'generate_annotations': True
            }
            batch_configs.append(config)
        
        # 禁用生成按钮
        self.ai_generate_btn.setEnabled(False)
        self.ai_generate_btn.setText("批量生成中...")
        
        # 显示进度条
        self.progress_bar.setVisible(True)
        self.batch_progress_bar.setVisible(True)
        self.progress_bar.setRange(0, batch_count)
        self.batch_progress_bar.setRange(0, batch_count)
        
        # 清空当前图片列表
        self.current_images = []
        self.current_image_index = 0
        self._update_image_display()
        
        # 启动批量生成工作线程
        self.batch_worker = BatchGenerationWorker(self.ai_client, batch_configs)
        self.batch_worker.progress.connect(self._on_batch_progress)
        self.batch_worker.batch_progress.connect(self._on_batch_progress_update)
        self.batch_worker.finished.connect(self._on_batch_finished)
        self.batch_worker.error.connect(self._on_batch_error)
        self.batch_worker.image_generated.connect(self._on_batch_image_generated)
        self.batch_worker.start()

    def _on_batch_progress(self, message, current, total):
        """批量生成进度更新"""
        self.status_label.setText(message)
        self.progress_label.setText(f"总进度: {current}/{total}")

    def _on_batch_progress_update(self, current, total):
        """批量进度条更新"""
        self.batch_progress_bar.setValue(current)
        self.batch_progress_label.setText(f"批次进度: {current}/{total}")

    def _on_batch_image_generated(self, result):
        """批量生成中单张图片完成"""
        images = result.get('images', [])
        for image_info in images:
            if image_info.get('file_path'):
                self.current_images.append(image_info)
        
        # 如果是第一张图片，立即显示
        if len(self.current_images) == 1:
            self.current_image_index = 0
            self._update_image_display()

    def _on_batch_finished(self, result):
        """批量生成完成"""
        self.progress_bar.setVisible(False)
        self.batch_progress_bar.setVisible(False)
        self.ai_generate_btn.setEnabled(True)
        self.ai_generate_btn.setText("批量生成图像")
        
        total_configs = result.get('total_configs', 0)
        successful_count = result.get('successful_count', 0)
        failed_count = result.get('failed_count', 0)
        
        self.status_label.setText(f"批量生成完成！成功: {successful_count}, 失败: {failed_count}")
        self.progress_label.setText(f"总进度: {total_configs}/{total_configs}")
        self.batch_progress_label.setText(f"批次进度: {total_configs}/{total_configs}")
        
        # 更新图片显示
        if self.current_images:
            self.current_image_index = 0
            self._update_image_display()
        
        QMessageBox.information(
            self, "批量生成完成", 
            f"批量生成完成！\n成功生成: {successful_count} 组\n失败: {failed_count} 组\n总图片数: {len(self.current_images)}"
        )

    def _on_batch_error(self, error_message):
        """批量生成错误"""
        self.progress_bar.setVisible(False)
        self.batch_progress_bar.setVisible(False)
        self.ai_generate_btn.setEnabled(True)
        self.ai_generate_btn.setText("批量生成图像")
        
        self.status_label.setText(f"批量生成失败: {error_message}")
        QMessageBox.critical(self, "批量生成失败", f"批量生成失败:\n{error_message}")

    def _on_generation_progress(self, message):
        """生成进度更新"""
        self.status_label.setText(message)

    def _on_generation_finished(self, result):
        """生成完成"""
        self.progress_bar.setVisible(False)
        self.ai_generate_btn.setEnabled(True)
        self.ai_generate_btn.setText("生成图像")
        
        generation_id = result.get('generation_id', '')
        images = result.get('images', [])
        
        if images:
            self.status_label.setText(f"生成完成！生成ID: {generation_id}")
            self.progress_label.setText(f"进度: {len(images)}/{len(images)}")
            
            # 更新图片列表
            self.current_images = [img for img in images if img.get('file_path')]
            self.current_image_index = 0
            self._update_image_display()
            
            QMessageBox.information(self, "成功", f"成功生成 {len(images)} 张图像！")
        else:
            self.status_label.setText("生成完成，但未获得图像")

    def _on_generation_error(self, error_message):
        """生成错误"""
        self.progress_bar.setVisible(False)
        self.ai_generate_btn.setEnabled(True)
        self.ai_generate_btn.setText("生成图像")
        
        self.status_label.setText(f"生成失败: {error_message}")
        QMessageBox.critical(self, "生成失败", f"图像生成失败:\n{error_message}")

    def _show_previous_image(self):
        """显示上一张图片"""
        if self.current_images and self.current_image_index > 0:
            self.current_image_index -= 1
            self._update_image_display()

    def _show_next_image(self):
        """显示下一张图片"""
        if self.current_images and self.current_image_index < len(self.current_images) - 1:
            self.current_image_index += 1
            self._update_image_display()

    def _update_image_display(self):
        """更新图片显示"""
        if not self.current_images:
            self.preview_label.clear()
            self.preview_label.setText("生成的图像将在此处显示")
            self.image_info_label.setText("0/0")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            return
        
        # 显示当前图片
        current_image = self.current_images[self.current_image_index]
        file_path = current_image.get('file_path')
        
        # 调试信息
        print(f"尝试显示图片: {file_path}")
        print(f"文件是否存在: {os.path.exists(file_path) if file_path else 'None'}")
        
        if file_path:
            # 处理路径格式 - 改进的路径处理逻辑
            abs_path = self._normalize_image_path(file_path)
            
            print(f"标准化后的路径: {abs_path}")
            
            if os.path.exists(abs_path):
                try:
                    pixmap = QPixmap(abs_path)
                    if not pixmap.isNull():
                        # 获取预览区域的实际大小
                        label_size = self.preview_label.size()
                        # 确保有合理的最小尺寸
                        if label_size.width() < 100 or label_size.height() < 100:
                            label_size = self.preview_label.sizeHint()
                        
                        # 缩放图像以适应预览区域
                        scaled_pixmap = pixmap.scaled(
                            label_size,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation
                        )
                        self.preview_label.setPixmap(scaled_pixmap)
                        self.preview_label.setText("")
                        print(f"图片显示成功: {abs_path}")
                    else:
                        error_msg = f"无法加载图片: {abs_path}"
                        print(error_msg)
                        self.preview_label.setText(error_msg)
                except Exception as e:
                    error_msg = f"显示图像失败: {str(e)}"
                    print(error_msg)
                    self.preview_label.setText(error_msg)
            else:
                error_msg = f"图片文件不存在: {abs_path}"
                print(error_msg)
                self.preview_label.setText(error_msg)
        else:
            self.preview_label.setText("图片路径为空")
        
        # 更新导航信息
        total_images = len(self.current_images)
        current_num = self.current_image_index + 1
        self.image_info_label.setText(f"{current_num}/{total_images}")
        
        # 更新按钮状态
        self.prev_btn.setEnabled(self.current_image_index > 0)
        self.next_btn.setEnabled(self.current_image_index < total_images - 1)

    def _normalize_image_path(self, file_path):
        """标准化图片路径"""
        if not file_path:
            return ""
        
        # 如果已经是绝对路径，直接返回
        if os.path.isabs(file_path):
            return file_path
        
        # 处理相对路径
        # 尝试不同的基础路径
        possible_bases = [
            "",  # 当前工作目录
            "../",  # 上级目录
            "backend/",  # backend目录
            "../backend/",  # 上级的backend目录
        ]
        
        for base in possible_bases:
            test_path = os.path.join(base, file_path)
            abs_test_path = os.path.abspath(test_path)
            if os.path.exists(abs_test_path):
                print(f"找到图片文件: {abs_test_path}")
                return abs_test_path
        
        # 如果都找不到，返回原始的绝对路径
        return os.path.abspath(file_path)

    def _test_image_display(self):
        """测试图片显示功能"""
        # 查找已生成的图片文件
        test_paths = [
            "data/generated/ai_generated/",
            "../data/generated/ai_generated/",
            "backend/data/generated/ai_generated/",
            "../backend/data/generated/ai_generated/"
        ]
        
        found_images = []
        for base_path in test_paths:
            abs_base_path = os.path.abspath(base_path)
            print(f"检查路径: {abs_base_path}")
            if os.path.exists(abs_base_path):
                print(f"路径存在，查找PNG文件...")
                for file in os.listdir(abs_base_path):
                    if file.endswith('.png'):
                        full_path = os.path.join(abs_base_path, file)
                        found_images.append({
                            'file_path': full_path,
                            'filename': file
                        })
                        print(f"找到图片: {file}")
                        if len(found_images) >= 5:  # 最多找5张测试
                            break
                break
        
        if found_images:
            self.current_images = found_images
            self.current_image_index = 0
            self._update_image_display()
            self.status_label.setText(f"找到 {len(found_images)} 张测试图片")
            print(f"测试图片加载完成，共 {len(found_images)} 张")
        else:
            self.status_label.setText("未找到测试图片")
            print("未找到任何测试图片")
            QMessageBox.information(self, "测试", "未找到已生成的图片文件进行测试")

    def closeEvent(self, event):
        """窗口关闭事件"""
        # 检查是否有正在运行的任务
        running_tasks = []
        if self.generation_worker and self.generation_worker.isRunning():
            running_tasks.append("单张生成")
        if self.batch_worker and self.batch_worker.isRunning():
            running_tasks.append("批量生成")
        
        if running_tasks:
            reply = QMessageBox.question(
                self, "确认退出", 
                f"以下任务正在进行中：{', '.join(running_tasks)}\n确定要退出吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            
            # 终止所有工作线程
            if self.generation_worker and self.generation_worker.isRunning():
                self.generation_worker.terminate()
                self.generation_worker.wait()
            if self.batch_worker and self.batch_worker.isRunning():
                self.batch_worker.terminate()
                self.batch_worker.wait()
        
        if hasattr(self, 'ai_client'):
            self.ai_client.close()
        
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
