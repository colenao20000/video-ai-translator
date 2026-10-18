"""Main window for the vTranslate Studio desktop application."""

import os
import sys
import time
import subprocess
import tempfile
import threading
from typing import Optional, List
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTabWidget, QFileDialog, QMessageBox, QStatusBar, QLabel, QPushButton,
    QProgressBar, QInputDialog
)
from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtGui import QDesktopServices

from ..app.dependency_container import DependencyContainer
from ..domain.project import Project
from ..domain.characters import Character
from ..domain.segments import SubtitleSegment
from ..domain.enums import FlipMode, SpeakerGender, SpeakerAgeGroup
from .styles import DARK_THEME_QSS
from .i18n import tr
from .widgets.video_preview import VideoPreviewWidget
from .widgets.character_panel import CharacterPanel
from .widgets.subtitle_panel import SubtitlePanel
from .widgets.blur_editor import BlurEditorWidget
from .widgets.timeline import TimelineWidget
from .dialogs.settings_dialog import SettingsDialog
from ..video.metadata import VideoMetadataService
from ..ai.diarization import DiarizationService
from ..ai.character_resolver import CharacterResolver
from ..ai.stt import SttService
from ..ai.translation import TranslationService
from ..ai.vision_character import VisionCharacterService
from ..services.nine_router_service import NineRouterService
from ..tts.engine import TtsEngine
from ..render.pipeline import RenderPipeline


class WorkerThread(QThread):
    progress = Signal(int, str)
    finished_success = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, task_func, parent=None, task_args=None, task_kwargs=None):
        super().__init__(parent)
        self.task_func = task_func
        self.task_args = task_args or ()
        self.task_kwargs = task_kwargs or {}
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self):
        try:
            import inspect
            kwargs = dict(self.task_kwargs)
            try:
                sig = inspect.signature(self.task_func)
                accepts_progress = (
                    "progress_callback" in sig.parameters or
                    any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                )
            except Exception:
                accepts_progress = True

            def report_progress(percent, message):
                if self._cancel_event.is_set():
                    raise WorkerCancelled()
                self.progress.emit(percent, message)

            if accepts_progress:
                kwargs["progress_callback"] = report_progress

            res = self.task_func(*self.task_args, **kwargs)
            if self._cancel_event.is_set():
                raise WorkerCancelled()
            self.finished_success.emit(res)
        except WorkerCancelled:
            self.cancelled.emit()
        except Exception as e:
            if self._cancel_event.is_set():
                self.cancelled.emit()
            else:
                self.failed.emit(str(e))


class WorkerCancelled(BaseException):
    """Internal cancellation signal that passes through broad `except Exception` handlers."""


class MainWindow(QMainWindow):
    def __init__(self, container: DependencyContainer):
        super().__init__()
        self.container = container
        self.project = Project()
        self._active_workers: List[WorkerThread] = []
        self._suppress_roi_sync = False

        # Initialize services
        self.metadata_service = VideoMetadataService(self.container.app_settings.ffprobe_path)
        self.diarization_service = DiarizationService(self.container.cache_service)
        self.stt_service = SttService(
            self.container.cache_service,
            engine=self.container.app_settings.asr_engine,
            firered_model_dir=self.container.app_settings.firered_model_dir,
            ffmpeg_path=self.container.app_settings.ffmpeg_path,
            temp_dir=self.container.app_settings.temp_dir,
        )
        self.translation_service = TranslationService(self.container.cache_service, self.container.keychain_service)
        self.tts_engine = TtsEngine(
            temp_dir=self.container.app_settings.temp_dir,
            ffmpeg_path=self.container.app_settings.ffmpeg_path,
            vieneu_model_dir=self.container.app_settings.vieneu_model_dir,
        )
        self.render_pipeline = RenderPipeline(
            self.container.app_settings.ffmpeg_path,
            self.container.app_settings.ffprobe_path,
        )

        # Audio player for voice preview
        self.preview_player = QMediaPlayer(self)
        self.preview_audio = QAudioOutput(self)
        self.preview_player.setAudioOutput(self.preview_audio)

        self._init_window()
        self._init_menus()
        self._init_ui()
        self._connect_signals()

        # Check 9router local connection
        self._check_9router_status()

    def _init_window(self) -> None:
        self.setWindowTitle("vTranslate Studio")
        self.resize(1420, 900)
        self.setStyleSheet(DARK_THEME_QSS)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(tr("common.ready"))

    def _run_worker(
        self, task_func, on_success, on_progress=None, on_failed=None,
        on_cancelled=None, *args, **kwargs
    ) -> Optional[WorkerThread]:
        active = next((worker for worker in self._active_workers if worker.isRunning()), None)
        if active:
            error = "Đang có một tác vụ khác chạy. Hãy đợi hoặc hủy tác vụ đó rồi thử lại."
            if on_failed:
                on_failed(error)
            else:
                QMessageBox.information(self, "Tác vụ đang chạy", error)
            return None
        worker = WorkerThread(task_func, parent=self, task_args=args, task_kwargs=kwargs)
        self._active_workers.append(worker)
        self._set_busy_ui(True)

        if on_progress:
            worker.progress.connect(on_progress)
        if on_success:
            worker.finished_success.connect(on_success)
        if on_failed:
            worker.failed.connect(on_failed)
        else:
            worker.failed.connect(lambda err: QMessageBox.critical(self, tr("common.task_error"), str(err)))
        if on_cancelled:
            worker.cancelled.connect(on_cancelled)
        else:
            worker.cancelled.connect(lambda: self.status_bar.showMessage("Đã hủy tác vụ."))

        def _cleanup():
            if worker in self._active_workers:
                self._active_workers.remove(worker)
            if not any(item.isRunning() for item in self._active_workers):
                self._set_busy_ui(False)

        worker.finished.connect(_cleanup)
        worker.start()
        return worker

    def _set_busy_ui(self, busy: bool) -> None:
        if hasattr(self, "tabs"):
            self.tabs.setEnabled(not busy)
        if hasattr(self, "timeline_widget"):
            self.timeline_widget.setEnabled(not busy)
        if self.menuBar():
            self.menuBar().setEnabled(not busy)
        for control_name in (
            "btn_import", "btn_export_video", "btn_quick_render", "btn_ocr_subtitle"
        ):
            control = getattr(self, control_name, None)
            if control is not None:
                control.setEnabled(not busy)

    def _init_menus(self) -> None:
        menu_bar = self.menuBar()

        # File menu
        menu_file = menu_bar.addMenu(tr("menu.file"))
        act_open = menu_file.addAction(tr("menu.open_video"))
        act_open.triggered.connect(self._on_import_video)
        act_open_project = menu_file.addAction("Mở dự án đã lưu…")
        act_open_project.triggered.connect(self._on_open_project)
        act_save = menu_file.addAction(tr("common.save"))
        act_save.triggered.connect(self._on_save_project)
        act_export = menu_file.addAction("Xuất video")
        act_export.triggered.connect(self._on_export_current_project)
        act_clear = menu_file.addAction(tr("menu.clear_cache"))
        act_clear.triggered.connect(self._on_clear_cache)
        menu_file.addSeparator()
        act_exit = menu_file.addAction(tr("menu.exit"))
        act_exit.triggered.connect(self.close)

        # AI menu
        menu_ai = menu_bar.addMenu(tr("menu.ai"))
        act_sep = menu_ai.addAction("🎵 Tách giọng & Nhạc nền (Demucs)")
        act_sep.triggered.connect(self._on_separate_audio)
        act_hybrid = menu_ai.addAction("⚡ Trích xuất Kép (ASR + OCR)")
        act_hybrid.triggered.connect(self._on_extract_hybrid)
        act_ocr = menu_ai.addAction("Quét OCR, dịch và xuất phụ đề")
        act_ocr.triggered.connect(self._on_ocr_subtitle_pipeline)
        act_stt = menu_ai.addAction("🎙️ " + tr("menu.stt"))
        act_stt.triggered.connect(self._on_extract_ai)
        act_diar = menu_ai.addAction("🎭 " + tr("menu.diarization"))
        act_diar.triggered.connect(lambda checked=False: self._on_auto_assign_voices())
        act_trans = menu_ai.addAction("🌐 " + tr("menu.translate"))
        act_trans.triggered.connect(self._on_translate_all)
        act_tts = menu_ai.addAction("🔊 " + tr("menu.tts"))
        act_tts.triggered.connect(self._on_generate_tts)

        # Settings menu
        menu_settings = menu_bar.addMenu(tr("menu.settings"))
        act_sub_style = menu_settings.addAction("🎨 Phông chữ & Kiểu dáng Phụ đề…")
        act_sub_style.triggered.connect(self._open_subtitle_style_settings)
        act_pref = menu_settings.addAction(tr("settings.title"))
        act_pref.triggered.connect(self._on_open_settings)

    def _init_ui(self) -> None:
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(12, 10, 12, 10)
        main_layout.setSpacing(10)

        # Workspace header and primary actions.
        self.top_bar = QWidget()
        self.top_bar.setObjectName("topBar")
        self.top_bar.setMinimumHeight(58)
        h_top = QHBoxLayout(self.top_bar)
        h_top.setContentsMargins(12, 6, 12, 6)
        h_top.setSpacing(10)

        # Logo
        lbl_logo = QLabel("vTranslate Studio")
        lbl_logo.setObjectName("logoLabel")
        h_top.addWidget(lbl_logo)

        # Current video badge
        self.label_video_info = QLabel("Chưa mở video")
        self.label_video_info.setObjectName("videoBadge")
        h_top.addWidget(self.label_video_info)

        # Real-time progress bar & Cancel button (shown during auto pipeline)
        self.quick_progress_bar = QProgressBar()
        self.quick_progress_bar.setFixedWidth(240)
        self.quick_progress_bar.setVisible(False)
        h_top.addWidget(self.quick_progress_bar)

        self.btn_cancel_quick = QPushButton("⏹ Hủy")
        self.btn_cancel_quick.setObjectName("dangerButton")
        self.btn_cancel_quick.setFixedHeight(28)
        self.btn_cancel_quick.setVisible(False)
        self.btn_cancel_quick.setToolTip("Dừng tiến trình tự động đang chạy")
        self.btn_cancel_quick.clicked.connect(self._on_cancel_quick_render)
        h_top.addWidget(self.btn_cancel_quick)

        self.quick_status_label = QLabel("")
        self.quick_status_label.setObjectName("mutedLabel")
        h_top.addWidget(self.quick_status_label)

        h_top.addStretch()

        # Action Buttons
        self.btn_import = QPushButton("Mở video")
        self.btn_import.setFixedHeight(34)
        self.btn_import.clicked.connect(self._on_import_video)
        h_top.addWidget(self.btn_import)

        self.btn_export_video = QPushButton("Xuất video")
        self.btn_export_video.setObjectName("secondaryButton")
        self.btn_export_video.setFixedHeight(36)
        self.btn_export_video.setToolTip("Xuất video từ phụ đề, giọng đọc và cài đặt hiện tại")
        self.btn_export_video.clicked.connect(self._on_export_current_project)
        h_top.addWidget(self.btn_export_video)

        # Main assisted workflow.
        self.btn_quick_render = QPushButton("Dịch & lồng tiếng tự động")
        self.btn_quick_render.setObjectName("primaryButton")
        self.btn_quick_render.setFixedHeight(36)
        self.btn_quick_render.setToolTip("Trích xuất lời thoại, phân tích người nói, dịch, tạo giọng và xuất video")
        self.btn_quick_render.clicked.connect(self._on_quick_render_clicked)
        h_top.addWidget(self.btn_quick_render)

        # Subtitle-only workflow.
        self.btn_ocr_subtitle = QPushButton("OCR · dịch · phụ đề")
        self.btn_ocr_subtitle.setObjectName("accentButton")
        self.btn_ocr_subtitle.setFixedHeight(36)
        self.btn_ocr_subtitle.setToolTip(
            "Quét phụ đề trong vùng đã chọn, dịch và xuất video với âm thanh nguồn."
        )
        self.btn_ocr_subtitle.clicked.connect(self._on_ocr_subtitle_pipeline)
        h_top.addWidget(self.btn_ocr_subtitle)


        main_layout.addWidget(self.top_bar)

        # Main editor workspace.
        v_splitter = QSplitter(Qt.Vertical)

        # Upper section: 3 Clean Left Tabs + Video Preview on Right
        h_splitter = QSplitter(Qt.Horizontal)

        # Left: Tabs panel (Phụ đề, Nhân vật, Xóa chữ gốc) - mở rộng tối đa hiển thị đầy đủ mọi cột
        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(500)

        self.subtitle_panel = SubtitlePanel()
        self.tabs.addTab(self.subtitle_panel, "Phụ đề")

        self.character_panel = CharacterPanel()
        self.tabs.addTab(self.character_panel, "Nhân vật & giọng")

        self.blur_panel = BlurEditorWidget()
        self.tabs.addTab(self.blur_panel, "Khung hình & logo")

        h_splitter.addWidget(self.tabs)

        # Keep preview and controls usable as the window is resized.
        self.preview_widget = VideoPreviewWidget()
        self.preview_widget.setMinimumWidth(360)
        h_splitter.addWidget(self.preview_widget)

        # Split the editor and preview proportionally.
        h_splitter.setStretchFactor(0, 3)
        h_splitter.setStretchFactor(1, 2)
        h_splitter.setSizes([760, 560])
        v_splitter.addWidget(h_splitter)

        # Timeline workspace.
        self.timeline_widget = TimelineWidget()
        v_splitter.addWidget(self.timeline_widget)
        v_splitter.setStretchFactor(0, 1)
        v_splitter.setStretchFactor(1, 1)
        v_splitter.setSizes([590, 300])

        main_layout.addWidget(v_splitter)
        self.setCentralWidget(main_widget)

    def _connect_signals(self) -> None:
        # Sync preview & timeline
        self.preview_widget.position_changed.connect(self.timeline_widget.set_current_time)
        self.preview_widget.duration_changed.connect(self.timeline_widget.set_duration)
        self.timeline_widget.seek_requested.connect(self.preview_widget.seek)
        self.timeline_widget.segment_selected.connect(self._on_timeline_segment_selected)

        # Preview ROI box sync with blur panel & video transform / logo / audio separation
        self.preview_widget.roi_coordinates_changed.connect(self._on_roi_coordinates_changed)
        self.blur_panel.settings_changed.connect(self._on_blur_changed)
        self.blur_panel.transform_changed.connect(self._on_transform_changed)
        self.blur_panel.logo_changed.connect(self._on_logo_changed)
        self.blur_panel.separate_audio_requested.connect(self._on_separate_audio)
        self.blur_panel.play_audio_requested.connect(self._on_play_audio_file)

        # Character panel
        self.character_panel.character_changed.connect(self._on_character_changed)
        self.character_panel.character_deleted.connect(self._on_character_deleted)
        self.character_panel.reset_requested.connect(self._on_reset_characters)
        self.character_panel.smart_assign_requested.connect(self._on_smart_assign_dialogue)
        self.character_panel.preview_voice_requested.connect(self._preview_voice)

        # Subtitle Panel
        self.subtitle_panel.translation_edited.connect(self._on_translation_edited)
        self.subtitle_panel.character_reassigned.connect(self._on_character_reassigned)
        self.subtitle_panel.speech_style_edited.connect(self._on_speech_style_edited)
        self.subtitle_panel.extract_hybrid_requested.connect(self._on_extract_hybrid)
        self.subtitle_panel.extract_ai_requested.connect(self._on_extract_ai)
        self.subtitle_panel.extract_ocr_requested.connect(self._on_extract_ocr)
        self.subtitle_panel.toggle_ocr_roi_requested.connect(self._on_toggle_ocr_roi)
        self.subtitle_panel.ocr_preset_selected.connect(self._on_ocr_preset_selected)
        self.subtitle_panel.translate_all_requested.connect(self._on_translate_all)
        self.subtitle_panel.generate_tts_requested.connect(self._on_generate_tts)
        self.subtitle_panel.render_requested.connect(self._on_export_current_project)
        self.subtitle_panel.sub_style_requested.connect(self._open_subtitle_style_settings)

        # Video preview OCR ROI interaction
        self.preview_widget.btn_ocr_box.toggled.connect(self._on_preview_ocr_box_toggled)

    def _on_import_video(self) -> None:
        if any(worker.isRunning() for worker in self._active_workers):
            QMessageBox.information(self, "Đang xử lý", "Hãy đợi hoặc hủy tác vụ hiện tại trước khi mở video khác.")
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self, tr("video.select_dialog"), "", "Video Files (*.mp4 *.mkv *.mov *.avi *.webm)"
        )
        if not file_path:
            return

        if self.container.app_settings.auto_save and self.project.video_path:
            try:
                self.container.project_repo.save(self.project)
            except Exception as exc:
                QMessageBox.warning(self, "Lưu dự án", f"Không thể tự lưu dự án hiện tại: {exc}")

        self.status_bar.showMessage(tr("video.loading_metadata"))

        def load_task(progress_callback=None):
            meta = self.metadata_service.probe(file_path)
            cached_stt = self.container.cache_service.get_stt(file_path)
            segs = [SubtitleSegment(**s) for s in cached_stt] if cached_stt else []
            return meta, segs

        def on_loaded(result):
            meta, segs = result
            base_name = os.path.basename(file_path)
            project = Project(
                name=os.path.splitext(base_name)[0],
                video_path=file_path,
                video_metadata=meta,
                segments=segs,
            )
            app_settings = self.container.app_settings
            project.audio_settings.original_volume = app_settings.original_volume
            project.audio_settings.dubbing_volume = app_settings.dubbing_volume
            project.audio_settings.max_tempo_stretch = app_settings.max_tempo_stretch
            project.subtitle_style.font_name = app_settings.font_name
            project.subtitle_style.font_size = app_settings.font_size
            project.subtitle_style.primary_color = app_settings.primary_color
            project.subtitle_style.outline_color = app_settings.outline_color
            project.subtitle_style.outline_width = app_settings.outline_width
            project.subtitle_style.margin_v = app_settings.margin_v
            self._activate_project(project)
            self.status_bar.showMessage(f"Đã mở: {base_name}")

        self._run_worker(load_task, on_success=on_loaded)

    def _activate_project(self, project: Project) -> None:
        """Load project state into the preview and all editor panels."""
        self.project = project
        metadata = project.video_metadata
        width = metadata.width if metadata else 1920
        height = metadata.height if metadata else 1080

        self._suppress_roi_sync = True
        try:
            self.preview_widget.load_video(project.video_path, width=width, height=height)
        finally:
            self._suppress_roi_sync = False

        if project.blur_settings.width <= 0 or project.blur_settings.height <= 0:
            default_roi = self.preview_widget.get_roi()
            project.blur_settings.x = default_roi["x"]
            project.blur_settings.y = default_roi["y"]
            project.blur_settings.width = default_roi["width"]
            project.blur_settings.height = default_roi["height"]
        self.preview_widget.set_roi_settings(project.blur_settings)
        self.blur_panel.set_settings(project.blur_settings)
        self.blur_panel.set_transform_settings(project.transform_settings)
        self.blur_panel.set_logo_settings(project.logo_settings)
        self.preview_widget.set_logo_settings(project.logo_settings)
        self.preview_widget.set_subtitle_style(project.subtitle_style)
        self.preview_widget.apply_transform(
            project.transform_settings.zoom_ratio, project.transform_settings.flip_mode
        )
        self.preview_widget.set_original_volume(project.audio_settings.original_volume)
        self.preview_widget.set_dubbing_volume(project.audio_settings.dubbing_volume)

        if (
            project.video_path
            and os.path.exists(project.video_path)
            and (not project.isolated_vocals_path or not os.path.exists(project.isolated_vocals_path))
        ):
            cached_vocals, cached_bgm = self.container.separator_service.get_cached_paths(project.video_path)
            if os.path.exists(cached_vocals) and os.path.exists(cached_bgm):
                project.isolated_vocals_path = cached_vocals
                project.isolated_background_path = cached_bgm

        vocals = project.isolated_vocals_path if project.isolated_vocals_path and os.path.exists(project.isolated_vocals_path) else None
        bgm = project.isolated_background_path if project.isolated_background_path and os.path.exists(project.isolated_background_path) else None
        if not vocals:
            project.isolated_vocals_path = None
        if not bgm:
            project.isolated_background_path = None
        separation_method = (
            self.container.separator_service.get_separation_method(vocals)
            if vocals else None
        )
        self.blur_panel.set_separation_status(vocals, bgm, separation_method)
        if bgm:
            self.preview_widget.set_background_preview_audio(bgm)

        duration = metadata.duration_sec if metadata else 0
        minutes, seconds = divmod(int(duration), 60)
        base_name = os.path.basename(project.video_path)
        self.label_video_info.setText(f"{base_name} · {minutes:02d}:{seconds:02d}")
        self.label_video_info.setToolTip(project.video_path)
        self.character_panel.set_characters(project.characters)
        self.subtitle_panel.set_data(project.segments, project.characters)
        self.timeline_widget.set_data(project.segments, project.characters)
        self.preview_widget.set_preview_segments(project.segments)

    def _on_open_project(self) -> None:
        if any(worker.isRunning() for worker in self._active_workers):
            QMessageBox.information(self, "Đang xử lý", "Hãy đợi hoặc hủy tác vụ hiện tại trước khi mở dự án khác.")
            return
        projects = self.container.project_repo.list_all()
        if not projects:
            QMessageBox.information(self, "Mở dự án", "Chưa có dự án đã lưu.")
            return

        labels = [
            f"{row['name']} · {os.path.basename(row['video_path'])} · {time.strftime('%Y-%m-%d', time.localtime(row['updated_at']))}"
            for row in projects
        ]
        selected, accepted = QInputDialog.getItem(
            self, "Mở dự án", "Chọn dự án:", labels, 0, False
        )
        if not accepted:
            return
        project_index = labels.index(selected)
        project = self.container.project_repo.get(projects[project_index]["project_id"])
        if not project:
            QMessageBox.warning(self, "Mở dự án", "Không đọc được dữ liệu dự án đã chọn.")
            return

        if not project.video_path or not os.path.exists(project.video_path):
            relinked, _ = QFileDialog.getOpenFileName(
                self, "Tìm video nguồn cho dự án", "", "Video Files (*.mp4 *.mkv *.mov *.avi *.webm)"
            )
            if not relinked:
                return
            project.video_path = relinked
            project.video_metadata = None

        if self.container.app_settings.auto_save and self.project.video_path:
            try:
                self.container.project_repo.save(self.project)
            except Exception as exc:
                QMessageBox.warning(self, "Lưu dự án", f"Không thể tự lưu dự án hiện tại: {exc}")

        project_snapshot = project.model_copy(deep=True)

        def load_saved_project(progress_callback=None):
            if not project_snapshot.video_metadata or project_snapshot.video_metadata.path != project_snapshot.video_path:
                project_snapshot.video_metadata = self.metadata_service.probe(project_snapshot.video_path)
            return project_snapshot

        def on_project_loaded(loaded_project):
            self._activate_project(loaded_project)
            self.status_bar.showMessage(f"Đã mở dự án: {loaded_project.name}")

        self._run_worker(
            load_saved_project,
            on_success=on_project_loaded,
            on_failed=lambda err: QMessageBox.critical(self, "Mở dự án", str(err)),
        )

    def _on_save_project(self) -> None:
        if not self.project.video_path:
            QMessageBox.warning(self, tr("common.warning"), tr("common.no_video_msg"))
            return
        try:
            self.container.project_repo.save(self.project)
            self.status_bar.showMessage(tr("common.project_saved"))
            QMessageBox.information(self, tr("common.success"), tr("common.project_saved"))
        except Exception as e:
            QMessageBox.warning(self, tr("common.error"), f"Không thể lưu dự án: {e}")

    def closeEvent(self, event) -> None:
        if any(worker.isRunning() for worker in self._active_workers):
            QMessageBox.information(
                self, "Đang xử lý", "Hãy đợi hoặc hủy các tác vụ đang chạy trước khi đóng ứng dụng."
            )
            event.ignore()
            return
        if self.container.app_settings.auto_save and self.project.video_path:
            try:
                self.container.project_repo.save(self.project)
            except Exception as exc:
                QMessageBox.critical(self, "Không thể lưu dự án", str(exc))
                event.ignore()
                return
        event.accept()

    def _on_clear_cache(self) -> None:
        reply = QMessageBox.question(
            self,
            "Xác nhận xoá Cache",
            "Bạn có chắc chắn muốn dọn dẹp sạch toàn bộ bộ nhớ đệm (Cache)?\n\n"
            "Các tệp âm thanh TTS đã sinh, bản dịch tạm và ảnh trích xuất tạm thời sẽ được dọn dẹp sạch sẽ.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.container.cache_service.clear_all_cache([self.container.app_settings.temp_dir])
            self.status_bar.showMessage("Đã xoá sạch toàn bộ Cache!")
            QMessageBox.information(self, "Thành công", "✅ Đã xoá sạch toàn bộ bộ nhớ đệm thành công!")

    def _on_open_settings(self) -> None:
        dlg = SettingsDialog(
            self.container.app_settings,
            self.container.keychain_service,
            cache_service=self.container.cache_service,
            parent=self
        )
        if dlg.exec():
            self.stt_service.set_engine(self.container.app_settings.asr_engine)
            self.stt_service.firered_manager.model_dir = self.container.app_settings.firered_model_dir

            # Cập nhật cấu hình TTS Engine
            self.tts_engine.reload_config(
                ffmpeg_path=self.container.app_settings.ffmpeg_path,
                vieneu_model_dir=self.container.app_settings.vieneu_model_dir,
            )
            self.stt_service.ffmpeg_path = self.container.app_settings.ffmpeg_path
            self.stt_service.temp_dir = os.path.expanduser(self.container.app_settings.temp_dir)
            self.metadata_service.ffprobe_path = self.container.app_settings.ffprobe_path
            self.render_pipeline.ffmpeg_path = self.container.app_settings.ffmpeg_path
            self.render_pipeline.ffprobe_path = self.container.app_settings.ffprobe_path
            self.container.separator_service.temp_dir = os.path.expanduser(self.container.app_settings.temp_dir)
            self.container.separator_service.ffmpeg_path = self.container.app_settings.ffmpeg_path
            os.makedirs(self.container.separator_service.temp_dir, exist_ok=True)
            self.container.settings_repo.set(
                "app_settings_json", self.container.app_settings.model_dump_json()
            )

            # Sync app audio settings to project
            self.project.audio_settings.original_volume = self.container.app_settings.original_volume
            self.project.audio_settings.dubbing_volume = self.container.app_settings.dubbing_volume
            self.project.audio_settings.max_tempo_stretch = self.container.app_settings.max_tempo_stretch

            # Sync subtitle style
            self.project.subtitle_style.font_name = self.container.app_settings.font_name
            self.project.subtitle_style.font_size = self.container.app_settings.font_size
            self.project.subtitle_style.margin_v = self.container.app_settings.margin_v
            self.project.subtitle_style.outline_width = self.container.app_settings.outline_width
            self.project.subtitle_style.primary_color = self.container.app_settings.primary_color
            self.project.subtitle_style.outline_color = self.container.app_settings.outline_color
            self.project.subtitle_style.alignment = getattr(self.container.app_settings, "alignment", 2)
            self.project.subtitle_style.single_line = getattr(self.container.app_settings, "single_line", True)
            self.preview_widget.set_subtitle_style(self.project.subtitle_style)
            self.preview_widget.set_original_volume(self.project.audio_settings.original_volume)
            self.preview_widget.set_dubbing_volume(self.project.audio_settings.dubbing_volume)

            self.status_bar.showMessage("Đã cập nhật cài đặt mới!")

    def _on_separate_audio(self) -> None:
        if not self.project.video_path:
            QMessageBox.warning(self, tr("common.no_video"), tr("common.no_video_msg"))
            return

        self.status_bar.showMessage("Đang xử lý rãnh giọng nói và nhạc nền...")

        def run_sep(progress_callback=None):
            return self.container.separator_service.separate(
                self.project.video_path,
                force_refresh=True,
                progress_callback=progress_callback
            )

        def on_sep_done(result):
            vocals_path, bgm_path = result
            method = self.container.separator_service.last_method
            self.project.isolated_vocals_path = vocals_path
            self.project.isolated_background_path = bgm_path
            self.project.mark_updated()
            self.blur_panel.set_separation_status(vocals_path, bgm_path, method)
            self.preview_widget.set_background_preview_audio(bgm_path)
            status = "Đã tách bằng Demucs." if method == "demucs" else "Đã tạo rãnh lọc DSP xấp xỉ."
            self.status_bar.showMessage(status)
            v_mb = os.path.getsize(vocals_path) / (1024 * 1024)
            b_mb = os.path.getsize(bgm_path) / (1024 * 1024)
            QMessageBox.information(
                self,
                tr("common.success"),
                (
                    "Demucs đã tách rãnh giọng và nhạc nền.\n\n"
                    if method == "demucs" else
                    "Demucs chưa khả dụng; các rãnh này dùng bộ lọc DSP xấp xỉ và có thể còn lẫn giọng/nhạc.\n\n"
                )
                + f"• Rãnh giọng: {v_mb:.1f} MB\n• Rãnh nền: {b_mb:.1f} MB"
            )

        self._run_worker(
            run_sep,
            on_success=on_sep_done,
            on_progress=lambda p, msg: self.status_bar.showMessage(f"{p}% - {msg}"),
            on_failed=lambda err: QMessageBox.critical(self, tr("common.error"), str(err))
        )

    def _on_play_audio_file(self, file_path: str) -> None:
        if file_path and os.path.exists(file_path):
            self.preview_player.setSource(QUrl.fromLocalFile(file_path))
            self.preview_player.play()
            self.status_bar.showMessage(f"Đang phát: {os.path.basename(file_path)}")

    def _on_logo_changed(self, logo_st) -> None:
        self.project.logo_settings = logo_st
        self.preview_widget.set_logo_settings(logo_st)

    def _open_subtitle_style_settings(self) -> None:
        """Mở trực tiếp Tab Kiểu dáng Phụ đề để chọn phông, cỡ chữ, vị trí và gom 1 dòng."""
        dlg = SettingsDialog(
            self.container.app_settings,
            self.container.keychain_service,
            cache_service=self.container.cache_service,
            initial_tab=3,
            parent=self
        )
        if dlg.exec():
            app_st = self.container.app_settings
            self.project.subtitle_style.font_name = app_st.font_name
            self.project.subtitle_style.font_size = app_st.font_size
            self.project.subtitle_style.margin_v = app_st.margin_v
            self.project.subtitle_style.outline_width = app_st.outline_width
            self.project.subtitle_style.primary_color = app_st.primary_color
            self.project.subtitle_style.outline_color = app_st.outline_color
            self.project.subtitle_style.alignment = getattr(app_st, "alignment", 2)
            self.project.subtitle_style.single_line = getattr(app_st, "single_line", True)
            self.preview_widget.set_subtitle_style(self.project.subtitle_style)
            self.container.settings_repo.set(
                "app_settings_json", self.container.app_settings.model_dump_json()
            )
            self.status_bar.showMessage("Đã lưu cấu hình phông chữ và kiểu dáng phụ đề!")

    def _on_transform_changed(self, zoom: int, flip: FlipMode) -> None:
        self.project.transform_settings.zoom_percent = zoom
        self.project.transform_settings.flip_mode = flip
        self.preview_widget.apply_transform(zoom / 100.0, flip)

    def _on_blur_changed(self, settings) -> None:
        self.project.blur_settings = settings
        self.preview_widget.set_roi_settings(settings)

    def _on_roi_coordinates_changed(self, x: int, y: int, w: int, h: int) -> None:
        if self._suppress_roi_sync:
            return
        self.project.blur_settings.x = x
        self.project.blur_settings.y = y
        self.project.blur_settings.width = w
        self.project.blur_settings.height = h
        self.blur_panel.update_coordinates_from_preview(x, y, w, h)

    def _on_character_changed(self, char: Character) -> None:
        self.project.add_or_update_character(char)
        self.subtitle_panel.set_data(self.project.segments, self.project.characters)
        self.timeline_widget.set_data(self.project.segments, self.project.characters)

    def _on_character_deleted(self, deleted_char_id: str) -> None:
        self.project.characters = [c for c in self.project.characters if c.id != deleted_char_id]
        for seg in self.project.segments:
            if seg.character_id == deleted_char_id:
                self.project.assign_segment_character(seg.segment_id, "char_unknown")
        self.project.mark_updated()
        self.character_panel.set_characters(self.project.characters)
        self.subtitle_panel.set_data(self.project.segments, self.project.characters)
        self.timeline_widget.set_data(self.project.segments, self.project.characters)

    def _on_reset_characters(self) -> None:
        self.project.characters = []
        for seg in self.project.segments:
            self.project.assign_segment_character(seg.segment_id, "char_unknown")
        self.project.mark_updated()
        self.character_panel.set_characters(self.project.characters)
        self.subtitle_panel.set_data(self.project.segments, self.project.characters)
        self.timeline_widget.set_data(self.project.segments, self.project.characters)

    def _on_translation_edited(self, seg_id: str, new_text: str) -> None:
        self.project.update_segment_translation(seg_id, new_text)
        self.subtitle_panel.set_data(self.project.segments, self.project.characters)
        self.timeline_widget.set_data(self.project.segments, self.project.characters)
        self.preview_widget.set_preview_segments(self.project.segments)

    def _on_speech_style_edited(self, seg_id: str, new_style: str) -> None:
        seg = self.project.get_segment_by_id(seg_id)
        if not seg:
            return
        seg.update_speech_style(new_style)
        self.project.mark_updated()
        self.subtitle_panel.set_data(self.project.segments, self.project.characters)
        self.timeline_widget.set_data(self.project.segments, self.project.characters)

    def _on_timeline_segment_selected(self, seg_id: str) -> None:
        for row in range(self.subtitle_panel.table.rowCount()):
            for column in range(self.subtitle_panel.table.columnCount()):
                item = self.subtitle_panel.table.item(row, column)
                if item and item.data(Qt.UserRole) == seg_id:
                    self.subtitle_panel.table.selectRow(row)
                    self.subtitle_panel.table.scrollToItem(item)
                    return

    def _on_character_reassigned(self, seg_id: str, char_id: str) -> None:
        self.project.assign_segment_character(seg_id, char_id)
        self.subtitle_panel.set_data(self.project.segments, self.project.characters)
        self.timeline_widget.set_data(self.project.segments, self.project.characters)

    def _preview_voice(self, voice_id: str) -> None:
        self.status_bar.showMessage(f"Đang nghe thử: {voice_id}...")
        try:
            wav_path = self.tts_engine.preview_voice(voice_id)
            if wav_path and os.path.exists(wav_path):
                self.preview_player.setSource(QUrl.fromLocalFile(wav_path))
                self.preview_player.play()
                self.status_bar.showMessage(f"Giọng đọc: {voice_id}")
        except Exception as e:
            QMessageBox.warning(self, tr("common.error"), str(e))

    def _on_extract_hybrid(self) -> None:
        if not self.project.video_path:
            QMessageBox.warning(self, tr("common.no_video"), tr("common.no_video_msg"))
            return

        self.status_bar.showMessage("Đang tách âm thanh & trích xuất kép ASR + OCR...")
        project_snapshot = self.project.model_copy(deep=True)
        roi_snapshot = self.preview_widget.get_roi()

        def run_hybrid_task(progress_callback=None):
            from ..subtitle.hybrid_extractor import HybridSubtitleExtractor
            vocals_path = None
            bgm_path = None
            try:
                if progress_callback:
                    progress_callback(5, "Đang tách giọng nói và nhạc nền...")
                vocals_path, bgm_path = self.container.separator_service.separate(
                    project_snapshot.video_path,
                    force_refresh=False,
                    progress_callback=lambda p, msg: progress_callback(int(5 + p * 0.25), msg) if progress_callback else None
                )
                project_snapshot.isolated_vocals_path = vocals_path
                project_snapshot.isolated_background_path = bgm_path
            except Exception as e:
                print(f"Lỗi tách âm thanh: {e}")

            segments = HybridSubtitleExtractor.extract_and_reconcile(
                video_path=project_snapshot.video_path,
                stt_service=self.stt_service,
                vocals_wav_path=vocals_path,
                roi=roi_snapshot,
                progress_callback=lambda p, msg: progress_callback(int(30 + p * 0.70), msg) if progress_callback else None
            )
            method = (
                self.container.separator_service.get_separation_method(vocals_path)
                if vocals_path else None
            )
            return {
                "segments": segments,
                "project": project_snapshot,
                "vocals": vocals_path,
                "bgm": bgm_path,
                "separation_method": method,
            }

        def on_hybrid_completed(result):
            self.project = result["project"]
            segments = result["segments"]
            self.project.segments = segments
            self.container.cache_service.save_stt(self.project.video_path, [s.model_dump() for s in segments])
            self.blur_panel.set_separation_status(
                result["vocals"], result["bgm"], result.get("separation_method")
            )
            if result["bgm"]:
                self.preview_widget.set_background_preview_audio(result["bgm"])
            self.subtitle_panel.set_data(self.project.segments, self.project.characters)
            self.timeline_widget.set_data(self.project.segments, self.project.characters)
            self.preview_widget.set_preview_segments(self.project.segments)
            self.status_bar.showMessage(tr("common.success"))
            QMessageBox.information(
                self, tr("common.success"),
                f"Đã trích xuất {len(segments)} câu từ ASR và OCR. Hãy rà lại các câu có độ tin cậy thấp trước khi dịch."
            )

        self._run_worker(
            run_hybrid_task,
            on_success=on_hybrid_completed,
            on_progress=lambda p, msg: self.status_bar.showMessage(f"{p}% - {msg}"),
            on_failed=lambda err: QMessageBox.critical(self, tr("common.error"), str(err))
        )

    def _on_extract_ai(self) -> None:
        if not self.project.video_path:
            QMessageBox.warning(self, tr("common.no_video"), tr("common.no_video_msg"))
            return

        self.status_bar.showMessage(tr("subtitle.extracting_stt"))
        project_snapshot = self.project.model_copy(deep=True)

        def run_stt(progress_callback=None):
            audio_target = project_snapshot.isolated_vocals_path if (project_snapshot.isolated_vocals_path and os.path.exists(project_snapshot.isolated_vocals_path)) else project_snapshot.video_path
            return self.stt_service.extract_subtitles(
                audio_target,
                force_refresh=True,
                progress_callback=progress_callback
            )

        def on_stt_completed(segments: List[SubtitleSegment]):
            self.project.segments = segments
            self.container.cache_service.save_stt(self.project.video_path, [s.model_dump() for s in segments])
            self.subtitle_panel.set_data(self.project.segments, self.project.characters)
            self.timeline_widget.set_data(self.project.segments, self.project.characters)
            self.preview_widget.set_preview_segments(self.project.segments)
            self.status_bar.showMessage(tr("common.success"))
            QMessageBox.information(
                self, tr("common.success"),
                f"Đã trích xuất {len(segments)} phân đoạn lời thoại!"
            )

        self._run_worker(
            run_stt,
            on_success=on_stt_completed,
            on_progress=lambda p, msg: self.status_bar.showMessage(f"{p}% - {msg}"),
            on_failed=lambda err: QMessageBox.critical(self, tr("common.error"), str(err))
        )

    def _on_toggle_ocr_roi(self, enabled: bool) -> None:
        self.preview_widget.btn_ocr_box.setChecked(enabled)

    def _on_preview_ocr_box_toggled(self, checked: bool) -> None:
        if hasattr(self.subtitle_panel, "btn_toggle_ocr_roi"):
            self.subtitle_panel.btn_toggle_ocr_roi.setChecked(checked)

    def _on_ocr_preset_selected(self, preset_key: str) -> None:
        self.preview_widget.apply_roi_preset(preset_key)

    def _on_extract_ocr(self) -> None:
        if not self.project.video_path:
            QMessageBox.warning(self, tr("common.no_video"), tr("common.no_video_msg"))
            return

        if self.project.segments:
            res = QMessageBox.question(
                self,
                "Xác nhận quét OCR",
                f"Dự án hiện có {len(self.project.segments)} câu. Bạn có muốn quét lại toàn bộ phụ đề từ hình ảnh (OCR) để thay thế không?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if res != QMessageBox.Yes:
                return

        roi_dict = self.preview_widget.get_roi()
        project_snapshot = self.project.model_copy(deep=True)
        from ..subtitle.ocr_extractor import OcrSubtitleExtractor

        self.status_bar.showMessage("Đang quét OCR phụ đề từ video...")

        def run_ocr_full(progress_callback=None):
            return OcrSubtitleExtractor.extract_subtitles_from_video(
                project_snapshot.video_path,
                roi=roi_dict,
                sample_interval=0.35,
                progress_callback=progress_callback
            )

        def on_ocr_full_done(segments):
            self.project.segments = segments
            self.container.cache_service.save_stt(self.project.video_path, [s.model_dump() for s in segments])
            self.subtitle_panel.set_data(self.project.segments, self.project.characters)
            self.timeline_widget.set_data(self.project.segments, self.project.characters)
            self.preview_widget.set_preview_segments(self.project.segments)
            self.status_bar.showMessage(tr("common.success"))
            QMessageBox.information(
                self,
                tr("common.success"),
                f"Đã trích xuất thành công {len(segments)} câu phụ đề qua OCR!"
            )

        self._run_worker(
            run_ocr_full,
            on_success=on_ocr_full_done,
            on_progress=lambda p, msg: self.status_bar.showMessage(f"{p}% - {msg}"),
            on_failed=lambda err: QMessageBox.critical(self, tr("common.error"), str(err))
        )

    def _check_9router_status(self) -> None:
        ok, msg, models = NineRouterService.check_connection()
        if ok:
            self.subtitle_panel.lbl_9router.setText(f"9router ({len(models)} models)")
            self.subtitle_panel.lbl_9router.setObjectName("badgeOk")
        else:
            self.subtitle_panel.lbl_9router.setText("9router Offline")
            self.subtitle_panel.lbl_9router.setObjectName("badgeOff")
        self.subtitle_panel.lbl_9router.style().unpolish(self.subtitle_panel.lbl_9router)
        self.subtitle_panel.lbl_9router.style().polish(self.subtitle_panel.lbl_9router)

    def _on_vision_detect_characters(self) -> None:
        if not self.project.video_path:
            QMessageBox.warning(self, tr("common.no_video"), tr("common.no_video_msg"))
            return

        self.status_bar.showMessage(tr("character.vision_start"))
        vis_model = self.container.app_settings.vision_model
        project_snapshot = self.project.model_copy(deep=True)

        def run_vision_task(progress_callback=None):
            visual_chars = VisionCharacterService.analyze_characters_with_vision(
                project_snapshot.video_path,
                model_name=vis_model,
                progress_callback=progress_callback
            )
            chars = [char.model_copy(deep=True) for char in project_snapshot.characters]
            used_ids = {char.id for char in chars}
            used_speakers = {char.speaker_id for char in chars}
            for idx, visual_char in enumerate(visual_chars):
                base_id = visual_char.id
                suffix = 2
                while visual_char.id in used_ids:
                    visual_char.id = f"{base_id}_{suffix}"
                    suffix += 1
                if visual_char.speaker_id in used_speakers:
                    suffix = idx + 1
                    visual_char.speaker_id = f"VISUAL_{suffix:02d}"
                    while visual_char.speaker_id in used_speakers:
                        suffix += 1
                        visual_char.speaker_id = f"VISUAL_{suffix:02d}"
                used_ids.add(visual_char.id)
                used_speakers.add(visual_char.speaker_id)
                chars.append(visual_char)

            segs = [seg.model_copy(deep=True) for seg in project_snapshot.segments]
            if segs:
                if progress_callback:
                    progress_callback(70, "Đang cập nhật hồ sơ và rà phân vai...")
                segs, chars = VisionCharacterService.assign_dialogue_smart(
                    segs, chars, model_name=vis_model, progress_callback=progress_callback
                )
            return chars, segs

        def on_vision_completed(result):
            chars, segs = result
            self.project.characters = chars
            if segs:
                self.project.segments = segs
            self.character_panel.set_characters(self.project.characters)
            self.subtitle_panel.set_data(self.project.segments, self.project.characters)
            self.timeline_widget.set_data(self.project.segments, self.project.characters)

            lines_info = []
            for c in chars:
                gender_str = (
                    tr("character.gender_male") if c.gender == SpeakerGender.MALE else
                    tr("character.gender_female") if c.gender == SpeakerGender.FEMALE else
                    "Chưa rõ"
                )
                lines_info.append(f"• {c.name} ({gender_str}) -> {c.voice_id}")

            self.status_bar.showMessage(tr("common.success"))
            visual_count = sum(1 for char in chars if char.id.startswith("visual_char_"))
            msg = (
                f"Đã cập nhật {len(chars)} hồ sơ, gồm {visual_count} gợi ý nhân vật từ hình ảnh. "
                "Ứng dụng tự ghép thoại theo dữ liệu người nói và ngữ cảnh hiện có; các câu thiếu căn cứ "
                "vẫn được gán giọng mặc định ổn định để tạo lồng tiếng. Có thể chỉnh trực tiếp nếu cần.\n\n"
                + "\n".join(lines_info)
            )
            QMessageBox.information(self, tr("character.vision_title"), msg)

        self._run_worker(
            run_vision_task,
            on_success=on_vision_completed,
            on_progress=lambda p, msg: self.status_bar.showMessage(f"{p}% - {msg}"),
            on_failed=lambda err: QMessageBox.critical(self, tr("common.error"), str(err))
        )

    def _on_smart_assign_dialogue(self) -> None:
        if not self.project.segments:
            QMessageBox.warning(self, tr("common.error"), "Cần có danh sách phụ đề trước khi phân vai. Vui lòng bấm 'Trích xuất ASR' hoặc 'Quét OCR' trước.")
            return

        self.status_bar.showMessage("Đang phân tích kịch bản & phân vai nhân vật...")
        vis_model = self.container.app_settings.vision_model
        project_snapshot = self.project.model_copy(deep=True)

        def run_assign(progress_callback=None):
            return VisionCharacterService.assign_dialogue_smart(
                [s.model_copy(deep=True) for s in project_snapshot.segments],
                [c.model_copy(deep=True) for c in project_snapshot.characters],
                model_name=vis_model,
                progress_callback=progress_callback,
            )

        def on_assign_done(result):
            segments, characters = result
            self.project.segments = segments
            self.project.characters = characters
            self.character_panel.set_characters(self.project.characters)
            self.subtitle_panel.set_data(self.project.segments, self.project.characters)
            self.timeline_widget.set_data(self.project.segments, self.project.characters)
            self.status_bar.showMessage(tr("common.success"))

            char_summaries = []
            for c in characters[:6]:
                style_str = f" - '{c.custom_prompt}'" if c.custom_prompt else ""
                char_summaries.append(f"• {c.name}: Giọng {c.voice_id}{style_str}")
            if len(characters) > 6:
                char_summaries.append(f"• ... và {len(characters) - 6} nhân vật khác.")

            summary_text = "\n".join(char_summaries)
            QMessageBox.information(
                self,
                "Tự động phân vai (AI) thành công",
                f"AI đã tự động phân vai {len(segments)} câu thoại và cập nhật {len(characters)} hồ sơ. "
                "Câu chưa đủ căn cứ sẽ tự nhận preset TTS mặc định; có thể chỉnh trực tiếp trong bảng nếu cần.\n\n"
                f"Nhân vật và giọng đọc đã gán:\n{summary_text}\n\n"
                "Các giọng được sinh cục bộ từ mô hình VieNeu và thư viện mẫu đã tích hợp."
            )

        self._run_worker(
            run_assign,
            on_success=on_assign_done,
            on_progress=lambda p, msg: self.status_bar.showMessage(f"{p}% - {msg}")
        )

    def _analyze_speakers(
        self,
        video_path: str,
        audio_source_path: str,
        segments: List[SubtitleSegment],
        existing_characters: List[Character],
        target_speakers: int = 0,
        progress_callback=None,
    ):
        """Run speaker diarization, profile speakers, and link subtitle cues by time."""
        temp_dir = self.container.app_settings.temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        fd, audio_temp = tempfile.mkstemp(prefix="vt_diar_", suffix=".wav", dir=temp_dir)
        os.close(fd)
        try:
            if progress_callback:
                progress_callback(10, "Đang chuẩn hóa âm thanh để nhận diện người nói...")
            subprocess.run(
                [
                    self.container.app_settings.ffmpeg_path, "-y", "-i", audio_source_path,
                    "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_temp,
                ],
                capture_output=True,
                check=True,
            )
            diar_segments = self.diarization_service.diarize(
                audio_temp,
                video_path,
                target_speakers=target_speakers,
                force_refresh=False,
            )
            profiles = self.diarization_service.analyze_speaker_profiles(audio_temp, diar_segments)
            characters = CharacterResolver.build_speaker_characters(
                diar_segments, profiles=profiles, existing_characters=existing_characters
            )
            assigned_segments = CharacterResolver.match_segments_with_diarization(
                segments, diar_segments, characters
            )
            return diar_segments, profiles, characters, assigned_segments
        finally:
            if os.path.exists(audio_temp):
                try:
                    os.remove(audio_temp)
                except OSError:
                    pass

    def _on_auto_assign_voices(self, target_speakers: Optional[int] = None) -> None:
        if not self.project.video_path:
            QMessageBox.warning(self, tr("common.no_video"), tr("common.no_video_msg"))
            return

        if target_speakers is None:
            target_speakers, accepted = QInputDialog.getInt(
                self,
                "Nhận diện người nói",
                "Số người nói trong video (0 để tự ước lượng):",
                value=0,
                min=0,
                max=5,
            )
            if not accepted:
                return

        project_snapshot = self.project.model_copy(deep=True)
        self.status_bar.showMessage("Đang ước lượng người nói từ âm thanh và lập hồ sơ giọng...")

        def run_diar_task(progress_callback=None):
            source_input = (
                project_snapshot.isolated_vocals_path
                if project_snapshot.isolated_vocals_path and os.path.exists(project_snapshot.isolated_vocals_path)
                else project_snapshot.video_path
            )
            result = self._analyze_speakers(
                project_snapshot.video_path,
                source_input,
                [s.model_copy(deep=True) for s in project_snapshot.segments],
                project_snapshot.characters,
                target_speakers=target_speakers,
                progress_callback=progress_callback,
            )
            diar_segments, profiles, chars, assigned_segments = result
            if assigned_segments:
                assigned_segments, chars = VisionCharacterService.assign_dialogue_smart(
                    assigned_segments,
                    chars,
                    model_name=self.container.app_settings.vision_model,
                    progress_callback=progress_callback,
                )
            return diar_segments, profiles, chars, assigned_segments

        def on_diar_completed(result):
            diar_segments, profiles, chars, assigned_segments = result
            self.project.characters = chars
            if assigned_segments:
                self.project.segments = assigned_segments

            self.character_panel.set_characters(self.project.characters)
            self.subtitle_panel.set_data(self.project.segments, self.project.characters)
            self.timeline_widget.set_data(self.project.segments, self.project.characters)

            lines_info = []
            for c in chars:
                gender_str = (
                    tr("character.gender_male") if c.gender == SpeakerGender.MALE else
                    tr("character.gender_female") if c.gender == SpeakerGender.FEMALE else
                    "Chưa rõ"
                )
                lines_info.append(f"• {c.name} ({gender_str}) -> {c.voice_id}")

            self.status_bar.showMessage(tr("common.success"))
            unresolved = sum(1 for seg in self.project.segments if seg.character_id == "char_unknown")
            msg = (
                f"Đã tìm thấy {len(diar_segments)} đoạn người nói và lập {len(chars)} hồ sơ. "
                f"Có {unresolved} câu chưa đủ dữ liệu để nhận diện chắc chắn; TTS tự gán preset mặc định cho các câu này.\n\n"
                + "\n".join(lines_info)
                + "\n\nCó thể chỉnh trực tiếp nhân vật và giọng đọc trong bảng; không cần bước xác nhận trước khi tạo TTS."
            )
            QMessageBox.information(self, tr("character.diarization_title"), msg)

        self._run_worker(
            run_diar_task,
            on_success=on_diar_completed,
            on_progress=lambda p, msg: self.status_bar.showMessage(f"{p}% - {msg}"),
            on_failed=lambda err: QMessageBox.critical(self, tr("common.error"), str(err))
        )

    def _on_translate_all(self) -> None:
        if not self.project.segments:
            QMessageBox.warning(
                self,
                tr("subtitle.title"),
                "Chưa có phân đoạn phụ đề nào. Vui lòng trích xuất ASR hoặc OCR trước."
            )
            return

        self.status_bar.showMessage(tr("subtitle.translate_all"))
        project_snapshot = self.project.model_copy(deep=True)

        def on_trans_prog(p, msg):
            self.status_bar.showMessage(f"{p}% - {msg}")

        self._run_worker(
            self.translation_service.translate_segments,
            on_success=self._on_translate_completed,
            on_progress=on_trans_prog,
            on_failed=lambda err: QMessageBox.critical(self, tr("common.error"), str(err)),
            segments=project_snapshot.segments,
            characters=project_snapshot.characters,
            source_lang="auto",
            target_lang="vi",
            force_retranslate=True,
            verify_pass=True,
            model_name=self.container.app_settings.vision_model
        )

    def _on_translate_completed(self, segments) -> None:
        self.project.segments = segments
        self.subtitle_panel.set_data(self.project.segments, self.project.characters)
        self.timeline_widget.set_data(self.project.segments, self.project.characters)
        self.preview_widget.set_preview_segments(self.project.segments)
        self.status_bar.showMessage(f"Đã dịch thành công {len(segments)} câu!")
        QMessageBox.information(
            self,
            tr("common.success"),
            f"Đã nhận bản dịch cho {len(segments)} câu. Hãy rà lại nội dung và xưng hô trước khi tạo giọng.",
        )

    def _on_quick_render_clicked(self) -> None:
        """Hành động RENDER NHANH 1-Click: Tự động chạy toàn bộ pipeline dịch và lồng tiếng từ A đến Z."""
        if not self.project.video_path or not os.path.exists(self.project.video_path):
            QMessageBox.warning(self, "Chưa chọn video", "Vui lòng bấm 'Mở Video' để chọn video cần dịch và lồng tiếng trước.")
            self._on_import_video()
            if not self.project.video_path or not os.path.exists(self.project.video_path):
                return

        # Prepare destination path
        app_st = self.container.app_settings
        safe_fallback_dir = os.path.expanduser("~/Movies/vTranslator")
        os.makedirs(safe_fallback_dir, exist_ok=True)
        dir_name = app_st.output_dir or (os.path.dirname(self.project.video_path) if self.project.video_path else safe_fallback_dir)
        if not os.path.exists(dir_name) or not os.access(dir_name, os.W_OK) or dir_name.rstrip("/") == "/Users":
            dir_name = safe_fallback_dir

        base_name = os.path.splitext(os.path.basename(self.project.video_path))[0] if self.project.video_path else "video"
        out_path = os.path.join(dir_name, f"{base_name}_vi.mp4")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

        # Sync settings from AppSettings
        app_st = self.container.app_settings
        self.project.audio_settings.original_volume = app_st.original_volume
        self.project.audio_settings.dubbing_volume = app_st.dubbing_volume
        self.project.audio_settings.max_tempo_stretch = app_st.max_tempo_stretch

        self.project.subtitle_style.font_name = app_st.font_name
        self.project.subtitle_style.font_size = app_st.font_size
        self.project.subtitle_style.primary_color = app_st.primary_color
        self.project.subtitle_style.outline_color = app_st.outline_color
        self.project.subtitle_style.outline_width = app_st.outline_width
        self.project.subtitle_style.margin_v = app_st.margin_v

        # Copy the selected source-frame ROI; keep the user's blur toggle intact.
        roi = self.preview_widget.get_roi()
        if roi and roi.get("width", 0) > 40 and roi.get("height", 0) > 15:
            self.project.blur_settings.x = roi["x"]
            self.project.blur_settings.y = roi["y"]
            self.project.blur_settings.width = roi["width"]
            self.project.blur_settings.height = roi["height"]
        self.preview_widget.set_roi_settings(self.project.blur_settings)
        project_snapshot = self.project.model_copy(deep=True)
        roi_snapshot = dict(roi or {})

        # UI state
        self.btn_quick_render.setEnabled(False)
        self.btn_quick_render.setText("⏳ Đang xử lý tự động...")
        self.quick_progress_bar.setVisible(True)
        self.btn_cancel_quick.setVisible(True)
        self.quick_progress_bar.setValue(2)
        self.quick_status_label.setText("Khởi động tiến trình tự động...")

        def on_prog(percent: int, msg: str):
            self.quick_progress_bar.setValue(percent)
            self.quick_status_label.setText(f"{percent}% - {msg}")
            self.status_bar.showMessage(f"{percent}% - {msg}")

        def on_success(result):
            self.btn_quick_render.setEnabled(True)
            self.btn_quick_render.setText("Dịch & lồng tiếng tự động")
            self.quick_progress_bar.setValue(100)
            self.btn_cancel_quick.setVisible(False)
            self.quick_status_label.setText("Hoàn thành.")
            self.status_bar.showMessage("Hoàn thành xuất video!")

            self.project = result["project"]
            self.project.segments = result["segments"]
            self.project.characters = result["characters"]
            if self.project.isolated_vocals_path or self.project.isolated_background_path:
                self.blur_panel.set_separation_status(
                    self.project.isolated_vocals_path,
                    self.project.isolated_background_path,
                )
            if self.project.isolated_background_path:
                self.preview_widget.set_background_preview_audio(self.project.isolated_background_path)
            self.subtitle_panel.set_data(self.project.segments, self.project.characters)
            self.character_panel.set_characters(self.project.characters)
            self.timeline_widget.set_data(self.project.segments, self.project.characters)
            self.preview_widget.set_preview_segments(self.project.segments)

            if os.path.exists(result["master_wav"]):
                self.preview_widget.set_master_preview_audio(result["master_wav"])

            # Prompt success dialog
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Xuất video thành công!")
            msg_box.setText(
                f"<b>🎉 Đã dịch và lồng tiếng hoàn chỉnh!</b><br><br>"
                f"• Số câu thoại đã dịch & lồng tiếng: <b>{len(result['segments'])}</b><br>"
                f"• Số nhân vật nhận diện: <b>{len(result['characters'])}</b><br>"
                f"• Tệp video xuất: <code>{result['output_path']}</code>"
            )
            btn_play = msg_box.addButton("▶ Xem video ngay", QMessageBox.ActionRole)
            btn_folder = msg_box.addButton("📁 Mở thư mục", QMessageBox.ActionRole)
            btn_ok = msg_box.addButton("Đóng", QMessageBox.RejectRole)
            msg_box.setDefaultButton(btn_play)
            msg_box.exec()

            clicked = msg_box.clickedButton()
            if clicked == btn_play:
                QDesktopServices.openUrl(QUrl.fromLocalFile(result["output_path"]))
            elif clicked == btn_folder:
                folder = os.path.dirname(os.path.abspath(result["output_path"]))
                if sys.platform == "darwin":
                    subprocess.run(["open", folder])
                elif sys.platform == "win32":
                    subprocess.run(["explorer", folder])
                else:
                    subprocess.run(["xdg-open", folder])

        def on_failed(err):
            self.btn_quick_render.setEnabled(True)
            self.btn_quick_render.setText("Dịch & lồng tiếng tự động")
            self.quick_progress_bar.setVisible(False)
            self.btn_cancel_quick.setVisible(False)
            self.quick_status_label.setText(f"Lỗi: {err}")
            self.status_bar.showMessage(f"Lỗi: {err}")
            QMessageBox.critical(self, "Lỗi xử lý", f"Có lỗi xảy ra trong quá trình xử lý:\n{err}")

        def on_cancelled():
            self.btn_quick_render.setEnabled(True)
            self.btn_quick_render.setText("Dịch & lồng tiếng tự động")
            self.quick_progress_bar.setVisible(False)
            self.btn_cancel_quick.setEnabled(True)
            self.btn_cancel_quick.setVisible(False)
            self.quick_status_label.setText("Đã hủy tác vụ.")
            self.status_bar.showMessage("Đã hủy tác vụ.")

        self._run_worker(
            self._run_quick_studio,
            on_success=on_success,
            on_progress=on_prog,
            on_failed=on_failed,
            on_cancelled=on_cancelled,
            out_path=out_path,
            voice_choice="auto",
            project=project_snapshot,
            roi=roi_snapshot,
        )

    def _on_export_current_project(self) -> None:
        """Xuất video thành phẩm từ toàn bộ dữ liệu phụ đề, giọng đọc, và cài đặt người dùng đã làm xong thủ công."""
        if not self.project.video_path or not os.path.exists(self.project.video_path):
            QMessageBox.warning(self, "Chưa chọn video", "Vui lòng mở video trước khi xuất bản.")
            return

        if not self.project.segments:
            QMessageBox.warning(
                self,
                "Chưa có phụ đề",
                "Dự án chưa có phụ đề nào để xuất video. Vui lòng bấm '1. Nhận diện thoại' hoặc quét OCR trước."
            )
            return

        missing_dubs = [
            seg for seg in self.project.segments
            if seg.translated_text.strip()
            and (not seg.tts_audio_path or not os.path.exists(seg.tts_audio_path))
        ]
        if missing_dubs:
            answer = QMessageBox.question(
                self,
                "Thiếu giọng lồng tiếng",
                f"Có {len(missing_dubs)} câu đã dịch nhưng chưa có âm thanh TTS. "
                "Nếu tiếp tục, các câu này sẽ không được lồng tiếng. Bạn vẫn muốn xuất video?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        # Prepare default out_path
        app_st = self.container.app_settings
        safe_fallback_dir = os.path.expanduser("~/Movies/vTranslator")
        os.makedirs(safe_fallback_dir, exist_ok=True)
        dir_name = app_st.output_dir or (os.path.dirname(self.project.video_path) if self.project.video_path else safe_fallback_dir)
        if not os.path.exists(dir_name) or not os.access(dir_name, os.W_OK) or dir_name.rstrip("/") == "/Users":
            dir_name = safe_fallback_dir

        base_name = os.path.splitext(os.path.basename(self.project.video_path))[0] if self.project.video_path else "video"
        default_out = os.path.join(dir_name, f"{base_name}_vi.mp4")

        # Let user confirm or pick destination path
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Chọn nơi lưu video thành phẩm",
            default_out,
            "MP4 Video (*.mp4)"
        )
        if not file_path:
            return

        out_path = file_path
        target_dir = os.path.dirname(os.path.abspath(out_path))
        if not os.access(target_dir, os.W_OK) or target_dir.rstrip("/") == "/Users":
            # Thư mục người dùng chọn không có quyền ghi (ví dụ /Users trên macOS)
            safe_out = os.path.join(safe_fallback_dir, os.path.basename(out_path))
            QMessageBox.information(
                self,
                "Đổi thư mục lưu",
                f"Thư mục '{target_dir}' không có quyền ghi tập tin.\n"
                f"Video sẽ được tự động lưu tại thư mục an toàn:\n{safe_out}"
            )
            out_path = safe_out

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

        # Sync current settings
        self.project.audio_settings.original_volume = app_st.original_volume
        self.project.audio_settings.dubbing_volume = app_st.dubbing_volume
        self.project.audio_settings.max_tempo_stretch = app_st.max_tempo_stretch

        self.project.subtitle_style.font_name = app_st.font_name
        self.project.subtitle_style.font_size = app_st.font_size
        self.project.subtitle_style.primary_color = app_st.primary_color
        self.project.subtitle_style.outline_color = app_st.outline_color
        self.project.subtitle_style.outline_width = app_st.outline_width
        self.project.subtitle_style.margin_v = app_st.margin_v

        # Sync ROI for blur if defined in preview
        roi = self.preview_widget.get_roi()
        if roi and roi.get("width", 0) > 40 and roi.get("height", 0) > 15:
            self.project.blur_settings.x = roi["x"]
            self.project.blur_settings.y = roi["y"]
            self.project.blur_settings.width = roi["width"]
            self.project.blur_settings.height = roi["height"]
        self.preview_widget.set_roi_settings(self.project.blur_settings)
        project_snapshot = self.project.model_copy(deep=True)

        # UI state
        self.btn_export_video.setEnabled(False)
        self.btn_export_video.setText("⏳ Đang xuất video...")
        if hasattr(self.subtitle_panel, "btn_render"):
            self.subtitle_panel.btn_render.setEnabled(False)
        self.quick_progress_bar.setVisible(True)
        self.btn_cancel_quick.setVisible(True)
        self.quick_progress_bar.setValue(5)
        self.quick_status_label.setText("Đang chuẩn bị âm thanh & phụ đề...")

        def on_prog(percent: int, msg: str):
            self.quick_progress_bar.setValue(percent)
            self.quick_status_label.setText(f"{percent}% - {msg}")
            self.status_bar.showMessage(f"{percent}% - {msg}")

        def on_success(result):
            self.btn_export_video.setEnabled(True)
            self.btn_export_video.setText("Xuất video")
            if hasattr(self.subtitle_panel, "btn_render"):
                self.subtitle_panel.btn_render.setEnabled(True)
            self.quick_progress_bar.setValue(100)
            self.btn_cancel_quick.setVisible(False)
            self.quick_status_label.setText("Đã xuất video.")
            self.status_bar.showMessage("Hoàn thành xuất video!")

            self.project = result["project"]
            self.project.segments = result["segments"]
            vocals_path = self.project.isolated_vocals_path
            bgm_path = self.project.isolated_background_path
            if vocals_path and bgm_path:
                method = self.container.separator_service.get_separation_method(vocals_path)
                self.blur_panel.set_separation_status(vocals_path, bgm_path, method)
            self.subtitle_panel.set_data(self.project.segments, self.project.characters)
            self.timeline_widget.set_data(self.project.segments, self.project.characters)
            self.preview_widget.set_preview_segments(self.project.segments)

            if os.path.exists(result["master_wav"]):
                self.preview_widget.set_master_preview_audio(result["master_wav"])

            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Xuất video thành công!")
            msg_box.setText(
                f"<b>🎉 Đã xuất video hoàn chỉnh thành công!</b><br><br>"
                f"• Số câu phụ đề & lồng tiếng: <b>{len(result['segments'])}</b><br>"
                f"• Tệp video xuất: <code>{result['output_path']}</code>"
            )
            btn_play = msg_box.addButton("▶ Xem video ngay", QMessageBox.ActionRole)
            btn_folder = msg_box.addButton("📁 Mở thư mục", QMessageBox.ActionRole)
            btn_ok = msg_box.addButton("Đóng", QMessageBox.RejectRole)
            msg_box.setDefaultButton(btn_play)
            msg_box.exec()

            clicked = msg_box.clickedButton()
            if clicked == btn_play:
                QDesktopServices.openUrl(QUrl.fromLocalFile(result["output_path"]))
            elif clicked == btn_folder:
                import sys
                folder = os.path.dirname(os.path.abspath(result["output_path"]))
                if sys.platform == "darwin":
                    subprocess.run(["open", folder])
                elif sys.platform == "win32":
                    subprocess.run(["explorer", folder])
                else:
                    subprocess.run(["xdg-open", folder])

        def on_failed(err):
            self.btn_export_video.setEnabled(True)
            self.btn_export_video.setText("Xuất video")
            if hasattr(self.subtitle_panel, "btn_render"):
                self.subtitle_panel.btn_render.setEnabled(True)
            self.quick_progress_bar.setVisible(False)
            self.btn_cancel_quick.setVisible(False)
            self.quick_status_label.setText(f"Lỗi: {err}")
            self.status_bar.showMessage(f"Lỗi: {err}")
            QMessageBox.critical(self, "Lỗi xuất video", f"Có lỗi xảy ra trong quá trình xuất video:\n{err}")

        def on_cancelled():
            self.btn_export_video.setEnabled(True)
            self.btn_export_video.setText("Xuất video")
            if hasattr(self.subtitle_panel, "btn_render"):
                self.subtitle_panel.btn_render.setEnabled(True)
            self.quick_progress_bar.setVisible(False)
            self.btn_cancel_quick.setEnabled(True)
            self.btn_cancel_quick.setVisible(False)
            self.quick_status_label.setText("Đã hủy tác vụ.")
            self.status_bar.showMessage("Đã hủy tác vụ.")

        self._run_worker(
            self._run_export_manual_pipeline,
            on_success=on_success,
            on_progress=on_prog,
            on_failed=on_failed,
            on_cancelled=on_cancelled,
            out_path=out_path,
            project=project_snapshot,
        )

    def _run_export_manual_pipeline(
        self,
        out_path: str,
        progress_callback=None,
        project: Optional[Project] = None,
    ) -> dict:
        """Xuất video từ dữ liệu phụ đề và giọng đọc hiện tại của người dùng."""
        from ..audio.fit_to_slot import AudioTimelineScheduler
        project = project or self.project.model_copy(deep=True)

        # 1. Refuse to synthesize source text as target-language dubbing.
        untranslated = [s for s in project.segments if s.original_text.strip() and not s.translated_text.strip()]
        if untranslated and any(not s.tts_audio_path or not os.path.exists(s.tts_audio_path) for s in untranslated):
            raise ValueError("Còn câu chưa dịch. Hãy dịch hoặc nhập bản dịch trước khi tạo giọng lồng tiếng.")

        if project.segments:
            if progress_callback:
                progress_callback(10, "Đang kiểm tra giọng đọc và tạo các câu còn thiếu...")
            self.tts_engine.generate_segments_batch(
                project.segments,
                characters=project.characters,
                force_refresh=False,
                inter_segment_delay=2.0,
                progress_callback=lambda p, msg: progress_callback(int(10 + (p / 100) * 30), msg) if progress_callback else None
            )

        # 2. Căn khớp timeline âm thanh (Fit-to-slot)
        if progress_callback:
            progress_callback(45, "Đang căn khớp timeline âm thanh & chống đè tiếng...")

        scheduled_segments = AudioTimelineScheduler.schedule_segments(
            project.segments,
            max_tempo=project.audio_settings.max_tempo_stretch,
            min_gap_ms=80,
            ffmpeg_path=self.container.app_settings.ffmpeg_path,
        )

        meta = project.video_metadata
        total_dur = meta.duration_sec if meta else 60.0
        out_dir = os.path.dirname(os.path.abspath(out_path))
        safe_dub_dir = out_dir if (os.path.exists(out_dir) and os.access(out_dir, os.W_OK) and out_dir.rstrip("/") != "/Users") else self.container.app_settings.temp_dir
        os.makedirs(safe_dub_dir, exist_ok=True)
        master_wav = os.path.join(safe_dub_dir, os.path.splitext(os.path.basename(out_path))[0] + "_master_dub.wav")
        master_wav = AudioTimelineScheduler.build_master_dubbing_track(
            segments=scheduled_segments,
            total_duration_sec=total_dur,
            output_wav_path=master_wav,
            sample_rate=44100,
            volume=1.0
        )

        # 3. Render video hoàn chỉnh bằng FFmpeg
        if progress_callback:
            progress_callback(55, "Đang hòa âm, chèn phụ đề & xuất video hoàn chỉnh...")

        project.segments = scheduled_segments

        preset = self.container.app_settings.render_preset
        crf = self.container.app_settings.render_crf
        burn = self.container.app_settings.burn_subtitles

        render_res = self.render_pipeline.render(
            project=project,
            output_path=out_path,
            burn_subtitles=burn,
            preset=preset,
            crf=crf,
            progress_callback=lambda p, msg: progress_callback(int(55 + p * 0.44), f"Render: {msg}") if progress_callback else None
        )

        return {
            "output_path": render_res,
            "segments": scheduled_segments,
            "characters": project.characters,
            "master_wav": master_wav,
            "project": project,
        }

    def _on_ocr_subtitle_pipeline(self) -> None:
        """Pipeline OCR siêu nhanh: OCR → Dịch → Burn phụ đề (KHÔNG dùng TTS/Demucs/ASR).
        Chỉ tạo phụ đề; âm thanh nguồn được giữ nguyên."""
        if not self.project.video_path or not os.path.exists(self.project.video_path):
            QMessageBox.warning(self, "Chưa chọn video", "Vui lòng bấm 'Mở Video' trước.")
            self._on_import_video()
            if not self.project.video_path or not os.path.exists(self.project.video_path):
                return

        app_st = self.container.app_settings
        safe_fallback_dir = os.path.expanduser("~/Movies/vTranslator")
        os.makedirs(safe_fallback_dir, exist_ok=True)
        dir_name = app_st.output_dir or (os.path.dirname(self.project.video_path) if self.project.video_path else safe_fallback_dir)
        if not os.path.exists(dir_name) or not os.access(dir_name, os.W_OK) or dir_name.rstrip("/") == "/Users":
            dir_name = safe_fallback_dir

        base_name = os.path.splitext(os.path.basename(self.project.video_path))[0] if self.project.video_path else "video"
        out_path = os.path.join(dir_name, f"{base_name}_sub.mp4")

        # Sync subtitle style
        self.project.subtitle_style.font_name = app_st.font_name
        self.project.subtitle_style.font_size = app_st.font_size
        self.project.subtitle_style.primary_color = app_st.primary_color
        self.project.subtitle_style.outline_color = app_st.outline_color
        self.project.subtitle_style.outline_width = app_st.outline_width
        self.project.subtitle_style.margin_v = app_st.margin_v

        project_snapshot = self.project.model_copy(deep=True)
        roi_snapshot = self.preview_widget.get_roi()

        # UI state
        self.btn_ocr_subtitle.setEnabled(False)
        self.btn_ocr_subtitle.setText("⏳ Đang OCR + Dịch...")
        self.quick_progress_bar.setVisible(True)
        self.btn_cancel_quick.setVisible(True)
        self.quick_progress_bar.setValue(2)
        self.quick_status_label.setText("Khởi động OCR pipeline...")

        def on_prog(p: int, msg: str):
            self.quick_progress_bar.setValue(p)
            self.quick_status_label.setText(f"{p}% - {msg}")
            self.status_bar.showMessage(f"{p}% - {msg}")

        def on_success(result):
            self.btn_ocr_subtitle.setEnabled(True)
            self.btn_ocr_subtitle.setText("OCR · dịch · phụ đề")
            self.quick_progress_bar.setValue(100)
            self.btn_cancel_quick.setVisible(False)
            self.quick_status_label.setText("Hoàn thành.")
            self.status_bar.showMessage("Hoàn thành xuất video phụ đề!")

            self.project = result["project"]
            self.project.segments = result["segments"]
            self.subtitle_panel.set_data(self.project.segments, self.project.characters)
            self.timeline_widget.set_data(self.project.segments, self.project.characters)
            self.preview_widget.set_preview_segments(self.project.segments)

            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Xuất video phụ đề thành công!")
            msg_box.setText(
                f"<b>🎉 OCR + Dịch + Burn phụ đề hoàn chỉnh!</b><br><br>"
                f"• Số câu phụ đề: <b>{len(result['segments'])}</b><br>"
                f"• Âm thanh gốc: <b>Được giữ nguyên</b><br>"
                f"• Tệp video: <code>{result['output_path']}</code>"
            )
            btn_play = msg_box.addButton("▶ Xem ngay", QMessageBox.ActionRole)
            btn_folder = msg_box.addButton("📁 Mở thư mục", QMessageBox.ActionRole)
            msg_box.addButton("Đóng", QMessageBox.RejectRole)
            msg_box.setDefaultButton(btn_play)
            msg_box.exec()
            clicked = msg_box.clickedButton()
            if clicked == btn_play:
                from PySide6.QtCore import QUrl
                from PySide6.QtGui import QDesktopServices
                QDesktopServices.openUrl(QUrl.fromLocalFile(result["output_path"]))
            elif clicked == btn_folder:
                folder = os.path.dirname(os.path.abspath(result["output_path"]))
                if sys.platform == "darwin":
                    subprocess.run(["open", folder])
                elif sys.platform == "win32":
                    subprocess.run(["explorer", folder])
                else:
                    subprocess.run(["xdg-open", folder])

        def on_failed(err):
            self.btn_ocr_subtitle.setEnabled(True)
            self.btn_ocr_subtitle.setText("OCR · dịch · phụ đề")
            self.quick_progress_bar.setVisible(False)
            self.btn_cancel_quick.setVisible(False)
            self.quick_status_label.setText(f"Lỗi: {err}")
            QMessageBox.critical(self, "Lỗi", f"Pipeline OCR gặp lỗi:\n{err}")

        def on_cancelled():
            self.btn_ocr_subtitle.setEnabled(True)
            self.btn_ocr_subtitle.setText("OCR · dịch · phụ đề")
            self.quick_progress_bar.setVisible(False)
            self.btn_cancel_quick.setEnabled(True)
            self.btn_cancel_quick.setVisible(False)
            self.quick_status_label.setText("Đã hủy tác vụ.")
            self.status_bar.showMessage("Đã hủy tác vụ.")

        self._run_worker(
            self._run_ocr_subtitle_pipeline,
            on_success=on_success,
            on_progress=on_prog,
            on_failed=on_failed,
            on_cancelled=on_cancelled,
            out_path=out_path,
            project=project_snapshot,
            roi=roi_snapshot,
        )

    def _run_ocr_subtitle_pipeline(
        self,
        out_path: str,
        progress_callback=None,
        project: Optional[Project] = None,
        roi: Optional[dict] = None,
    ) -> dict:
        """Thực thi pipeline OCR siêu nhanh: OCR → Dịch → Burn phụ đề (subtitle_only=True)."""
        from ..subtitle.ocr_extractor import OcrSubtitleExtractor
        project = project or self.project.model_copy(deep=True)

        # Bước 1: Quét OCR phụ đề gốc
        if progress_callback:
            progress_callback(5, "Bước 1/3: Đang quét OCR phụ đề gốc từ video...")

        segments = OcrSubtitleExtractor.extract_subtitles_from_video(
            project.video_path,
            roi=roi,
            sample_interval=0.35,
            progress_callback=lambda p, msg: progress_callback(int(5 + p * 0.40), f"OCR: {msg}") if progress_callback else None
        )

        if not segments:
            raise RuntimeError("OCR không tìm thấy phụ đề nào trong video. Hãy kiểm tra lại vùng ROI (khung quét chữ).")

        if progress_callback:
            progress_callback(46, f"Bước 1/3 xong: Tìm thấy {len(segments)} câu phụ đề.")

        # Phân tích nhân vật nhanh từ kịch bản OCR để xưng hô chuẩn xác
        characters = project.characters
        try:
            if progress_callback:
                progress_callback(47, "Đang nhận diện nhân vật & bối cảnh xưng hô...")
            segments, characters = VisionCharacterService.assign_dialogue_smart(
                segments, project.characters, model_name=self.container.app_settings.vision_model
            )
            project.characters = characters
        except Exception as e:
            print(f"Lưu ý phân vai OCR: {e}")
            characters = project.characters

        # Bước 2: Dịch sang Tiếng Việt
        if progress_callback:
            progress_callback(50, "Bước 2/3: Đang dịch sang tiếng Việt...")

        translated_segments = self.translation_service.translate_segments(
            segments=segments,
            characters=characters,
            source_lang="auto",
            target_lang="vi",
            force_retranslate=True,
            verify_pass=True,
            model_name=self.container.app_settings.vision_model,
            progress_callback=lambda p, msg: progress_callback(int(50 + p * 0.28), f"Dịch: {msg}") if progress_callback else None
        )

        if progress_callback:
            progress_callback(78, f"Bước 2/3 xong: Đã dịch {len(translated_segments)} câu.")

        # Bước 3: Burn phụ đề vào video (subtitle_only=True - giữ nguyên audio)
        if progress_callback:
            progress_callback(79, "Bước 3/3: Đang burn phụ đề vào video (giữ nguyên audio gốc)...")

        project.segments = translated_segments

        # Sync subtitle style
        app_st = self.container.app_settings
        project.subtitle_style.font_name = app_st.font_name
        project.subtitle_style.font_size = app_st.font_size
        project.subtitle_style.primary_color = app_st.primary_color
        project.subtitle_style.outline_color = app_st.outline_color
        project.subtitle_style.outline_width = app_st.outline_width
        project.subtitle_style.margin_v = app_st.margin_v

        render_res = self.render_pipeline.render(
            project=project,
            output_path=out_path,
            burn_subtitles=True,
            preset=self.container.app_settings.render_preset,
            crf=self.container.app_settings.render_crf,
            subtitle_only=True,  # ← Không TTS, không mix audio
            progress_callback=lambda p, msg: progress_callback(int(79 + p * 0.20), f"Render: {msg}") if progress_callback else None
        )

        # Cache kết quả
        project.segments = translated_segments
        self.container.cache_service.save_stt(project.video_path, [s.model_dump() for s in translated_segments])

        return {
            "output_path": render_res,
            "segments": translated_segments,
            "project": project,
        }

    def _on_cancel_quick_render(self) -> None:
        """Request cancellation without killing a worker thread mid-write."""
        self.btn_cancel_quick.setEnabled(False)
        self.quick_status_label.setText("Đang dừng tác vụ an toàn…")
        self.render_pipeline.cancel()
        for worker in list(self._active_workers):
            worker.cancel()


    def _run_quick_studio(
        self,
        out_path: str,
        voice_choice: str = "auto",
        progress_callback=None,
        project: Optional[Project] = None,
        roi: Optional[dict] = None,
    ) -> dict:
        """Thực thi pipeline tự động hóa toàn diện từ đầu đến cuối với Tách âm thanh & Đối chiếu ASR+OCR."""
        from ..audio.fit_to_slot import AudioTimelineScheduler
        from ..subtitle.hybrid_extractor import HybridSubtitleExtractor
        project = project or self.project.model_copy(deep=True)

        # 1. Tách âm thanh thành Giọng nói sạch & Nhạc nền/SFX không lời
        if progress_callback:
            progress_callback(5, "Bước 1/6: Đang tách giọng nói và nhạc nền/hiệu ứng âm thanh...")

        try:
            vocals_path, bgm_path = self.container.separator_service.separate(
                project.video_path,
                force_refresh=False,
                progress_callback=lambda p, msg: progress_callback(int(5 + p * 0.12), msg) if progress_callback else None
            )
            project.isolated_vocals_path = vocals_path
            project.isolated_background_path = bgm_path
            print(f"✅ Tách âm thanh thành công: Vocals={vocals_path}, BGM={bgm_path}")
        except Exception as e:
            print(f"⚠️ Không thể tách âm thanh ({e}). Tiếp tục với âm thanh gốc...")
            vocals_path = None
            bgm_path = None

        # 2. Trích xuất phụ đề kép kết hợp ASR (từ giọng sạch) + OCR (từ khung hình)
        if progress_callback:
            progress_callback(18, "Bước 2/6: Trích xuất phụ đề kép kết hợp ASR (giọng sạch) + OCR (khung hình)...")

        segments = HybridSubtitleExtractor.extract_and_reconcile(
            video_path=project.video_path,
            stt_service=self.stt_service,
            vocals_wav_path=vocals_path,
            roi=roi,
            progress_callback=lambda p, msg: progress_callback(int(18 + p * 0.14), msg) if progress_callback else None
        )

        if not segments:
            raise RuntimeError("Không tìm thấy phụ đề nào trong video.")

        # 3. Phân tích người nói và gợi ý hồ sơ giọng VieNeu
        if progress_callback:
            progress_callback(34, "Bước 3/6: Đang phân tích người nói và gợi ý giọng đọc...")

        # Use measured speaker turns to keep a stable voice across scenes.
        if vocals_path and os.path.exists(vocals_path):
            diar_source = vocals_path
        else:
            diar_source = project.video_path
        try:
            diar_segments, profiles, characters, segments = self._analyze_speakers(
                project.video_path,
                diar_source,
                segments,
                project.characters,
                progress_callback=lambda p, msg: progress_callback(int(34 + p * 0.10), msg) if progress_callback else None,
            )
        except Exception as e:
            print(f"Không phân tích được speaker ({e}); giữ các câu ở trạng thái chưa xác định.")
            characters = []
            for seg in segments:
                seg.character_id = "char_unknown"
                seg.speaker_id = "SPEAKER_UNKNOWN"
                seg.speaker_source = "unassigned"
                seg.confidence = 0.0
                seg.voice_id = ""

        # Script analysis may suggest emotion and role labels. It cannot
        # replace a speaker identity measured from the audio timeline.
        if characters or not any(s.speaker_source == "diarization" for s in segments):
            segments, characters = VisionCharacterService.assign_dialogue_smart(
                segments,
                characters,
                model_name=self.container.app_settings.vision_model,
                progress_callback=lambda p, msg: progress_callback(int(44 + p * 0.06), msg) if progress_callback else None,
            )

        if voice_choice != "auto":
            for c in characters:
                c.update_voice(voice_choice)
            for s in segments:
                s.sync_voice_from_character(voice_choice)

        # 4. Dịch thuật điện ảnh
        if progress_callback:
            progress_callback(44, "Bước 4/6: Đang dịch sang tiếng Việt...")

        translated_segments = self.translation_service.translate_segments(
            segments=segments,
            characters=characters,
            source_lang="auto",
            target_lang="vi",
            force_retranslate=True,
            verify_pass=True,
            model_name=self.container.app_settings.vision_model,
            progress_callback=lambda p, msg: progress_callback(int(44 + p * 0.18), f"Bước 4/6: {msg}") if progress_callback else None
        )

        # 5. Sinh giọng đọc TTS & Đồng bộ timeline (Fit-to-slot)
        if progress_callback:
            progress_callback(64, f"Bước 5/6: Đang tổng hợp giọng cục bộ cho {len(translated_segments)} câu thoại...")

        self.tts_engine.generate_segments_batch(
            translated_segments,
            characters=characters,
            force_refresh=False,
            inter_segment_delay=0.0,
            progress_callback=lambda p, msg: progress_callback(int(64 + (p / 100) * 11), f"Bước 5/6: {msg}") if progress_callback else None
        )

        if progress_callback:
            progress_callback(75, "Bước 5/6: Đồng bộ timeline âm thanh (Fit-to-slot)...")

        scheduled_segments = AudioTimelineScheduler.schedule_segments(
            translated_segments,
            max_tempo=project.audio_settings.max_tempo_stretch,
            min_gap_ms=80,
            ffmpeg_path=self.container.app_settings.ffmpeg_path,
        )

        meta = project.video_metadata
        total_dur = meta.duration_sec if meta else 60.0
        out_dir = os.path.dirname(os.path.abspath(out_path))
        safe_dub_dir = out_dir if (os.path.exists(out_dir) and os.access(out_dir, os.W_OK) and out_dir.rstrip("/") != "/Users") else self.container.app_settings.temp_dir
        os.makedirs(safe_dub_dir, exist_ok=True)
        master_wav = os.path.join(safe_dub_dir, os.path.splitext(os.path.basename(out_path))[0] + "_master_dub.wav")
        master_wav = AudioTimelineScheduler.build_master_dubbing_track(
            segments=scheduled_segments,
            total_duration_sec=total_dur,
            output_wav_path=master_wav,
            sample_rate=44100,
            volume=1.0
        )

        # 6. Render Video Hoàn Chỉnh (Ghép lại BGM/SFX không lời + Giọng mới + Xóa chữ cũ + Logo)
        if progress_callback:
            progress_callback(78, "Bước 6/6: Hòa âm nhạc nền gốc, chèn phụ đề & xuất video hoàn chỉnh...")

        project.segments = scheduled_segments
        project.characters = characters

        preset = self.container.app_settings.render_preset
        crf = self.container.app_settings.render_crf
        burn = self.container.app_settings.burn_subtitles

        render_res = self.render_pipeline.render(
            project=project,
            output_path=out_path,
            burn_subtitles=burn,
            preset=preset,
            crf=crf,
            progress_callback=lambda p, msg: progress_callback(int(78 + p * 0.21), f"Bước 6/6: {msg}") if progress_callback else None
        )

        return {
            "output_path": render_res,
            "segments": scheduled_segments,
            "characters": characters,
            "master_wav": master_wav,
            "project": project,
        }

    def _on_generate_tts(self) -> None:
        if not self.project.segments:
            return

        self.status_bar.showMessage(tr("subtitle.generate_tts"))
        project_snapshot = self.project.model_copy(deep=True)

        def run_tts(progress_callback=None):
            from ..audio.fit_to_slot import AudioTimelineScheduler
            self.tts_engine.generate_segments_batch(
                project_snapshot.segments,
                characters=project_snapshot.characters,
                force_refresh=False,
                inter_segment_delay=0.0,
                progress_callback=lambda p, msg: progress_callback(int((p / 100) * 80), msg) if progress_callback else None
            )

            if progress_callback:
                progress_callback(85, "Syncing audio timeline...")

            AudioTimelineScheduler.schedule_segments(
                project_snapshot.segments,
                max_tempo=project_snapshot.audio_settings.max_tempo_stretch,
                min_gap_ms=80,
                ffmpeg_path=self.container.app_settings.ffmpeg_path,
            )

            meta = project_snapshot.video_metadata
            total_dur = meta.duration_sec if meta else 60.0
            master_wav = os.path.join(self.container.app_settings.temp_dir, "preview_master_dub.wav")
            AudioTimelineScheduler.build_master_dubbing_track(
                project_snapshot.segments,
                total_duration_sec=total_dur,
                output_wav_path=master_wav,
                sample_rate=44100,
                volume=1.0
            )
            return {"segments": project_snapshot.segments, "master_wav": master_wav, "project": project_snapshot}

        self._run_worker(
            run_tts,
            on_success=self._on_tts_completed,
            on_progress=lambda p, msg: self.status_bar.showMessage(f"{p}% - {msg}"),
            on_failed=lambda err: QMessageBox.critical(self, tr("common.error"), str(err))
        )

    def _on_tts_completed(self, result: dict) -> None:
        self.project = result.get("project", self.project)
        self.project.segments = result["segments"]
        self.subtitle_panel.set_data(self.project.segments, self.project.characters)
        self.timeline_widget.set_data(self.project.segments, self.project.characters)
        self.preview_widget.set_preview_segments(self.project.segments)
        if os.path.exists(result["master_wav"]):
            self.preview_widget.set_master_preview_audio(result["master_wav"])
        self.status_bar.showMessage(tr("common.success"))
