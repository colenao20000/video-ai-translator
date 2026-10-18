"""Full FFmpeg Video Render Pipeline with Master Dubbing Track Mixing.

Strictly adheres to RENDER_PIPELINE.md & VIDEO_TRANSFORM.md:
- Lanczos 2-even scaling: scale=w='trunc(iw/2)*2':h='trunc(ih/2)*2':flags=lanczos
- tpad frame clone padding when audio exceeds video duration: tpad=stop_mode=clone:stop_duration=...
- Video transform (Flip & Zoom) applied before subtitles to protect text integrity.
- Audio resampling and mixing at 44.1 kHz, stereo, float planar format.
- amix=inputs=2:duration=longest:normalize=0 (preserves volume without division degradation).
- Fast-start MP4 container: -movflags +faststart.
"""

import os
import re
import shutil
import subprocess
import tempfile
from typing import Callable, Optional
import soundfile as sf
from ..domain.project import Project
from ..audio.fit_to_slot import AudioTimelineScheduler
from ..subtitle.ass import AssGenerator
from ..video.transform import VideoTransformBuilder
from ..video.blur import BlurCoverBuilder
from ..domain.validators import ProjectIntegrityValidator


class RenderPipeline:
    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self._current_process: Optional[subprocess.Popen] = None

    def cancel(self) -> None:
        """Cancel active render process."""
        if self._current_process:
            try:
                self._current_process.kill()
            except Exception:
                pass
            self._current_process = None

    def render(
        self,
        project: Project,
        output_path: str,
        burn_subtitles: bool = True,
        preset: str = "fast",
        crf: int = 20,
        subtitle_only: bool = False,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> str:
        """Execute video render pipeline.
        
        Args:
            subtitle_only: Nếu True, bỏ qua TTS/audio – chỉ burn phụ đề vào video gốc.
                           Nhanh hơn rất nhiều, không cần audio dub đã sinh trước.
        """
        if not os.path.exists(project.video_path):
            raise FileNotFoundError(f"Không tìm thấy video gốc: {project.video_path}")

        val_res = ProjectIntegrityValidator.validate_project(project)
        if not val_res.is_valid:
            err_msgs = [e.message for e in val_res.errors]
            raise ValueError(f"Dự án không vượt qua kiểm tra toàn vẹn: {'; '.join(err_msgs)}")

        target_dir = os.path.dirname(os.path.abspath(output_path))
        if not os.access(target_dir, os.W_OK) or target_dir.rstrip("/") == "/Users":
            safe_dir = os.path.expanduser("~/Movies/vTranslator")
            os.makedirs(safe_dir, exist_ok=True)
            output_path = os.path.join(safe_dir, os.path.basename(output_path))
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix="vt_render_")

        try:
            total_duration_sec = project.video_metadata.duration_sec if project.video_metadata else 60.0
            video_w = project.video_metadata.width if project.video_metadata else 1920
            video_h = project.video_metadata.height if project.video_metadata else 1080
            video_w = (video_w // 2) * 2
            video_h = (video_h // 2) * 2

            # 1. Timeline scheduling & Master Dubbing Track (bỏ qua nếu subtitle_only)
            if progress_callback:
                progress_callback(5, "Đang kiểm tra âm thanh thuyết minh..." if not subtitle_only else "Chế độ Subtitle-Only: bỏ qua TTS...")

            has_tts = (not subtitle_only) and any(s.tts_audio_path and os.path.exists(s.tts_audio_path) for s in project.segments)
            master_dub_wav = None
            master_dub_dur = 0.0

            if has_tts:
                project.segments = AudioTimelineScheduler.schedule_segments(
                    project.segments,
                    max_tempo=project.audio_settings.max_tempo_stretch,
                    min_gap_ms=100,
                    ffmpeg_path=self.ffmpeg_path,
                )

                master_dub_wav = os.path.join(temp_dir, "dubbing_master.wav")
                AudioTimelineScheduler.build_master_dubbing_track(
                    segments=project.segments,
                    total_duration_sec=total_duration_sec,
                    output_wav_path=master_dub_wav,
                    sample_rate=44100,
                    volume=1.0
                )
                try:
                    info_dub = sf.info(master_dub_wav)
                    master_dub_dur = info_dub.duration
                except Exception:
                    master_dub_dur = total_duration_sec

            # 2. ASS Subtitle generation
            ass_path = None
            if burn_subtitles and project.segments:
                if progress_callback:
                    progress_callback(15, "Đang tạo tệp phụ đề ASS chuẩn hóa...")
                ass_path = os.path.join(temp_dir, "subtitles.ass")
                AssGenerator.generate(project.segments, ass_path, project.subtitle_style, video_w, video_h)

            # 3. Build FFmpeg Filter Graph
            if progress_callback:
                progress_callback(25, "Đang thiết lập chuỗi Filter Graph FFmpeg...")

            inputs = ["-i", project.video_path]
            filter_complex_parts = []

            orig_vol = project.audio_settings.original_volume
            dub_vol = project.audio_settings.dubbing_volume
            bg_vol = getattr(project.audio_settings, "bgm_volume", 0.20)

            has_bgm = (
                getattr(project, "isolated_background_path", None)
                and os.path.exists(project.isolated_background_path)
            )
            has_source_audio = not project.video_metadata or project.video_metadata.audio_codec != "none"

            if subtitle_only:
                # Optional mapping handles silent source videos too.
                audio_map_out = "0:a?" if has_source_audio else None
                _audio_is_direct_stream = True
            else:
                _audio_is_direct_stream = False
                if has_bgm and master_dub_wav and os.path.exists(master_dub_wav):
                    inputs.extend(["-i", master_dub_wav])
                    inputs.extend(["-i", project.isolated_background_path])
                    filter_complex_parts.append(
                        f"[1:a]volume={dub_vol:.2f},aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo[a_dub];"
                        f"[2:a]volume={bg_vol:.2f},aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo[a_bg];"
                        f"[a_bg][a_dub]amix=inputs=2:duration=longest:normalize=0[a_final];"
                    )
                    audio_map_out = "[a_final]"
                elif master_dub_wav and os.path.exists(master_dub_wav) and has_source_audio:
                    inputs.extend(["-i", master_dub_wav])
                    filter_complex_parts.append(
                        f"[0:a]volume={orig_vol:.2f},aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo[a_orig];"
                        f"[1:a]volume={dub_vol:.2f},aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo[a_dub];"
                        f"[a_orig][a_dub]amix=inputs=2:duration=longest:normalize=0[a_final];"
                    )
                    audio_map_out = "[a_final]"
                elif master_dub_wav and os.path.exists(master_dub_wav):
                    inputs.extend(["-i", master_dub_wav])
                    filter_complex_parts.append(
                        f"[1:a]volume={dub_vol:.2f},aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo[a_final];"
                    )
                    audio_map_out = "[a_final]"
                elif has_bgm and not has_source_audio:
                    inputs.extend(["-i", project.isolated_background_path])
                    filter_complex_parts.append(
                        f"[1:a]volume={bg_vol:.2f},aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo[a_final];"
                    )
                    audio_map_out = "[a_final]"
                elif has_source_audio:
                    filter_complex_parts.append(
                        f"[0:a]volume={orig_vol:.2f},aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo[a_final];"
                    )
                    audio_map_out = "[a_final]"
                else:
                    audio_map_out = None

            # 3.2 Video Filter Graph
            filter_complex_parts.append(
                "[0:v]scale=w='trunc(iw/2)*2':h='trunc(ih/2)*2':flags=lanczos[v_base];"
            )
            curr_v = "[v_base]"
            v_idx = 0

            audio_overshoot = max(0.0, master_dub_dur - total_duration_sec)
            if audio_overshoot > 0.05:
                next_v = f"[v_pad_{v_idx}]"
                filter_complex_parts.append(
                    f"{curr_v}tpad=stop_mode=clone:stop_duration={audio_overshoot:.3f}{next_v};"
                )
                curr_v = next_v
                v_idx += 1

            # Blur coordinates are selected on the original frame, so apply the
            # cover before zoom/flip changes that frame's coordinate system.
            if project.blur_settings.enabled and project.blur_settings.width > 0 and project.blur_settings.height > 0:
                next_v = f"[v_blur_{v_idx}]"
                blur_stage = BlurCoverBuilder.build_filter(
                    project.blur_settings, in_label=curr_v, out_label=next_v, max_w=video_w, max_h=video_h
                )
                if blur_stage:
                    filter_complex_parts.append(blur_stage)
                    curr_v = next_v
                    v_idx += 1

            tf_f = VideoTransformBuilder.build_filter_string(
                project.transform_settings, target_width=video_w, target_height=video_h
            )
            if tf_f:
                next_v = f"[v_transform_{v_idx}]"
                filter_complex_parts.append(f"{curr_v}{tf_f}{next_v};")
                curr_v = next_v
                v_idx += 1

            # Logo Watermark overlay (applied before subtitles so subtitles are always on top)
            has_logo = (
                getattr(project, "logo_settings", None)
                and project.logo_settings.enabled
                and project.logo_settings.path
                and os.path.exists(project.logo_settings.path)
            )
            if has_logo:
                logo_in_idx = len(inputs) // 2
                inputs.extend(["-i", project.logo_settings.path])
                logo_scale = project.logo_settings.scale_percent / 100.0
                target_logo_w = max(40, int(video_w * logo_scale))
                target_logo_w = (target_logo_w // 2) * 2
                op = max(0.1, min(1.0, project.logo_settings.opacity))
                mx = project.logo_settings.margin_x
                my = project.logo_settings.margin_y

                pos = project.logo_settings.position.lower()
                if "top_left" in pos:
                    overlay_xy = f"{mx}:{my}"
                elif "bottom_left" in pos:
                    overlay_xy = f"{mx}:main_h-overlay_h-{my}"
                elif "bottom_right" in pos:
                    overlay_xy = f"main_w-overlay_w-{mx}:main_h-overlay_h-{my}"
                else:
                    overlay_xy = f"main_w-overlay_w-{mx}:{my}"

                next_v = f"[v_logo_{v_idx}]"
                filter_complex_parts.append(
                    f"[{logo_in_idx}:v]scale={target_logo_w}:-1,format=rgba,colorchannelmixer=aa={op:.2f}[logo_scaled];"
                    f"{curr_v}[logo_scaled]overlay={overlay_xy}:format=auto{next_v};"
                )
                curr_v = next_v
                v_idx += 1

            # Burn ASS subtitles
            if ass_path:
                escaped_ass = ass_path.replace("\\", "/").replace(":", "\\:")
                next_v = f"[v_sub_{v_idx}]"
                filter_complex_parts.append(f"{curr_v}subtitles='{escaped_ass}'{next_v};")
                curr_v = next_v
                v_idx += 1

            # Final YUV420p format
            filter_complex_parts.append(f"{curr_v}format=yuv420p[v_final]")

            if subtitle_only and _audio_is_direct_stream:
                # Subtitle-Only: dùng -map 0:v, -map 0:a + audio copy để siêu nhanh
                full_cmd = [
                    self.ffmpeg_path, "-y", "-threads", "0"
                ] + inputs + [
                    "-filter_complex", "".join(filter_complex_parts),
                    "-map", "[v_final]",
                    "-c:v", "libx264",
                    "-preset", preset,
                    "-crf", str(crf),
                    "-fps_mode", "cfr",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "copy",   # Copy audio stream trực tiếp – không decode lại
                    "-movflags", "+faststart",
                    output_path
                ]
                if audio_map_out:
                    map_insert_at = full_cmd.index("-c:v")
                    full_cmd[map_insert_at:map_insert_at] = ["-map", audio_map_out]
            else:
                full_cmd = [
                    self.ffmpeg_path, "-y", "-threads", "0"
                ] + inputs + [
                    "-filter_complex", "".join(filter_complex_parts),
                    "-map", "[v_final]",
                    "-c:v", "libx264",
                    "-preset", preset,
                    "-crf", str(crf),
                    "-fps_mode", "cfr",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-ar", "44100",
                    "-movflags", "+faststart",
                    output_path
                ]
                if audio_map_out:
                    map_insert_at = full_cmd.index("-c:v")
                    full_cmd[map_insert_at:map_insert_at] = ["-map", audio_map_out]


            if progress_callback:
                progress_callback(35, "Bắt đầu tiến trình xuất bản video...")

            self._current_process = subprocess.Popen(
                full_cmd,
                stderr=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                universal_newlines=True
            )

            time_pattern = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")
            effective_target_sec = max(total_duration_sec, master_dub_dur)
            stderr_lines = []

            for line in self._current_process.stderr:
                stderr_lines.append(line)
                match = time_pattern.search(line)
                if match:
                    hours, mins, secs = match.groups()
                    current_sec = int(hours) * 3600 + int(mins) * 60 + float(secs)
                    percent = min(99, int(35 + (current_sec / max(1.0, effective_target_sec)) * 60))
                    if progress_callback:
                        progress_callback(percent, f"Đang render: {int(current_sec)}s / {int(effective_target_sec)}s ({percent}%)")

            self._current_process.wait()
            if self._current_process.returncode != 0:
                err_text = "".join(stderr_lines[-15:])
                raise RuntimeError(f"Tiến trình xuất video FFmpeg gặp lỗi: {err_text}")

            out_val = ProjectIntegrityValidator.validate_output_video(
                output_path, total_duration_sec, ffprobe_path=self.ffprobe_path
            )
            if not out_val.is_valid:
                err_msgs = [e.message for e in out_val.errors]
                raise RuntimeError(f"Video đầu ra không hợp lệ: {'; '.join(err_msgs)}")

            if progress_callback:
                progress_callback(100, f"✅ Xuất video hoàn tất thành công: {os.path.basename(output_path)}")

            return output_path

        finally:
            self._current_process = None
            shutil.rmtree(temp_dir, ignore_errors=True)
