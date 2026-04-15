#!/usr/bin/env python3
"""
Storyboard Pipeline for ltx-2-mlx
多镜头故事视频生成：JSON storyboard → sequential shot generation → ffmpeg merge

Usage:
    uv run python storyboard_pipeline.py storyboard_test.json --preview
    uv run python storyboard_pipeline.py storyboard_example.json
    uv run python storyboard_pipeline.py storyboard_example.json --ref-image cat.jpg
    uv run python storyboard_pipeline.py storyboard_example.json --reshoot 2 3
    uv run python storyboard_pipeline.py storyboard_example.json --assemble-only
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


class StoryboardPipeline:
    """多镜头故事视频生成 Pipeline"""

    def __init__(self, model_dir: str, low_memory: bool = True):
        self.model_dir = model_dir
        self.low_memory = low_memory

    def run(
        self,
        storyboard_path: str,
        preview: bool = False,
        ref_image: str | None = None,
        reshoot: list[int] | None = None,
        assemble_only: bool = False,
        bgm: str | None = None,
    ) -> str:
        """主入口：加载故事板 → 逐镜头生成 → 拼接 → 返回最终视频路径"""
        storyboard = self._load_storyboard(storyboard_path)
        settings = storyboard["settings"]
        output_dir = Path(settings.get("output_dir", "output"))
        shots_dir = output_dir / "shots"
        seeds_dir = output_dir / "seeds"
        shots_dir.mkdir(parents=True, exist_ok=True)
        seeds_dir.mkdir(parents=True, exist_ok=True)

        shots = storyboard["shots"]
        shot_paths = [shots_dir / f"shot_{s['id']:03d}.mp4" for s in shots]

        if not assemble_only:
            self._generate_all_shots(
                storyboard,
                shot_paths,
                seeds_dir,
                preview=preview,
                ref_image=ref_image,
                reshoot=reshoot,
            )

        # 过滤出已存在的镜头
        existing = [(s, p) for s, p in zip(shots, shot_paths) if p.exists()]
        if not existing:
            print("ERROR: 没有任何镜头文件存在，无法拼接")
            sys.exit(1)

        existing_paths = [str(p) for _, p in existing]
        final_path = output_dir / "final.mp4"

        self._assemble_video(
            existing_paths,
            str(final_path),
            crossfade=settings.get("crossfade_duration", 0.5),
        )

        if bgm:
            self._add_bgm(str(final_path), bgm)

        print(f"\n{'=' * 60}")
        print(f"完成！最终视频: {final_path}")
        print(f"镜头数: {len(existing)}/{len(shots)}")
        print(f"{'=' * 60}")
        return str(final_path)

    def _load_storyboard(self, path: str) -> dict:
        """加载并验证 JSON 故事板"""
        with open(path, "r", encoding="utf-8") as f:
            sb = json.load(f)

        # 验证必需字段
        assert "shots" in sb, "storyboard 缺少 'shots' 字段"
        assert len(sb["shots"]) > 0, "storyboard 的 'shots' 为空"
        if "settings" not in sb:
            sb["settings"] = {}

        # 填充默认值
        defaults = {
            "height": 480,
            "width": 704,
            "num_frames": 97,
            "seed_base": 42,
            "crossfade_duration": 0.5,
            "output_dir": "output",
            "model_dir": self.model_dir,
        }
        for k, v in defaults.items():
            sb["settings"].setdefault(k, v)

        return sb

    def _build_prompt(self, shot: dict, storyboard: dict) -> str:
        """替换模板变量 {character}, {style}"""
        prompt = shot["prompt"]
        char_desc = storyboard.get("character_description", "")
        style = storyboard.get("style_anchor", "")

        prompt = prompt.replace("{character}", char_desc)
        prompt = prompt.replace("{style}", style)

        # 如果有 style_anchor 且 prompt 里没用模板，追加到末尾
        if style and "{style}" not in shot["prompt"]:
            prompt = f"{prompt}, {style}"

        return prompt

    def _generate_all_shots(
        self,
        storyboard: dict,
        shot_paths: list[Path],
        seeds_dir: Path,
        preview: bool = False,
        ref_image: str | None = None,
        reshoot: list[int] | None = None,
    ):
        """逐镜头生成"""
        shots = storyboard["shots"]
        settings = storyboard["settings"]
        total = len(shots)

        for i, shot in enumerate(shots):
            shot_id = shot["id"]
            shot_path = shot_paths[i]

            # reshoot 模式：只重新生成指定镜头
            if reshoot and shot_id not in reshoot:
                if shot_path.exists():
                    print(f"[{i + 1}/{total}] 跳过 shot {shot_id}（已存在，不在 reshoot 列表中）")
                    continue

            # 决定参考图
            current_ref = None
            shot_ref = shot.get("ref_image")

            if i == 0 and ref_image:
                # 第一个镜头使用命令行 --ref-image
                current_ref = ref_image
            elif shot_ref == "auto" and i > 0:
                # 自动提取上一个镜头的最后一帧
                prev_path = shot_paths[i - 1]
                if prev_path.exists():
                    seed_path = seeds_dir / f"seed_{shot_id:03d}.png"
                    current_ref = self._extract_last_frame(str(prev_path), str(seed_path))
                else:
                    print(f"  WARN: 上一镜头 {prev_path} 不存在，fallback 到 T2V")
            elif shot_ref and shot_ref != "auto":
                # 手动指定参考图
                current_ref = shot_ref

            mode = "I2V" if current_ref else "T2V"
            print(f"\n[{i + 1}/{total}] 生成 shot {shot_id} ({mode})")
            if current_ref:
                print(f"  参考图: {current_ref}")

            t0 = time.time()
            try:
                self._generate_shot(
                    shot,
                    storyboard,
                    str(shot_path),
                    ref_image=current_ref,
                    preview=preview,
                )
                elapsed = time.time() - t0
                print(f"  完成! 耗时 {elapsed:.1f}s → {shot_path}")
            except Exception as e:
                elapsed = time.time() - t0
                print(f"  ERROR: shot {shot_id} 生成失败 ({elapsed:.1f}s): {e}")
                # 尝试降级重试
                if not preview:
                    print(f"  降级重试: num_frames 97 → 41")
                    try:
                        self._generate_shot(
                            shot,
                            storyboard,
                            str(shot_path),
                            ref_image=current_ref,
                            preview=True,  # 降级到预览参数
                        )
                        print(f"  降级成功! → {shot_path}")
                    except Exception as e2:
                        print(f"  降级也失败: {e2}")
                        print(f"  跳过 shot {shot_id}，后续可用 --reshoot {shot_id} 补拍")

    def _generate_shot(
        self,
        shot: dict,
        storyboard: dict,
        output_path: str,
        ref_image: str | None = None,
        preview: bool = False,
    ):
        """生成单个镜头"""
        settings = storyboard["settings"]
        h = 384 if preview else settings["height"]
        w = 576 if preview else settings["width"]
        nf = 41 if preview else shot.get("num_frames", settings["num_frames"])
        seed = settings["seed_base"] + shot["id"]
        prompt = self._build_prompt(shot, storyboard)

        print(f"  prompt: {prompt[:80]}...")
        print(f"  params: {w}x{h}, {nf} frames, seed={seed}")

        if ref_image is None:
            # T2V
            from ltx_pipelines_mlx import TextToVideoPipeline

            pipe = TextToVideoPipeline(
                model_dir=settings.get("model_dir", self.model_dir),
                low_memory=self.low_memory,
            )
            pipe.generate_and_save(
                prompt=prompt,
                output_path=output_path,
                height=h,
                width=w,
                num_frames=nf,
                seed=seed,
            )
        else:
            # I2V
            from ltx_pipelines_mlx import ImageToVideoPipeline

            pipe = ImageToVideoPipeline(
                model_dir=settings.get("model_dir", self.model_dir),
                low_memory=self.low_memory,
            )
            pipe.generate_and_save(
                prompt=prompt,
                output_path=output_path,
                image=ref_image,
                height=h,
                width=w,
                num_frames=nf,
                seed=seed,
            )

        # 强制释放模型内存
        del pipe
        try:
            from ltx_core_mlx.utils.memory import aggressive_cleanup

            aggressive_cleanup()
        except ImportError:
            pass  # cleanup 不是必须的

    def _extract_last_frame(self, video_path: str, output_path: str) -> str:
        """用 ffmpeg 提取视频最后一帧"""
        cmd = [
            "ffmpeg",
            "-y",
            "-sseof",
            "-0.1",
            "-i",
            video_path,
            "-vframes",
            "1",
            "-q:v",
            "2",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg 提取最后一帧失败: {result.stderr}")
        return output_path

    def _get_duration(self, video_path: str) -> float:
        """获取视频时长（秒）"""
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe 失败: {result.stderr}")
        return float(result.stdout.strip())

    def _assemble_video(
        self,
        shot_paths: list[str],
        output_path: str,
        crossfade: float = 0.5,
    ):
        """用 ffmpeg xfade 链式拼接所有镜头"""
        n = len(shot_paths)
        if n == 0:
            raise ValueError("没有镜头可拼接")

        if n == 1:
            # 只有一个镜头，直接复制
            subprocess.run(
                ["ffmpeg", "-y", "-i", shot_paths[0], "-c", "copy", output_path],
                check=True,
                capture_output=True,
            )
            return

        print(f"\n拼接 {n} 个镜头 (crossfade={crossfade}s)...")

        # 获取每个镜头的时长
        durations = []
        for p in shot_paths:
            try:
                d = self._get_duration(p)
                durations.append(d)
                print(f"  {Path(p).name}: {d:.2f}s")
            except Exception as e:
                print(f"  WARN: 获取 {p} 时长失败: {e}, 使用默认 4.0s")
                durations.append(4.0)

        try:
            self._xfade_concat(shot_paths, durations, output_path, crossfade)
            print(f"  xfade 拼接成功 → {output_path}")
        except Exception as e:
            print(f"  WARN: xfade 拼接失败 ({e})，fallback 到简单拼接")
            self._simple_concat(shot_paths, output_path)

    def _xfade_concat(
        self,
        paths: list[str],
        durations: list[float],
        output: str,
        crossfade: float,
    ):
        """ffmpeg xfade 链式拼接"""
        n = len(paths)
        inputs = []
        for p in paths:
            inputs += ["-i", p]

        # 构建视频 xfade filter chain
        v_filters = []
        prev = "[0:v]"
        offset = 0.0
        for i in range(1, n):
            offset += durations[i - 1] - crossfade
            if offset < 0:
                offset = 0  # 防止极短视频导致负 offset
            out = f"[xf{i}]" if i < n - 1 else "[vout]"
            v_filters.append(f"{prev}[{i}:v]xfade=transition=fade:duration={crossfade}:offset={offset:.3f}{out}")
            prev = f"[xf{i}]"

        # 构建音频 acrossfade filter chain
        a_filters = []
        a_prev = "[0:a]"
        for i in range(1, n):
            a_out = f"[af{i}]" if i < n - 1 else "[aout]"
            a_filters.append(f"{a_prev}[{i}:a]acrossfade=d={crossfade}:c1=tri:c2=tri{a_out}")
            a_prev = f"[af{i}]"

        # 先尝试带音频
        filter_complex = ";".join(v_filters + a_filters)
        cmd = (
            ["ffmpeg", "-y"]
            + inputs
            + ["-filter_complex", filter_complex]
            + ["-map", "[vout]", "-map", "[aout]"]
            + ["-c:v", "libx264", "-crf", "18", "-c:a", "aac", output]
        )

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # 音频 filter 可能失败（某些镜头无音频轨），fallback 到仅视频
            print("  音频 acrossfade 失败，尝试仅视频拼接...")
            filter_complex = ";".join(v_filters)
            cmd = (
                ["ffmpeg", "-y"]
                + inputs
                + ["-filter_complex", filter_complex]
                + ["-map", "[vout]", "-an"]
                + ["-c:v", "libx264", "-crf", "18", output]
            )
            subprocess.run(cmd, check=True, capture_output=True)

    def _simple_concat(self, paths: list[str], output: str):
        """简单拼接（无 crossfade），作为 fallback"""
        import tempfile

        list_file = tempfile.mktemp(suffix=".txt")
        with open(list_file, "w") as f:
            for p in paths:
                f.write(f"file '{os.path.abspath(p)}'\n")

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_file,
            "-c",
            "copy",
            output,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"  简单拼接成功 → {output}")
        finally:
            os.unlink(list_file)

    def _add_bgm(self, video_path: str, bgm_path: str):
        """添加背景音乐（替换原音频）"""
        tmp = video_path + ".tmp.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-i",
            bgm_path,
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            tmp,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            os.replace(tmp, video_path)
            print(f"  BGM 添加成功: {bgm_path}")
        else:
            print(f"  WARN: BGM 添加失败: {result.stderr}")
            if os.path.exists(tmp):
                os.unlink(tmp)


def main():
    parser = argparse.ArgumentParser(
        description="Storyboard Pipeline for ltx-2-mlx",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  uv run python storyboard_pipeline.py storyboard_test.json --preview
  uv run python storyboard_pipeline.py storyboard_example.json
  uv run python storyboard_pipeline.py storyboard_example.json --ref-image cat.jpg
  uv run python storyboard_pipeline.py storyboard_example.json --reshoot 2 3
  uv run python storyboard_pipeline.py storyboard_example.json --assemble-only
        """,
    )
    parser.add_argument("storyboard", help="故事板 JSON 文件路径")
    parser.add_argument("--preview", action="store_true", help="预览模式 (384x576, 41帧, 快速验证)")
    parser.add_argument("--ref-image", help="参考图路径（第一个镜头用 I2V）")
    parser.add_argument("--reshoot", type=int, nargs="+", help="重新生成指定镜头 ID")
    parser.add_argument("--assemble-only", action="store_true", help="仅重新拼接，不重新生成")
    parser.add_argument("--bgm", help="背景音乐路径（替换 AI 音频）")
    parser.add_argument(
        "--model",
        default="models/ltx-2.3-mlx-q8",
        help="模型目录路径 (default: models/ltx-2.3-mlx-q8)",
    )

    args = parser.parse_args()

    pipeline = StoryboardPipeline(model_dir=args.model)
    pipeline.run(
        storyboard_path=args.storyboard,
        preview=args.preview,
        ref_image=args.ref_image,
        reshoot=args.reshoot,
        assemble_only=args.assemble_only,
        bgm=args.bgm,
    )


if __name__ == "__main__":
    main()
