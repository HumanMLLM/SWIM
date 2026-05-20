from __future__ import annotations

import base64
import copy
import logging
import math
import os
import sys
import time
import warnings
from functools import lru_cache
from io import BytesIO
from typing import Optional

import requests
import torch
import torchvision
from packaging import version
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
import numpy as np
import decord

import builtins as __builtin__
import torch.distributed as dist

def print_rank0(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 3:
        __builtin__.print(*args, **kwargs)

logger = logging.getLogger(__name__)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 65536 * 28 * 28
MAX_RATIO = 1000

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768

# Set the maximum number of video token inputs.
# Here, 128K represents the maximum number of input tokens for the VLLM model.
# Remember to adjust it according to your own configuration.
VIDEO_TOTAL_PIXELS = int(float(os.environ.get('VIDEO_MAX_PIXELS', 128000 * 28 * 28 * 0.9)))
logger.info(f"set VIDEO_TOTAL_PIXELS: {VIDEO_TOTAL_PIXELS}")


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == 'RGBA':
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
        return white_background
    else:
        return pil_image.convert("RGB")


def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        # fix memory leak issue while using BytesIO
        with requests.get(image, stream=True) as response:
            response.raise_for_status()
            with BytesIO(response.content) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            # fix memory leak issue while using BytesIO
            with BytesIO(data) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = to_rgb(image_obj)
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))

    return image


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning(f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes


def _read_video_torchvision(
    ele: dict,
) -> (torch.Tensor, float):
    """read video using torchvision.io.read_video

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    video_path = ele["video"]
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if "http://" in video_path or "https://" in video_path:
            warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
        if "file://" in video_path:
            video_path = video_path[7:]
    st = time.time()
    video, audio, info = io.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end", None),
        pts_unit="sec",
        output_format="TCHW",
    )
    total_frames, video_fps = video.size(0), info["video_fps"]
    logger.info(f"torchvision:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video = video[idx]
    return video, sample_fps


def is_decord_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("decord") is not None


def calculate_video_frame_range(
    ele: dict,
    total_frames: int,
    video_fps: float,
) -> tuple[int, int, int]:
    """
    Calculate the start and end frame indices based on the given time range.

    Args:
        ele (dict): A dictionary containing optional 'video_start' and 'video_end' keys (in seconds).
        total_frames (int): Total number of frames in the video.
        video_fps (float): Frames per second of the video.

    Returns:
        tuple: A tuple containing (start_frame, end_frame, frame_count).

    Raises:
        ValueError: If input parameters are invalid or the time range is inconsistent.
    """
    # Validate essential parameters
    if video_fps <= 0:
        raise ValueError("video_fps must be a positive number")
    if total_frames <= 0:
        raise ValueError("total_frames must be a positive integer")

    # Get start and end time in seconds
    video_start = ele.get("video_start", None)
    video_end = ele.get("video_end", None)
    if video_start is None and video_end is None:
        return 0, total_frames - 1, total_frames

    max_duration = total_frames / video_fps
    # Process start frame
    if video_start is not None:
        video_start_clamped = max(0.0, min(video_start, max_duration))
        start_frame = math.ceil(video_start_clamped * video_fps)
    else:
        start_frame = 0
    # Process end frame
    if video_end is not None:
        video_end_clamped = max(0.0, min(video_end, max_duration))
        end_frame = math.floor(video_end_clamped * video_fps)
        end_frame = min(end_frame, total_frames - 1)
    else:
        end_frame = total_frames - 1

    # Validate frame order
    if start_frame >= end_frame:
        raise ValueError(
            f"Invalid time range: Start frame {start_frame} (at {video_start_clamped if video_start is not None else 0}s) "
            f"exceeds end frame {end_frame} (at {video_end_clamped if video_end is not None else max_duration}s). "
            f"Video duration: {max_duration:.2f}s ({total_frames} frames @ {video_fps}fps)"
        )

    logger.info(f"calculate video frame range: {start_frame=}, {end_frame=}, {total_frames=} from {video_start=}, {video_end=}, {video_fps=:.3f}")
    return start_frame, end_frame, end_frame - start_frame + 1


def _read_video_decord(
    ele: dict,
) -> (torch.Tensor, float):
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import decord
    video_path = ele["video"]
    st = time.time()
    vr = decord.VideoReader(video_path)
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    # start_frame, end_frame, total_frames = calculate_video_frame_range(
    #     ele,
    #     total_frames,
    #     video_fps,
    # )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    return video, sample_fps

# def _read_video_decord(
#     ele: dict,
# ) -> (torch.Tensor, float):
#     """read video using decord.VideoReader

#     Args:
#         ele (dict): a dict contains the configuration of video.
#         support keys:
#             - video: the path of video. support "file://", "http://", "https://" and local path.
#             - video_start: the start time of video.
#             - video_end: the end time of video.
#     Returns:
#         torch.Tensor: the video tensor with shape (T, C, H, W).
#     """
#     import decord
#     video_path = ele["video"]
#     st = time.time()
#     vr = decord.VideoReader(video_path)
#     total_frames, video_fps = len(vr), vr.get_avg_fps()
#     start_frame, end_frame, total_frames = calculate_video_frame_range(
#         ele,
#         total_frames,
#         video_fps,
#     )
#     nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
#     idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
#     video = vr.get_batch(idx).asnumpy()
#     video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
#     logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
#     sample_fps = nframes / max(total_frames, 1e-6) * video_fps
#     return video, sample_fps


def is_torchcodec_available() -> bool:
    """Check if torchcodec is available and properly installed."""
    try:
        import importlib.util
        if importlib.util.find_spec("torchcodec") is None:
            return False
        from torchcodec.decoders import VideoDecoder
        return True
    except (ImportError, AttributeError, Exception):
        return False


def _read_video_torchcodec(
    ele: dict,
) -> (torch.Tensor, float):
    """read video using torchcodec.decoders.VideoDecoder

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    from torchcodec.decoders import VideoDecoder
    TORCHCODEC_NUM_THREADS = int(os.environ.get('TORCHCODEC_NUM_THREADS', 8))
    logger.info(f"set TORCHCODEC_NUM_THREADS: {TORCHCODEC_NUM_THREADS}")
    video_path = ele["video"]
    st = time.time()
    decoder = VideoDecoder(video_path, num_ffmpeg_threads=TORCHCODEC_NUM_THREADS)
    video_fps = decoder.metadata.average_fps
    total_frames = decoder.metadata.num_frames
    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele,
        total_frames,
        video_fps,
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video = decoder.get_frames_at(indices=idx).data
    logger.info(f"torchcodec:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    return video, sample_fps


VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
    "torchcodec": _read_video_torchcodec,
}

FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)


@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if FORCE_QWENVL_VIDEO_READER is not None:
        video_reader_backend = FORCE_QWENVL_VIDEO_READER
    elif is_torchcodec_available():
        video_reader_backend = "torchcodec"
    elif is_decord_available():
        video_reader_backend = "decord"
    else:
        video_reader_backend = "torchvision"
    print(f"qwen-vl-utils using {video_reader_backend} to read video.", file=sys.stderr)
    return video_reader_backend

def frame_sample(duration, video_frames=16):
    num_frames = 32
    seg_size = float(duration - 1) / num_frames

    raw_frame_ids = []
    for i in range(num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        raw_frame_ids.append((start + end) // 2)

    sampled_frame_ids = []
    seg_size = float(num_frames - 1) / video_frames
    for i in range(video_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        sampled_frame_ids.append(raw_frame_ids[(start + end) // 2])
    return sampled_frame_ids

def safe_video_reader(video_path, ctx=None, num_threads=1):
    """安全初始化 Decord VideoReader，失败时返回 None"""
    try:
        if ctx is None:
            ctx = decord.cpu(0)
        return decord.VideoReader(video_path, ctx=ctx, num_threads=num_threads)
    except Exception as e:
        print(f"[警告] 无法读取视频: {video_path}, 错误: {e}")
        return None
def fetch_video(
    ele: dict,
    image_factor: int = IMAGE_FACTOR,
    return_video_sample_fps: bool = False,
    max_frames: int | None = 32,   # 新增参数
    sample_strategy: str = "uniform",  # ["truncate", "uniform"]
    include_idx: list = None,
) -> torch.Tensor | list[Image.Image]:
    # print('ele', ele)
    # exit()
    if isinstance(ele["video"], str):
        # video_reader_backend = get_video_reader_backend()
        # print("video_reader_backend", video_reader_backend)
        # exit()        
        # try:
        #     video, sample_fps = VIDEO_READER_BACKENDS[video_reader_backend](ele)
        # except Exception as e:
        #     logger.warning(f"video_reader_backend {video_reader_backend} error, use torchvision as default, msg: {e}")
        #     video, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)

        # nframes, _, height, width = video.shape

        vreader = safe_video_reader(ele["video"], num_threads=1)
        if vreader is None:
            # 读取失败时返回一个空张量，避免中断训练
            video = torch.zeros((1, 3, 224, 224))  # 这里的尺寸可以按你模型需要设置
            sample_fps = 0.0
            return (video, sample_fps) if return_video_sample_fps else video
        nframes, sample_fps = len(vreader), vreader.get_avg_fps()
        # frame_id_list = frame_sample(nframes, video_frames=max_frames)
        

        # ---------------- 限制帧数 ----------------
        # print(max_frames, nframes, sample_strategy)
        if max_frames is not None:
            # print_rank0(f"[INFO] 帧数限制: {max_frames=}, {nframes=}, {sample_strategy=}, {include_idx=}")
            if include_idx is not None and len(include_idx) > 0:
                # 去重并裁剪到合法范围
                include_idx = sorted(set([i for i in include_idx if 0 <= i < nframes]))
                # print_rank0(f"[INFO] 帧数限制: {max_frames=}, {nframes=}, {sample_strategy=}, {include_idx=}")
                # exit()
                
                # 还需要采样的帧数
                remaining = max_frames - len(include_idx)
                if remaining <= 0:
                    uniform_idx = []
                    # raise ValueError(f"include_idx 太多({len(include_idx)})，超过了 max_frames({max_frames})")
                else:
                    uniform_idx = torch.linspace(0, nframes - 1, steps=remaining).long().tolist()
                
                # 合并并去重
                idx = sorted(set(include_idx + uniform_idx))
                # video = video[idx]
                video = vreader.get_batch(idx).asnumpy()
                mask = [frame if frame in include_idx else 0 for frame in idx]
                mask = mask[1::2]
                # print_rank0(mask)
            else:
                if sample_strategy == "truncate":
                    idx = list(range(max_frame))
                    # video = video[:max_frames]
                    video = vreader.get_batch(idx).asnumpy()
                elif sample_strategy == "uniform":
                    idx = torch.linspace(0, nframes - 1, steps=max_frames).long()
                    # video = video[idx]
                    video = vreader.get_batch(idx).asnumpy()
            nframes = video.shape[0]
        video = torch.tensor(video).permute(0, 3, 1, 2)
        nframes, _, height, width = video.shape
        # ----------- 分辨率控制逻辑 ------------
        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels_supposed = ele.get("max_pixels", max_pixels)
        if max_pixels_supposed > max_pixels:
            logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
        max_pixels = min(max_pixels_supposed, max_pixels)

        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()

        # print('video shape', video.shape)
        # print('return_video_sample_fps', return_video_sample_fps)
        # exit()
        if return_video_sample_fps:
            # print('include_idx', include_idx, "mask ", mask)
            # exit()
            # print("include_idx", include_idx)
            if include_idx is not None:
                return video, sample_fps, mask
            return video, sample_fps
        return video
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [
            fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
            for video_element in ele["video"]
        ]
        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        if return_video_sample_fps:
            return images, process_info.pop("fps", 2.0)
        return images

# def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR, return_video_sample_fps: bool = False) -> torch.Tensor | list[Image.Image]:
#     if isinstance(ele["video"], str):
#         video_reader_backend = get_video_reader_backend()
#         try:
#             video, sample_fps = VIDEO_READER_BACKENDS[video_reader_backend](ele)
#         except Exception as e:
#             logger.warning(f"video_reader_backend {video_reader_backend} error, use torchvision as default, msg: {e}")
#             video, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)

#         nframes, _, height, width = video.shape
#         min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
#         total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
#         max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
#         max_pixels_supposed = ele.get("max_pixels", max_pixels)
#         if max_pixels_supposed > max_pixels:
#             logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
#         max_pixels = min(max_pixels_supposed, max_pixels)
#         if "resized_height" in ele and "resized_width" in ele:
#             resized_height, resized_width = smart_resize(
#                 ele["resized_height"],
#                 ele["resized_width"],
#                 factor=image_factor,
#             )
#         else:
#             resized_height, resized_width = smart_resize(
#                 height,
#                 width,
#                 factor=image_factor,
#                 min_pixels=min_pixels,
#                 max_pixels=max_pixels,
#             )
#         video = transforms.functional.resize(
#             video,
#             [resized_height, resized_width],
#             interpolation=InterpolationMode.BICUBIC,
#             antialias=True,
#         ).float()
#         if return_video_sample_fps:
#             return video, sample_fps
#         return video
#     else:
#         assert isinstance(ele["video"], (list, tuple))
#         process_info = ele.copy()
#         process_info.pop("type", None)
#         process_info.pop("video", None)
#         images = [
#             fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
#             for video_element in ele["video"]
#         ]
#         nframes = ceil_by_factor(len(images), FRAME_FACTOR)
#         if len(images) < nframes:
#             images.extend([images[-1]] * (nframes - len(images)))
#         if return_video_sample_fps:
#             return images, process_info.pop("fps", 2.0)
#         return images

def extract_mask_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    mask_infos = []
    # print_rank0(conversations)
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        # mask_infos_per_conv = []
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "object_masks" in ele
                        or ele.get("type","") in ("object_masks")
                    ):
                        mask_infos.append(ele)
        # mask_infos.append(mask_infos_per_conv)
    return mask_infos

def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele.get("type","") in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: list[dict] | list[list[dict]],
    return_video_kwargs: bool = False,
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:

    vision_infos = extract_vision_info(conversations)
    # print(vision_infos)
    # exit()
    mask_infos = extract_mask_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    sample_index_inputs = []

    video_sample_fps_list = []
    # print(vision_infos)
    # exit()
    sample_frames = []
    # 这里无需拆分batch，但是需要拆分convs
    for mask_info_cur_conv in mask_infos:
        sample_frames_cur_video = []
        # print_rank0(mask_info)
        # exit()
        # 获取当前conv的全部mask
        cur_video_mask = mask_info_cur_conv['object_masks']
        # 获取当前视频的每一个instance
        for cur_instance in cur_video_mask:
            # 获取每一个instance的mask
            for cur_mask_frame in cur_instance:
                sample_frames.append(int(cur_mask_frame))
    sample_frames = sorted(set(sample_frames))
    cnt = 0
    # print_rank0("len(vision_infos): ",len(vision_infos))
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            if len(mask_infos) > 0:
                # print(sample_frames[cnt])
                video_input, video_sample_fps, sample_index = fetch_video(vision_info, return_video_sample_fps=True, include_idx=sample_frames)
                sample_index_inputs.extend(sample_index)
            else:
                video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
        cnt += 1
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, {'fps': video_sample_fps_list}
    if len(sample_index_inputs) > 0:
        # print_rank0("sample_index_inputs:", sample_index_inputs)
        return image_inputs, video_inputs, sample_index_inputs
    return image_inputs, video_inputs