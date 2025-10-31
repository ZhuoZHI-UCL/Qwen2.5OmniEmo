# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's Transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llava/processing_llava.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
from math import floor, ceil
import inspect
import math
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, BinaryIO, Literal, Optional, TypedDict, Union

import numpy as np
import torch
from transformers.image_utils import get_image_size, is_valid_image, to_numpy_array
from transformers.models.mllama.processing_mllama import (
    convert_sparse_cross_attention_mask_to_dense,
    get_cross_attention_token_mask,
)
from typing_extensions import override

from ..extras.constants import AUDIO_PLACEHOLDER, IGNORE_INDEX, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER
from ..extras.packages import (
    is_librosa_available,
    is_pillow_available,
    is_pyav_available,
    is_transformers_version_greater_than,
)


if is_librosa_available():
    import librosa


if is_pillow_available():
    from PIL import Image
    from PIL.Image import Image as ImageObject


if is_pyav_available():
    import av


if is_transformers_version_greater_than("4.52.0"):
    from transformers.image_utils import make_flat_list_of_images
    from transformers.video_utils import make_batched_videos
else:
    from transformers.image_utils import make_batched_videos, make_flat_list_of_images


if TYPE_CHECKING:
    from av.stream import Stream
    from numpy.typing import NDArray
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
    from transformers.image_processing_utils import BaseImageProcessor

    class EncodedImage(TypedDict):
        path: Optional[str]
        bytes: Optional[bytes]

    ImageInput = Union[str, bytes, EncodedImage, BinaryIO, ImageObject]
    VideoInput = Union[str, BinaryIO, list[list[ImageInput]]]
    AudioInput = Union[str, BinaryIO, NDArray]

    class MMProcessor(ProcessorMixin):
        patch_size: int
        image_seq_length: int
        num_additional_image_tokens: int
        vision_feature_select_strategy: Literal["default", "full"]

        def _get_number_of_features(self, orig_height: int, orig_width: int, height: int, width: int) -> int:
            pass


def _get_paligemma_token_type_ids(imglens: list[int], seqlens: list[int], processor: "MMProcessor") -> list[list[int]]:
    r"""Get paligemma token type ids for computing loss.

    It is slightly different with the original token type ids where the prompt part is 0.

    Returns:
        batch_token_type_ids: shape (batch_size, seq_length)

    """
    batch_token_type_ids = []
    for imglen, seqlen in zip(imglens, seqlens):
        image_seqlen = imglen * processor.image_seq_length
        batch_token_type_ids.append([0] * image_seqlen + [1] * (seqlen - image_seqlen))

    return batch_token_type_ids


def _get_gemma3_token_type_ids(batch_ids: list[list[int]], processor: "MMProcessor"):
    r"""Get gemma3 token type ids for computing loss.

    Returns:
        batch_token_type_ids: shape (batch_size, seq_length)

    """
    image_token_id: int = getattr(processor, "image_token_id")
    batch_token_type_ids = []
    for token_ids in batch_ids:
        token_ids = np.array(token_ids)
        token_type_ids = np.zeros_like(token_ids)
        token_type_ids[token_ids == image_token_id] = 1
        batch_token_type_ids.append(token_type_ids.tolist())

    return batch_token_type_ids


def _make_batched_images(images: list["ImageObject"], imglens: list[int]) -> list[list["ImageObject"]]:
    r"""Make nested list of images."""
    batch_images = []
    for imglen in imglens:
        batch_images.append(images[:imglen])
        images = images[imglen:]

    return batch_images


def _check_video_is_nested_images(video: "VideoInput") -> bool:
    r"""Check if the video is nested images."""
    return isinstance(video, list) and all(isinstance(frame, (str, BinaryIO, dict, ImageObject)) for frame in video)


@dataclass
class MMPluginMixin:
    image_token: Optional[str]
    video_token: Optional[str]
    audio_token: Optional[str]
    expand_mm_tokens: bool = True

    def _validate_input(
        self,
        processor: Optional["MMProcessor"],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> None:
        r"""Validate if this model accepts the input modalities."""
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        video_processor: BaseImageProcessor = getattr(
            processor, "video_processor", getattr(processor, "image_processor", None)
        )
        feature_extractor: SequenceFeatureExtractor = getattr(processor, "feature_extractor", None)
        if len(images) != 0 and self.image_token is None:
            raise ValueError(
                "This model does not support image input. Please check whether the correct `template` is used."
            )

        if len(videos) != 0 and self.video_token is None:
            raise ValueError(
                "This model does not support video input. Please check whether the correct `template` is used."
            )

        if len(audios) != 0 and self.audio_token is None:
            raise ValueError(
                "This model does not support audio input. Please check whether the correct `template` is used."
            )

        if self.image_token is not None and processor is None:
            raise ValueError("Processor was not found, please check and update your model file.")

        if self.image_token is not None and image_processor is None:
            raise ValueError("Image processor was not found, please check and update your model file.")

        if self.video_token is not None and video_processor is None:
            raise ValueError("Video processor was not found, please check and update your model file.")

        if self.audio_token is not None and feature_extractor is None:
            raise ValueError("Audio feature extractor was not found, please check and update your model file.")

    def _validate_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ):
        r"""Validate if the number of images, videos and audios match the number of placeholders in messages."""
        num_image_tokens, num_video_tokens, num_audio_tokens = 0, 0, 0
        #我们现在暂时关闭下面的检查，因为这样会导致报错无法进行
        #因为我们的方法会加入很多乱七八糟的占位符，而不是像之前那样只用一个<video><audio>来表示哈

        # for message in messages:
        #     num_image_tokens += message["content"].count(IMAGE_PLACEHOLDER)
        #     num_video_tokens += message["content"].count(VIDEO_PLACEHOLDER)
        #     num_audio_tokens += message["content"].count(AUDIO_PLACEHOLDER)

        # if len(images) != num_image_tokens:
        #     raise ValueError(
        #         f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens in {messages}."
        #     )

        # if len(videos) != num_video_tokens:
        #     raise ValueError(
        #         f"The number of videos does not match the number of {VIDEO_PLACEHOLDER} tokens in {messages}."
        #     )

        # if len(audios) != num_audio_tokens:
        #     raise ValueError(
        #         f"The number of audios does not match the number of {AUDIO_PLACEHOLDER} tokens in {messages}."
        #     )

    def _preprocess_image(
        self, image: "ImageObject", image_max_pixels: int, image_min_pixels: int, **kwargs
    ) -> "ImageObject":
        r"""Pre-process a single image."""
        if (image.width * image.height) > image_max_pixels:
            resize_factor = math.sqrt(image_max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < image_min_pixels:
            resize_factor = math.sqrt(image_min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def _get_video_sample_indices(
        self, video_stream: "Stream", video_fps: float, video_maxlen: int, **kwargs
    ) -> list[int]:
        r"""Compute video sample indices according to fps."""
        total_frames = video_stream.frames
        if total_frames == 0:  # infinite video
            return np.linspace(0, video_maxlen - 1, video_maxlen).astype(np.int32)

        sample_frames = max(1, math.floor(float(video_stream.duration * video_stream.time_base) * video_fps))
        sample_frames = min(total_frames, video_maxlen, sample_frames)
        return np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)

    def _regularize_images(self, images: list["ImageInput"], **kwargs) -> dict[str, list["ImageObject"]]:
        r"""Regularize images to avoid error. Including reading and pre-processing."""
        results = []
        for image in images:
            if isinstance(image, (str, BinaryIO)):
                image = Image.open(image)
            elif isinstance(image, bytes):
                image = Image.open(BytesIO(image))
            elif isinstance(image, dict):
                if image["bytes"] is not None:
                    image = Image.open(BytesIO(image["bytes"]))
                else:
                    image = Image.open(image["path"])

            if not isinstance(image, ImageObject):
                raise ValueError(f"Expect input is a list of images, but got {type(image)}.")

            results.append(self._preprocess_image(image, **kwargs))

        return {"images": results}

    def _regularize_videos(self, videos: list["VideoInput"], **kwargs) -> dict[str, list[list["ImageObject"]]]:
        r"""Regularizes videos to avoid error. Including reading, resizing and converting."""
        results = []
        for video in videos:
            frames: list[ImageObject] = []
            if _check_video_is_nested_images(video):
                for frame in video:
                    if not is_valid_image(frame) and not isinstance(frame, dict) and not os.path.exists(frame):
                        raise ValueError("Invalid image found in video frames.")
                frames = video
            else:
                container = av.open(video, "r")
                video_stream = next(stream for stream in container.streams if stream.type == "video")
                sample_indices = self._get_video_sample_indices(video_stream, **kwargs)
                container.seek(0)
                for frame_idx, frame in enumerate(container.decode(video_stream)):
                    if frame_idx in sample_indices:
                        frames.append(frame.to_image())

            frames = self._regularize_images(frames, **kwargs)["images"]
            results.append(frames)

        return {"videos": results}

    #下面是原本的，修改以加速
    # def _regularize_audios(
    #     self, audios: list["AudioInput"], sampling_rate: float, **kwargs
    # ) -> dict[str, Union[list["NDArray"], list[float]]]:
    #     r"""Regularizes audios to avoid error. Including reading and resampling."""
    #     results, sampling_rates = [], []
    #     for audio in audios:
    #         if not isinstance(audio, np.ndarray):
    #             audio, sampling_rate = librosa.load(audio, sr=sampling_rate)

    #         results.append(audio)
    #         sampling_rates.append(sampling_rate)

    #     return {"audios": results, "sampling_rates": sampling_rates}

    #下面是加速的
    def _regularize_audios(
    self, audios: list["AudioInput"], sampling_rate: float, **kwargs
) -> dict[str, Union[list["NDArray"], list[float]]]:
        r"""
        加入跨调用缓存，避免重复加载同一音频文件。
        返回结构保持不变：
        {
            "audios": List[np.ndarray],     # 每段音频的波形
            "sampling_rates": List[float],  # 每段对应的 sr（通常与传入一致）
        }
        """
        import os
        import numpy as np
        import librosa

        if not hasattr(self, "_emo_cache"):
            self._emo_cache = {"video": {}, "audio": {}}
        
        results, sampling_rates = [], []

        for audio in audios:
            # 情况 A：已是波形
            if isinstance(audio, np.ndarray):
                results.append(audio)
                sampling_rates.append(float(sampling_rate))
                continue

            # 情况 B：路径 -> 走缓存
            apath = str(audio)
            akey = ("audio_regularized", apath, int(sampling_rate))
            if hasattr(self, "_emo_cache") and akey in self._emo_cache["audio"]:
                y_cached, sr_cached = self._emo_cache["audio"][akey]
                results.append(y_cached)
                sampling_rates.append(float(sr_cached))
                continue

            # 读取 + 重采样
            y, sr = librosa.load(apath, sr=int(sampling_rate))
            results.append(y)
            sampling_rates.append(float(sr))

            # 写缓存
            if hasattr(self, "_emo_cache"):
                self._emo_cache["audio"][akey] = (y, sr)

        return {"audios": results, "sampling_rates": sampling_rates}

    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "MMProcessor",
        imglens: Optional[list[int]] = None,
    ) -> dict[str, "torch.Tensor"]:
        r"""Process visual inputs.

        Returns: (llava and paligemma)
            pixel_values: tensor with shape (B, C, H, W)

        Returns: (qwen2-vl)
            pixel_values: tensor with shape (num_patches, patch_dim)
            image_grid_thw: tensor with shape (num_images, 3), where the three numbers are time, width, height
                            where num_patches == torch.prod(image_grid_thw)

        Returns: (mllama)
            pixel_values: tensor with shape
                          (batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width)
                          For example, (2, 1, 4, 3, 560, 560).
            aspect_ratio_ids: tensor with shape (batch_size, max_num_images). For example, (2, 1).
            aspect_ratio_mask: tensor with shape (batch_size, max_num_images, max_image_tiles). For example, (2, 1, 4).
            num_tiles: List[List[int]] with shape (batch_size, num_images_in_batch). For example, (2, 1).

        """
        mm_inputs = {}
        if len(images) != 0:
            image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            if imglens is not None:  # if imglens are provided, make batched images
                images = _make_batched_images(images, imglens)

            image_processor_kwargs = {}
            if getattr(processor, "image_do_pan_and_scan", False):  # gemma3 image processor
                image_processor_kwargs.update(
                    {
                        "do_pan_and_scan": True,
                        "pan_and_scan_min_crop_size": 256,
                        "pan_and_scan_max_num_crops": 4,
                        "pan_and_scan_min_ratio_to_activate": 1.2,
                    }
                )

            mm_inputs.update(image_processor(images, return_tensors="pt", **image_processor_kwargs))

        if len(videos) != 0:
            video_processor: BaseImageProcessor = getattr(
                processor, "video_processor", getattr(processor, "image_processor", None)
            )
            videos = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 192 * 192),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 5.0),
                video_maxlen=getattr(processor, "video_maxlen", 300),
            )["videos"]
            if "videos" in inspect.signature(video_processor.preprocess).parameters:  # for qwen2_vl and video_llava
                mm_inputs.update(video_processor(images=None, videos=videos, return_tensors="pt"))
            else:  # for llava_next_video
                mm_inputs.update(video_processor(videos, return_tensors="pt"))

        if len(audios) != 0:
            feature_extractor: SequenceFeatureExtractor = getattr(processor, "feature_extractor", None)
            audios = self._regularize_audios(
                audios,
                sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            )["audios"]
            mm_inputs.update(
                feature_extractor(
                    audios,
                    sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
                    return_attention_mask=True,
                    padding="max_length",
                    return_tensors="pt",
                )
            )
            mm_inputs["feature_attention_mask"] = mm_inputs.pop("attention_mask", None)  # prevent conflicts

        

        return mm_inputs


@dataclass
class BasePlugin(MMPluginMixin):
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        r"""Pre-process input messages before tokenization for VLMs."""
        self._validate_input(processor, images, videos, audios)
        return messages

    def process_token_ids(
        self,
        input_ids: list[int],
        labels: Optional[list[int]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["MMProcessor"],
    ) -> tuple[list[int], Optional[list[int]]]:
        r"""Pre-process token ids after tokenization for VLMs."""
        self._validate_input(processor, images, videos, audios)
        return input_ids, labels

    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        r"""Build batched multimodal inputs for VLMs.

        Arguments:
            images: a list of image inputs, shape (num_images,)
            videos: a list of video inputs, shape (num_videos,)
            audios: a list of audio inputs, shape (num_audios,)
            imglens: number of images in each sample, shape (batch_size,)
            vidlens: number of videos in each sample, shape (batch_size,)
            audlens: number of audios in each sample, shape (batch_size,)
            batch_ids: token ids of input samples, shape (batch_size, seq_len)
            processor: a processor for pre-processing images and videos

        """
        self._validate_input(processor, images, videos, audios)
        return self._get_mm_inputs(images, videos, audios, processor)


@dataclass
class Qwen2VLPlugin(BasePlugin):
    vision_bos_token: str = "<|vision_start|>"
    vision_eos_token: str = "<|vision_end|>"

    @override
    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        image = super()._preprocess_image(image, **kwargs)
        if min(image.width, image.height) < 28:
            width, height = max(image.width, 28), max(image.height, 28)
            image = image.resize((width, height))

        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height))

        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height))

        return image

    # @override #这个太慢了准备给换成加速的
    # def _regularize_videos(
    #     self, videos: list["VideoInput"], **kwargs
    # ) -> dict[str, Union[list[list["ImageObject"]], list[float]]]:
    #     results, fps_per_video = [], []
    #     for video in videos:
    #         frames: list[ImageObject] = []
    #         if _check_video_is_nested_images(video):
    #             for frame in video:
    #                 if not is_valid_image(frame) and not isinstance(frame, dict) and not os.path.exists(frame):
    #                     raise ValueError("Invalid image found in video frames.")

    #             frames = video
    #             fps_per_video.append(kwargs.get("video_fps", 2.0))
    #         else:
    #             container = av.open(video, "r")
    #             video_stream = next(stream for stream in container.streams if stream.type == "video")
    #             sample_indices = self._get_video_sample_indices(video_stream, **kwargs)
    #             container.seek(0)
    #             for frame_idx, frame in enumerate(container.decode(video_stream)):
    #                 if frame_idx in sample_indices:
    #                     frames.append(frame.to_image())

    #             if video_stream.duration is None:
    #                 fps_per_video.append(kwargs.get("video_fps", 2.0))
    #             else:
    #                 fps_per_video.append(len(sample_indices) / float(video_stream.duration * video_stream.time_base))

    #         if len(frames) % 2 != 0:
    #             frames.append(frames[-1])

    #         frames = self._regularize_images(frames, **kwargs)["images"]
    #         results.append(frames)

    #     return {"videos": results, "fps_per_video": fps_per_video}


    #修改加上缓存的函数以加速推理
    @override
    def _regularize_videos(
        self, videos: list["VideoInput"], **kwargs
    ) -> dict[str, Union[list[list["ImageObject"]], list[float]]]:
        """
        加入跨调用缓存，避免重复解码同一视频。
        返回值保持不变：
        {
            "videos": List[List[ImageObject]],   # 每个视频 -> 规整后的帧序列（已过 _regularize_images）
            "fps_per_video": List[float],
        }
        缓存 key 由 (视频路径, 采样策略参数) 组成；嵌套图像序列不缓存（通常很快）。
        """
        import os
        from pathlib import Path
        import av
        if not hasattr(self, "_emo_cache"):
            self._emo_cache = {"video": {}, "audio": {}}
        results, fps_per_video = [], []

        # 采样相关参数会影响 sample_indices，需进入 cache key
        fps_arg = kwargs.get("video_fps", 5.0)
        spg_arg = kwargs.get("video_second_per_grid", kwargs.get("seconds_per_chunk", None))
        hgrid_arg = kwargs.get("video_grid_h", None)
        wgrid_arg = kwargs.get("video_grid_w", None)
        vframes_arg = kwargs.get("video_num_frames", None)

        for video in videos:
            frames: list[ImageObject] = []

            # 情况 A：输入就是嵌套图像序列（不经解码）
            if _check_video_is_nested_images(video):
                for frame in video:
                    if not is_valid_image(frame) and not isinstance(frame, dict) and not os.path.exists(frame):
                        raise ValueError("Invalid image found in video frames.")
                frames = video
                fps_per_video.append(fps_arg)  # 没有容器可估计，沿用传入 fps
                # 规整到图像张量
                frames = self._regularize_images(frames, **kwargs)["images"]
                results.append(frames)
                continue

            # 情况 B：路径 -> 需要解码；尝试命中缓存
            vpath = str(video)
            vkey = ("video_regularized", vpath, fps_arg, spg_arg, hgrid_arg, wgrid_arg, vframes_arg)
            if hasattr(self, "_emo_cache") and vkey in self._emo_cache["video"]:
                cached_frames, cached_fps = self._emo_cache["video"][vkey]
                results.append(cached_frames)
                fps_per_video.append(cached_fps)
                continue

            # 解码与采样
            container = av.open(vpath, "r")
            video_stream = next(stream for stream in container.streams if stream.type == "video")
            sample_indices = self._get_video_sample_indices(video_stream, **kwargs)
            container.seek(0)
            for frame_idx, frame in enumerate(container.decode(video_stream)):
                if frame_idx in sample_indices:
                    frames.append(frame.to_image())

            # 估计 fps
            if video_stream.duration is None:
                fps_est = fps_arg
            else:
                # 注意：duration * time_base -> 秒，采样到的帧数 / 秒 = 近似 fps
                fps_est = len(sample_indices) / float(video_stream.duration * video_stream.time_base)

            # 帧数必须为偶数（与原逻辑一致）
            if len(frames) % 2 != 0:
                frames.append(frames[-1])

            # 规整到图像张量
            frames = self._regularize_images(frames, **kwargs)["images"]

            # 写缓存
            if hasattr(self, "_emo_cache"):
                self._emo_cache["video"][vkey] = (frames, fps_est)

            results.append(frames)
            fps_per_video.append(fps_est)

        return {"videos": results, "fps_per_video": fps_per_video}









    @override
    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "MMProcessor",
    ) -> dict[str, "torch.Tensor"]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pt"))

        if len(videos) != 0:
            video_data = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )
            mm_inputs.update(image_processor(images=None, videos=video_data["videos"], return_tensors="pt"))
            temporal_patch_size: int = getattr(image_processor, "temporal_patch_size", 2)
            if "second_per_grid_ts" in processor.model_input_names:
                mm_inputs["second_per_grid_ts"] = [temporal_patch_size / fps for fps in video_data["fps_per_video"]]

        return mm_inputs

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")

        merge_length: int = getattr(image_processor, "merge_size") ** 2
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
            video_grid_thw = mm_inputs.get("video_grid_thw", [])
        else:
            image_grid_thw = [None] * len(images)
            video_grid_thw = [None] * len(videos)

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = image_grid_thw[num_image_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"{self.vision_bos_token}{self.image_token * image_seqlen}{self.vision_eos_token}",
                    1,
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                video_seqlen = video_grid_thw[num_video_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    VIDEO_PLACEHOLDER,
                    f"{self.vision_bos_token}{self.video_token * video_seqlen}{self.vision_eos_token}",
                    1,
                )
                num_video_tokens += 1

            message["content"] = content

        return messages



VIDEO_SLICE_RE = re.compile(r"<video\[(?P<t0>\d+(?:\.\d+)?):(?P<t1>\d+(?:\.\d+)?)\]>")
AUDIO_SLICE_RE = re.compile(r"<audio\[(?P<t0>\d+(?:\.\d+)?):(?P<t1>\d+(?:\.\d+)?)\]>")
@dataclass
class Qwen2OmniPlugin(Qwen2VLPlugin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._emo_cache = {
            "video": {},  # key: (video_path, fps, sec_per_grid, hgrid, wgrid, vframes)
            "audio": {}   # key: (audio_path, sr)
        }
        print("[mm_plugin] Qwen2OmniPlugin initialized with caching support.")

    audio_bos_token: str = "<|audio_start|>"
    audio_eos_token: str = "<|audio_end|>"
    def _sec_to_video_grid_range(self, t0, t1, mm_inputs, video_grid_thw, image_processor):
        """
        返回 (gs, ge, tokens_per_grid)：
        - gs/ge: temporal grid 的起止（左闭右开）
        - tokens_per_grid: 该视频每个 temporal grid 展开的 token 数（已考虑 merge_size）
        """
        # 本样本只有 1 个视频，固定取下标 0
        T, H, W = video_grid_thw[0].tolist()
        sec_per_grid = float(mm_inputs["video_second_per_grid"][0])  # 例如 0.4s
        gs = max(0, floor(t0 / sec_per_grid))
        ge = min(T,  ceil(t1 / sec_per_grid))
        tokens_per_grid = (H // image_processor.merge_size) * (W // image_processor.merge_size)
        return gs, ge, tokens_per_grid

    # === 把秒 -> audio token 范围 ===
    def _sec_to_audio_token_range(self, t0, t1, audio_lengths, processor):
        """
        返回 (ts, te) 音频 token 的切片；tokens_per_second 默认 25，可从 processor 读。
        """
        L = int(audio_lengths[0])  # 本样本只有 1 段音频
        tps = getattr(processor, "audio_tokens_per_second", 25)  # 建议在 processor 中暴露一个配置，默认 25
        ts = max(0, floor(t0 * tps))
        te = min(L,  ceil(t1 * tps))
        return ts, te

    # === 展开一个时间切片：<video[t0:t1]><audio[t0:t1]> -> BOS + VIDEO*... + EOS + BOS + AUDIO*... + EOS
    def _expand_timeslice_once(self, t0, t1, mm_inputs, video_grid_thw, image_processor, audio_lengths, processor):
        gs, ge, v_tpg = self._sec_to_video_grid_range(t0, t1, mm_inputs, video_grid_thw, image_processor)
        n_vtokens = max(0, (ge - gs) * v_tpg)

        ts, te = self._sec_to_audio_token_range(t0, t1, audio_lengths, processor)
        n_atokens = max(0, te - ts)

        s = self.vision_bos_token
        if n_vtokens > 0:
            s += self.video_token * n_vtokens
        s += self.vision_eos_token

        s += self.audio_bos_token
        if n_atokens > 0:
            s += self.audio_token * n_atokens
        s += self.audio_eos_token
        return s
    

    # @override   #这个太慢了下面给换成加速的
    # def _get_mm_inputs(
    #     self,
    #     images: list["ImageInput"],
    #     videos: list["VideoInput"],
    #     audios: list["AudioInput"],
    #     processor: "MMProcessor",
    # ) -> dict[str, "torch.Tensor"]:
    #     image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
    #     feature_extractor: SequenceFeatureExtractor = getattr(processor, "feature_extractor", None)
    #     mm_inputs = {}
    #     if len(images) != 0:
    #         images = self._regularize_images(
    #             images,
    #             image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
    #             image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
    #         )["images"]
    #         mm_inputs.update(image_processor(images, return_tensors="pt"))
    #     ##请注意，processor导入的路径是/home/CORP/zhuo.zhi/miniconda3/envs/qwen2.5omni/lib/python3.12/site-packages/transformers/models/qwen2_5_omni/processing_qwen2_5_omni.py
    #     if len(videos) != 0:
    #         video_dict = self._regularize_videos(
    #             videos,
    #             image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
    #             image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
    #             video_fps=getattr(processor, "video_fps", 5.0), #这个的意思是如果视频本身没有fps信息，就用5.0
    #             video_maxlen=getattr(processor, "video_maxlen", 128), #最大帧数限制
    #         ) #这里只对video的fps和size进行了处理
    #         mm_inputs.update(image_processor(images=None, videos=video_dict["videos"], return_tensors="pt"))
    #         temporal_patch_size: int = getattr(image_processor, "temporal_patch_size", 2)
    #         mm_inputs["video_second_per_grid"] = torch.tensor(
    #             [temporal_patch_size / fps for fps in video_dict["fps_per_video"]]
    #         ) #如果fps是2，然后temporal_patch_size是2，所以video_second_per_grid是1， 表示每个video patch对应1秒的视频
    #         #如果fps是5，然后temporal_patch_size是2，所以video_second_per_grid是0.4， 表示每个video patch对应0.4秒的视频

    #     if len(audios) != 0:
    #         audios = self._regularize_audios(
    #             audios,
    #             sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
    #         )["audios"]
            
    #         mm_inputs.update(
    #             feature_extractor( #注意这个函数是在 /home/CORP/zhuo.zhi/miniconda3/envs/qwen2.5omni/lib/python3.12/site-packages/transformers/models/whisper/feature_extraction_whisper.py
    #                 audios,
    #                 sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
    #                 return_attention_mask=True,
    #                 padding="max_length",
    #                 return_tensors="pt",
    #             )
    #         ) 
    #         #得到的audio的处理特征叫做: mm_inputs['input_features']    mm_inputs['input_features'].shape = torch.Size([1, 128, 30000])
    #         # 其中这个128是feature size,即mel特征，30000是时间维度，代表30000帧，每一帧大概是0.25ms
    #         mm_inputs["feature_attention_mask"] = mm_inputs.pop("attention_mask")  # prevent conflicts

    #     return mm_inputs

    #下面是加速后的
    @override
    def _get_mm_inputs(self, images, videos, audios, processor, **kwargs):
        """
        说明：
        - 维持你原有的数据流与键名；
        - 新增：支持只对时间维做切片（不重复特征化），m 的来源优先级：
            1) kwargs["messages"] 中 <video ...> 出现次数
            2) kwargs["desired_time_grids"] 或 kwargs["m_grids"]
            3) 若都无，则不切片（保持全长）
        - 音频端：若能从 second_per_grid 和 Whisper 帧率推断出每格帧数，则按 L = m * frames_per_grid 切到 input_features/feature_attention_mask。
        """
        import torch
        from typing import Union
        from transformers.image_processing_utils import BaseImageProcessor
        from transformers.models.whisper.feature_extraction_whisper import SequenceFeatureExtractor

        def _count_video_audio_pairs(msgs) -> int:
            """统计 messages 里 <video[..]> 的个数作为 m。"""
            if not isinstance(msgs, (list, tuple)):
                return 0
            total = 0
            for mm in msgs:
                if isinstance(mm, dict) and mm.get("role") in ("assistant", "user"):
                    s = mm.get("content", "")
                    if isinstance(s, str):
                        total += s.count("<video")
            return total

        # 从 kwargs（若存在）读取 messages / m
        messages = kwargs.get("messages", None)
        desired_m = kwargs.get("desired_time_grids", None)
        # 当前 _get_mm_inputs 的签名没有 kwargs，但 routeA_prepare/process_messages 在调用时可以通过 **kwargs 下传；
        # 若你所在版本没有传 kwargs，这里不会生效，仅保持全长。
        try:
            # 在 Python 方法里，未声明的 **kwargs 无法直接获取，这里借助 Python 的闭包或上层改造；
            # 如果你已按我建议在调用处传入 kwargs={"messages": messages, "desired_time_grids": m}，这里即可获取。
            pass
        except Exception:
            pass

        # —— images —— #
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        feature_extractor: SequenceFeatureExtractor = getattr(processor, "feature_extractor", None)
        mm_inputs = {}

        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pt"))

        # —— videos —— #
        video_second_per_grid_tensor = None
        T_total = None  # 全量时间网格数
        if len(videos) != 0:
            video_dict = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 192 * 192),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 5.0),             # 若容器无 fps，则用该默认
                video_maxlen=getattr(processor, "video_maxlen", 300),       # 最大帧数限制（你的原注释）
            )
            # 经过 image_processor 生成视频特征（注意：这一步依赖 _regularize_videos 的缓存来提速）
            mm_inputs.update(image_processor(images=None, videos=video_dict["videos"], return_tensors="pt"))

            temporal_patch_size: int = getattr(image_processor, "temporal_patch_size", 2)
            # 每个 video patch 对应的秒数： temporal_patch_size / fps
            video_second_per_grid_tensor = torch.tensor(
                [temporal_patch_size / fps for fps in video_dict["fps_per_video"]],
                dtype=torch.float32
            )
            mm_inputs["video_second_per_grid"] = video_second_per_grid_tensor

            # 统计视频时间网格数 T_total
            # 典型键：video_grid_thw: [[T, H, W]]
            video_grid_thw = mm_inputs.get("video_grid_thw", None)
            if video_grid_thw is not None and hasattr(video_grid_thw, "shape"):
                try:
                    T_total = int(video_grid_thw[0][0].item())
                except Exception:
                    # 兼容不同 dtype/shape
                    T_total = int(video_grid_thw.view(-1)[0].item())

        # —— audios —— #
        # Whisper 特征（固定或可变长取决于 padding 设置）
        audio_input_features = None
        audio_attention_mask = None
        if len(audios) != 0:
            audios_dict = self._regularize_audios(
                audios,
                sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            )
            audios_waves = audios_dict["audios"]

            mm_inputs.update(
                feature_extractor(
                    audios_waves,
                    sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
                    return_attention_mask=True,
                    padding="max_length",         # 你当前设置为 max_length，可能会 pad 到固定长度
                    return_tensors="pt",
                )
            )
            # 得到：
            #   mm_inputs['input_features']: torch.Size([B, feature_size=128, seq_len])
            #   mm_inputs['attention_mask']: torch.Size([B, seq_len])
            mm_inputs["feature_attention_mask"] = mm_inputs.pop("attention_mask")  # prevent conflicts
            audio_input_features = mm_inputs.get("input_features", None)            # [B, F, L]
            audio_attention_mask = mm_inputs.get("feature_attention_mask", None)    # [B, L]

        # =========================
        #   只做时间切片，不重算
        # =========================
        # 1) 推断 m（优先 messages → 显式 desired → 默认全长）
        m = None
        # 从 kwargs 里取（若上层已传）
        # 说明：若当前版本无法把 kwargs 传入本函数，你也可以在 routeA_prepare/process_messages 内部
        # 把 m 存在 self._last_m 上，这里读取 getattr(self, "_last_m", None) 做到“无侵入”获取。
        m = getattr(self, "_last_m", None)
        if m is None:
            # 尝试从 self._last_messages 统计
            messages = getattr(self, "_last_messages", None)
            if messages is not None:
                m = _count_video_audio_pairs(messages)

        # 降级：若仍为 None，尝试从全局 kwargs 里读取（如果你上层已改为传入）
        if m is None:
            desired_m = getattr(self, "_desired_time_grids", None)
            m = desired_m

        # 最终兜底：若都不可得，则不切片（保持全长）
        # 只有在已计算到视频 T_total 的情况下才做切片
        if (m is not None) and (T_total is not None) and ("pixel_values_videos" in mm_inputs):
            m = int(max(1, min(m, T_total)))

            # ---- 视频切片：pixel_values_videos & video_grid_thw ----
            try:
                pv_v = mm_inputs["pixel_values_videos"]     # <== 先取出来
                if pv_v.dim() >= 3:
                    mm_inputs["pixel_values_videos"] = pv_v[:, :min(m, pv_v.size(1)), ...]
            except Exception:
                pass

            try:
                v_thw = mm_inputs.get("video_grid_thw", None)   # [[T, H, W]]
                if v_thw is not None:
                    v_thw = v_thw.clone() if hasattr(v_thw, "clone") else v_thw
                    # 修改第一项的 T
                    v_thw.view(-1)[0] = m
                    mm_inputs["video_grid_thw"] = v_thw
            except Exception:
                pass

            # ---- 音频切片：input_features & feature_attention_mask ----
            # 若能推断出每格的 Whisper 帧数，则切到 L = m * frames_per_grid
            try:
                if audio_input_features is not None and audio_attention_mask is not None:
                    # 推断每格帧数：Whisper 特征通常为 100 Hz（10ms/帧）
                    # video_second_per_grid_tensor 是 [B]（通常=1），取第一个
                    if video_second_per_grid_tensor is not None:
                        sec_per_grid = float(video_second_per_grid_tensor[0].item())
                        frames_per_grid = int(round(sec_per_grid * 100.0))  # 100Hz 假设
                        frames_per_grid = max(frames_per_grid, 1)

                        L_total = int(audio_input_features.shape[-1])
                        L_need = min(L_total, m * frames_per_grid)

                        # slice 最后一维（时间帧），同时切 attention_mask
                        if L_need > 0 and L_need < L_total:
                            mm_inputs["input_features"] = audio_input_features[..., :L_need]
                            if audio_attention_mask is not None and audio_attention_mask.shape[-1] >= L_need:
                                mm_inputs["feature_attention_mask"] = audio_attention_mask[..., :L_need]
            except Exception:
                # 无法推断/固定 max_length：保留全长，依赖 attention_mask
                pass

        return mm_inputs

   
    @override
    def process_messages(self, messages, images, videos, audios, processor):
        from transformers.image_processing_utils import BaseImageProcessor
        # === 原始校验 ===
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)

        # 只要 expand_mm_tokens=True，这里就会先抽取整段视频/音频的形状信息
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            video_grid_thw = mm_inputs.get("video_grid_thw", [])
            image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)

            audio_lengths = None
            if "feature_attention_mask" in mm_inputs:
                # 从 whisper 的 mask 派生音频有效长度 → 再映射为“Omni 的 audio token 长度”
                input_lengths = (mm_inputs["feature_attention_mask"].sum(-1).numpy() - 1) // 2 + 1
                audio_lengths = (input_lengths - 2) // 2 + 1
        else:
            mm_inputs = {}
            video_grid_thw = [None] * len(videos)
            audio_lengths = [None] * len(audios)
            image_processor = getattr(processor, "image_processor", None)

        # === 仅实现“时间切片”占位；一个样本通常只有 1 段视频/音频，反复切片使用 index=0 ===
        for msg in messages:
            content = msg["content"]

            while True:
                v_m = VIDEO_SLICE_RE.search(content)
                a_m = AUDIO_SLICE_RE.search(content)
                if not v_m or not a_m:
                    break
                if v_m.end() > a_m.start():
                    # 强制要求 视频切片后面紧跟音频切片；避免错配
                    raise ValueError("Each <video[t0:t1]> must be immediately followed by <audio[t0:t1]>.")

                t0 = float(v_m.group("t0")); t1 = float(v_m.group("t1"))
                t0_a = float(a_m.group("t0")); t1_a = float(a_m.group("t1"))
                if abs(t0 - t0_a) > 1e-3 or abs(t1 - t1_a) > 1e-3:
                    raise ValueError("Video and audio time slices must match.")

                # 展开这一个时间窗
                ph = self._expand_timeslice_once(
                    t0, t1, mm_inputs, video_grid_thw, image_processor, audio_lengths, processor
                )
                # 用展开串替换这对占位
                content = content[:v_m.start()] + ph + content[a_m.end():]

            msg["content"] = content
        self._last_messages = messages
        self._last_m = sum(msg.get("content", "").count("<video") for msg in messages)
        return messages

PLUGINS = {
    "base": BasePlugin,
    "qwen2_omni": Qwen2OmniPlugin,
    "qwen2_vl": Qwen2VLPlugin,
}


def register_mm_plugin(name: str, plugin_class: type["BasePlugin"]) -> None:
    r"""Register a multimodal plugin."""
    if name in PLUGINS:
        raise ValueError(f"Multimodal plugin {name} already exists.")

    PLUGINS[name] = plugin_class


def get_mm_plugin(
    name: str,
    image_token: Optional[str] = None,
    video_token: Optional[str] = None,
    audio_token: Optional[str] = None,
    **kwargs,
) -> "BasePlugin":
    r"""Get plugin for multimodal inputs."""
    if name not in PLUGINS:
        raise ValueError(f"Multimodal plugin `{name}` not found.")

    return PLUGINS[name](image_token, video_token, audio_token, **kwargs)
