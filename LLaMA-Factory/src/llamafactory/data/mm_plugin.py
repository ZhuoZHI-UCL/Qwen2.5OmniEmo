
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

    def _regularize_audios(
        self, audios: list["AudioInput"], sampling_rate: float, **kwargs
    ) -> dict[str, Union[list["NDArray"], list[float]]]:
        r"""Regularizes audios to avoid error. Including reading and resampling."""
        results, sampling_rates = [], []
        for audio in audios:
            if not isinstance(audio, np.ndarray):
                audio, sampling_rate = librosa.load(audio, sr=sampling_rate)

            results.append(audio)
            sampling_rates.append(sampling_rate)

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
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )["videos"]
            if "videos" in inspect.signature(video_processor.preprocess).parameters:  # for qwen2_vl and video_llava
                # mm_inputs.update(video_processor(images=None, videos=videos, return_tensors="pt"))
                mm_inputs.update(video_processor(images=None, videos=videos))
            else:  # for llava_next_video
                # mm_inputs.update(video_processor(videos, return_tensors="pt"))
                mm_inputs.update(video_processor(videos))

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

    @override
    def _regularize_videos(
        self, videos: list["VideoInput"], **kwargs
    ) -> dict[str, Union[list[list["ImageObject"]], list[float]]]:
        results, fps_per_video = [], []
        for video in videos:
            frames: list[ImageObject] = []
            if _check_video_is_nested_images(video):
                for frame in video:
                    if not is_valid_image(frame) and not isinstance(frame, dict) and not os.path.exists(frame):
                        raise ValueError("Invalid image found in video frames.")

                frames = video
                fps_per_video.append(kwargs.get("video_fps", 2.0))
            else:
                container = av.open(video, "r")
                video_stream = next(stream for stream in container.streams if stream.type == "video")
                sample_indices = self._get_video_sample_indices(video_stream, **kwargs)
                container.seek(0)
                for frame_idx, frame in enumerate(container.decode(video_stream)):
                    if frame_idx in sample_indices:
                        frames.append(frame.to_image())

                if video_stream.duration is None:
                    fps_per_video.append(kwargs.get("video_fps", 2.0))
                else:
                    fps_per_video.append(len(sample_indices) / float(video_stream.duration * video_stream.time_base))

            if len(frames) % 2 != 0:
                frames.append(frames[-1])

            frames = self._regularize_images(frames, **kwargs)["images"]
            results.append(frames)

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
    audio_bos_token: str = "<|audio_start|>"
    audio_eos_token: str = "<|audio_end|>"

    def _sec_to_video_grid_range(self, t0, t1, mm_inputs, video_grid_thw, video_processor):
        T, H, W = video_grid_thw[0].tolist()
        sec_per_grid = float(mm_inputs["video_second_per_grid"][0])  # e.g., 0.4004

        # half-open [t0, t1)
        gs = int(max(0, math.floor(t0 / sec_per_grid)))
        # 注意：当 t1 恰好落在边界时，希望不包含下一格，所以用 (t1 - eps)
        eps = 1e-9
        ge = int(min(T, math.floor((t1 - eps) / sec_per_grid) + 1))

        merge = getattr(video_processor, "merge_size", 2)
        tokens_per_grid = (H // merge) * (W // merge)
        return gs, ge, tokens_per_grid

    # === 把秒 -> audio token 范围 ===
    def _sec_to_audio_token_range(self, t0, t1, audio_lengths, processor):
        L = int(audio_lengths[0])  # 这就是 get_audio_features 之后的“行数”
        tps = getattr(processor, "audio_tokens_per_second", 25)
        ts = int(max(0, math.floor(t0 * tps)))
        te = int(min(L, math.floor((t1 - 1e-9) * tps) + 1))
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
    @override
    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "MMProcessor",
    ) -> dict[str, "torch.Tensor"]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        video_processor: BaseImageProcessor = getattr(processor, "video_processor", None)  # ← 新增
        feature_extractor: SequenceFeatureExtractor = getattr(processor, "feature_extractor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            # mm_inputs.update(image_processor(images, return_tensors="pt"))
            mm_inputs.update(image_processor(images))

        ##请注意，processor导入的路径是/home/CORP/zhuo.zhi/miniconda3/envs/qwen2.5omni/lib/python3.12/site-packages/transformers/models/qwen2_5_omni/processing_qwen2_5_omni.py
        '''
        if len(videos) != 0:
            video_dict = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 5.0), #这个的意思是如果视频本身没有fps信息，就用5.0
                video_maxlen=getattr(processor, "video_maxlen", 128), #最大帧数限制
            ) #这里只对video的fps和size进行了处理
            # mm_inputs.update(image_processor(images=None, videos=video_dict["videos"], return_tensors="pt")) 
            mm_inputs.update(video_processor(videos=video_dict["videos"], return_tensors="pt"))
            # temporal_patch_size: int = getattr(image_processor, "temporal_patch_size", 2)
            temporal_patch_size: int = getattr(video_processor, "temporal_patch_size", 2)  # ← 用 video_processor
            mm_inputs["video_second_per_grid"] = torch.tensor(
                [temporal_patch_size / fps for fps in video_dict["fps_per_video"]]
            ) #如果fps是2，然后temporal_patch_size是2，所以video_second_per_grid是1， 表示每个video patch对应1秒的视频
            #如果fps是5，然后temporal_patch_size是2，所以video_second_per_grid是0.4， 表示每个video patch对应0.4秒的视频
'''
        if len(videos) != 0:
            try:
                video_dict = self._regularize_videos(
                    videos,
                    image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                    image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                    video_fps=getattr(processor, "video_fps", 5.0),
                    video_maxlen=getattr(processor, "video_maxlen", 128),
                )
                # out = video_processor(videos=video_dict["videos"], return_tensors="pt")
                out = video_processor(videos=video_dict["videos"])
            except Exception as e:
                print(f"[mm_plugin] Skip invalid video sample: {e}")
            else:
                mm_inputs.update(out)
                temporal_patch_size: int = getattr(video_processor, "temporal_patch_size", 2)
                mm_inputs["video_second_per_grid"] = torch.tensor(
                    [temporal_patch_size / fps for fps in video_dict["fps_per_video"]]
                )

        if len(audios) != 0:
            audios = self._regularize_audios(
                audios,
                sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            )["audios"]
            
            mm_inputs.update(
                feature_extractor( #注意这个函数是在 /home/CORP/zhuo.zhi/miniconda3/envs/qwen2.5omni/lib/python3.12/site-packages/transformers/models/whisper/feature_extraction_whisper.py
                    audios,
                    sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
                    return_attention_mask=True,
                    padding="max_length",
                    return_tensors="pt",
                )
            ) 
            #得到的audio的处理特征叫做: mm_inputs['input_features']    mm_inputs['input_features'].shape = torch.Size([1, 128, 30000])
            # 其中这个128是feature size,即mel特征，30000是时间维度，代表30000帧，每一帧大概是0.25ms
            mm_inputs["feature_attention_mask"] = mm_inputs.pop("attention_mask")  # prevent conflicts

        return mm_inputs

    @override
    def process_messages(self, messages, images, videos, audios, processor):
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)

        expand = self.expand_mm_tokens

        # === 1) 预取整段 mm 形状 ===
        if expand:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            video_grid_thw = mm_inputs.get("video_grid_thw", [])
            video_second_per_grid = mm_inputs.get("video_second_per_grid", None)
            video_processor = getattr(processor, "video_processor", None)

            # 从 whisper 的 mask 推导出“Omni 的音频 token 长度”（~25Hz）
            if "feature_attention_mask" in mm_inputs:
                input_lengths = (mm_inputs["feature_attention_mask"].sum(-1).numpy() - 1) // 2 + 1
                audio_lengths = (input_lengths - 2) // 2 + 1  # list-like
            else:
                audio_lengths = [None] * len(audios)
        else:
            mm_inputs = {}
            video_grid_thw = [None] * len(videos)
            video_second_per_grid = None
            audio_lengths = [None] * len(audios)
            video_processor = getattr(processor, "video_processor", None)

        # === 2) 统计本条样本里一共有多少对 <video[..]><audio[..]> ===
        import re
        VIDEO_SLICE_RE = re.compile(r"<video\[(?P<t0>\d+(?:\.\d+)?):(?P<t1>\d+(?:\.\d+)?)\]>")
        AUDIO_SLICE_RE = re.compile(r"<audio\[(?P<t0>\d+(?:\.\d+)?):(?P<t1>\d+(?:\.\d+)?)\]>")

        def count_pairs(text: str) -> int:
            n = 0; i = 0
            while True:
                v = VIDEO_SLICE_RE.search(text, i)
                a = AUDIO_SLICE_RE.search(text, i)
                if not v or not a:
                    break
                if v.end() != a.start():
                    # 强制要求紧邻，避免错配
                    raise ValueError("Each <video[t0:t1]> must be immediately followed by <audio[t0:t1]>.")
                n += 1
                i = a.end()
            return n

        total_pairs = 0
        for msg in messages:
            if isinstance(msg.get("content"), str):
                total_pairs += count_pairs(msg["content"])

        # === 3) 构建“整段”的时间索引，并切成不重叠 chunk ===
        def _non_empty(chunks):
            # 过滤零长度
            return [(int(s), int(e)) for (s, e) in chunks if int(e) > int(s)]

        t_ntoken_per_chunk = int(getattr(processor, "position_id_per_seconds", 25) *getattr(processor, "seconds_per_chunk", 0.4))

        video_chunk_indices = []
        audio_chunk_indices = []

        if expand and len(video_grid_thw):
            T, H, W = map(int, video_grid_thw[0].tolist())
            merge = getattr(video_processor, "merge_size", 2)
            v_tokens_per_grid = (H // merge) * (W // merge)
            sec_per_grid = float(video_second_per_grid[0])
            pos_per_sec = getattr(processor, "position_id_per_seconds", 25)

            # 每个 temporal grid 的“时间位移索引”
            t_index = (torch.arange(T) * sec_per_grid * pos_per_sec).long()
            # 展成 token 粒度（每个 grid 复制 v_tokens_per_grid 次）
            video_t_index = t_index.view(-1, 1).expand(-1, v_tokens_per_grid).reshape(-1)

            # 切块 + 过滤空块
            video_chunk_indices = _non_empty(processor.get_chunked_index(video_t_index, t_ntoken_per_chunk))

        if expand and audio_lengths and audio_lengths[0] is not None:
            L = int(audio_lengths[0])
            audio_t_index = torch.arange(L).long()  # 25Hz 等间隔
            audio_chunk_indices = _non_empty(processor.get_chunked_index(audio_t_index, t_ntoken_per_chunk))

        # 截断到“消息里的切片对数”，多余的丢弃，避免生成比特征更多的占位
        if len(video_chunk_indices) > total_pairs:
            video_chunk_indices = video_chunk_indices[:total_pairs]
        if len(audio_chunk_indices) > total_pairs:
            audio_chunk_indices = audio_chunk_indices[:total_pairs]

        # === 4) 消费 chunk：遇到一对切片就各取一个 chunk；块不够则 n_v/n_a=0 ===
        v_ptr, a_ptr = 0, 0
        for msg in messages:
            content = msg["content"]
            out = []
            i = 0
            while True:
                v_m = VIDEO_SLICE_RE.search(content, i)
                a_m = AUDIO_SLICE_RE.search(content, i)
                if not v_m:
                    out.append(content[i:])
                    break
                if not a_m or a_m.start() != v_m.end():
                    raise ValueError("Each <video[t0:t1]> must be immediately followed by <audio[t0:t1]>.")

                # 抄入片段前的纯文本
                out.append(content[i:v_m.start()])

                # 这对切片不再用 t0,t1 算长度，而是拿“下一个 chunk”
                n_v = 0
                if v_ptr < len(video_chunk_indices):
                    s, e = video_chunk_indices[v_ptr]
                    n_v = e - s
                    v_ptr += 1

                n_a = 0
                if a_ptr < len(audio_chunk_indices):
                    s, e = audio_chunk_indices[a_ptr]
                    n_a = e - s
                    a_ptr += 1

                # 构造占位（注意：n_v 或 n_a 为 0 时，不放 modality token，只放 BOS/EOS）
                ph_parts = []

                if n_v > 0:
                    seg = self.vision_bos_token + (self.video_token * int(n_v)) + self.vision_eos_token
                    ph_parts.append(seg)

                if n_a > 0:
                    seg = self.audio_bos_token + (self.audio_token * int(n_a)) + self.audio_eos_token
                    ph_parts.append(seg)

                ph = "".join(ph_parts)

                out.append(ph)
                i = a_m.end()

            msg["content"] = "".join(out)

        return messages



PLUGINS = {
    "base": BasePlugin,
    "qwen2_omni": Qwen2OmniPlugin,
    "qwen2_vl": Qwen2VLPlugin
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

# ===== helper for external inference (Route-A) =====

def get_qwen2_omni_plugin():
    """
    Return a ready-to-use Qwen2OmniPlugin instance for external inference scripts.
    """
    # 这里假定 Qwen2OmniPlugin 就在本文件中已定义
    return Qwen2OmniPlugin()


def routeA_prepare(
    processor,
    messages,
    videos,
    audios,
    add_stream_generation_prompt: bool = True,
):
    """
    External entry for Route-A inference.

    Steps:
    1) Build multimodal inputs (pixel_values_videos, video_grid_thw, input_features, feature_attention_mask, ...)
       via Qwen2OmniPlugin._get_mm_inputs(...) on the SAME processor used in training.
    2) Expand <video[t0:t1]><audio[t0:t1]> placeholders to BOS + VIDEO_TOKEN*n + EOS (+ audio ...) by
       Qwen2OmniPlugin.process_messages(...).
    3) Render final text by processor.apply_chat_template(..., add_stream_generation_prompt=...).

    Args:
        processor: the AutoProcessor (Qwen2.5-Omni) with trust_remote_code=True
        messages: list[dict] chat like [{"role":"user","content":"..."}, {"role":"assistant","content":"<video...><audio...>"}]
        videos:   list[list[str]] or list[str], paths to videos (batch-aware). For single sample, pass [[/path/to.mp4]]
        audios:   list[list[str]] or list[str], paths to audios. For single sample, pass [[/path/to.wav]]
        add_stream_generation_prompt: whether to end with ']\nAssistant:' (stream generation mode)

    Returns:
        text: str
        mm_inputs: dict[str, torch.Tensor]
        messages_expanded: list[dict]  # messages whose assistant content has been expanded to modality tokens
    """
    plugin = get_qwen2_omni_plugin()

    # 统一成 batch-aware 的 [[...]] 形式
    if isinstance(videos, list) and (len(videos) > 0) and isinstance(videos[0], str):
        videos = [videos]
    if isinstance(audios, list) and (len(audios) > 0) and isinstance(audios[0], str):
        audios = [audios]

    # 1) 先做多模态特征（形状探测也在这里完成）
    mm_inputs = plugin._get_mm_inputs(images=[], videos=videos, audios=audios, processor=processor)

    # 2) 再展开占位到 token（非常关键！）
    messages_expanded = plugin.process_messages(messages=messages, images=[], videos=videos, audios=audios, processor=processor)

    # 3) 渲染文本
    text = processor.apply_chat_template(messages_expanded, add_stream_generation_prompt=add_stream_generation_prompt, tokenize=False)

    return text, mm_inputs, messages_expanded
