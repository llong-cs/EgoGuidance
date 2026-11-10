# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from functools import lru_cache

import av
import cv2
import numpy as np

import torch  # isort: skip
import torchvision  # isort: skip

# Import decord with graceful fallback
try:
    import decord

    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

try:
    import torchcodec

    TORCHCODEC_AVAILABLE = True
except (ImportError, RuntimeError):
    TORCHCODEC_AVAILABLE = False


def _configure_threads(stream) -> None:
    """Pick a safe decode-threading policy per codec.

    Two distinct decoder pathologies motivate a codec-aware policy:

    * av1 / libdav1d (e.g. LIBERO and DROID clips): default frame-threading
      (thread_count=0 == ncpu) leaks per-thread frame buffers across open/close — RSS
      grows ~2.8 MB/decode unbounded until ``avcodec_open2`` fails with ENOMEM.
      Single-threaded decode keeps memory bounded and is faster for small frames.

    * vp9 (e.g. Ego4D clips): memory growth is thread-INDEPENDENT (a separate per-open
      leak, handled by worker recycling — see video_datasets.samples_per_epoch), so it
      is not single-threaded: single-threaded VP9 at 1440x1080 is ~2.2x slower and
      turns dataloader workers into training stragglers. Two threads per decoder with
      more parallel workers gave the best throughput with a ~16-CPU-per-rank budget
      shared by two co-training loaders.

    Must be set BEFORE the first ``decode()``.
    """
    try:
        name = (getattr(stream.codec_context, "name", "") or "").lower()
        if "av1" in name or "dav1d" in name or "libaom" in name:
            stream.thread_type = "NONE"
            stream.codec_context.thread_count = 1
        else:
            stream.codec_context.thread_count = 2
    except Exception:
        # Never let a threading hint break decoding; fall back to defaults.
        pass


@lru_cache(maxsize=1024)
def _stream_start_seconds(video_path: str) -> float:
    """Return the absolute PTS origin PyAV uses for timestamp lookup.

    Decord normalizes ``get_frame_timestamp`` to zero even when the encoded
    stream starts at a non-zero PTS.  StarVLA timestamps are expressed on the
    media timeline (the same convention as the PyAV backend), so comparing them
    directly to Decord's normalized values selects a frame late by exactly the
    stream start offset.  Opening the container for metadata does not initialize
    the decoder; cache the resulting scalar per worker to keep the fast path
    cheap without retaining any container or frame objects.
    """

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        if stream.start_time is None:
            return 0.0
        return float(stream.start_time * stream.time_base)


_PYAV_BATCH_MAX_SPAN_SECONDS = 1.0


def _pyav_frames_by_indices_per_seek(video_path: str, indices) -> np.ndarray:
    """Exact PyAV path: seek and decode once per requested frame."""

    container = None
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        _configure_threads(stream)
        time_base = float(stream.time_base)
        fps = float(stream.average_rate) if stream.average_rate else float(stream.guessed_rate)
        loaded_frames = []

        for target_idx in indices:
            target_ts = target_idx / fps
            target_pts = int(target_ts / time_base)
            container.seek(target_pts, stream=stream, backward=True, any_frame=False)

            closest_frame = None
            closest_idx_diff = float("inf")
            decoder = container.decode(video=0)
            try:
                for frame in decoder:
                    current_ts = float(frame.pts * time_base)
                    current_idx = round(current_ts * fps)
                    current_diff = abs(current_idx - target_idx)

                    if current_diff < closest_idx_diff:
                        closest_idx_diff = current_diff
                        closest_frame = frame

                    if current_idx > target_idx and current_diff > closest_idx_diff:
                        break
                    if current_idx > target_idx + 5:
                        break
            finally:
                decoder.close()

            if closest_frame is None:
                raise ValueError(f"Unable to find frame at index {target_idx}")
            loaded_frames.append(closest_frame.to_ndarray(format="rgb24"))

        return np.array(loaded_frames)
    finally:
        if container is not None:
            container.close()


def _pyav_frames_by_indices(video_path: str, indices) -> np.ndarray:
    """Decode a compact index window after one seek, preserving PyAV pixels.

    StarVLA normally requests ``[observation, *future_targets]`` from one short
    temporal window. Seeking independently for each target repeats demux and GOP
    decode work. For windows of at most one second, seek once before the earliest
    target and select every nearest frame during the same sequential decode. Wider
    requests fall back to one seek per frame to avoid scanning long videos.
    """

    targets = list(indices)
    if not targets:
        return np.array([])

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        _configure_threads(stream)
        time_base = float(stream.time_base)
        fps = float(stream.average_rate) if stream.average_rate else float(stream.guessed_rate)
        if (max(targets) - min(targets)) / fps > _PYAV_BATCH_MAX_SPAN_SECONDS:
            return _pyav_frames_by_indices_per_seek(video_path, targets)

        earliest = min(targets)
        latest = max(targets)
        container.seek(int((earliest / fps) / time_base), stream=stream, backward=True, any_frame=False)

        closest_frames = [None] * len(targets)
        closest_diffs = [float("inf")] * len(targets)
        latest_slot = targets.index(latest)
        decoder = container.decode(video=0)
        try:
            for frame in decoder:
                current_ts = float(frame.pts * time_base)
                current_idx = round(current_ts * fps)
                for slot, target_idx in enumerate(targets):
                    current_diff = abs(current_idx - target_idx)
                    if current_diff < closest_diffs[slot]:
                        closest_diffs[slot] = current_diff
                        closest_frames[slot] = frame

                if current_idx > latest and abs(current_idx - latest) > closest_diffs[latest_slot]:
                    break
                if current_idx > latest + 5:
                    break
        finally:
            decoder.close()

        if any(frame is None for frame in closest_frames):
            missing = [targets[i] for i, frame in enumerate(closest_frames) if frame is None]
            raise ValueError(f"Unable to find frames at indices {missing}")
        return np.array([frame.to_ndarray(format="rgb24") for frame in closest_frames])


def _pyav_frames_by_timestamps_per_seek(video_path: str, timestamps) -> np.ndarray:
    """Exact PyAV path: seek and decode once per timestamp."""

    container = None
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        _configure_threads(stream)
        time_base = float(stream.time_base)
        loaded_frames = []

        for target_ts in timestamps:
            target_pts = int(target_ts / time_base)
            container.seek(target_pts, stream=stream, backward=True, any_frame=False)

            closest_frame = None
            closest_ts_diff = float("inf")
            decoder = container.decode(video=0)
            try:
                for frame in decoder:
                    current_ts = float(frame.pts * time_base)
                    current_diff = abs(current_ts - target_ts)

                    if current_diff < closest_ts_diff:
                        closest_ts_diff = current_diff
                        closest_frame = frame

                    if current_ts > target_ts and current_diff > closest_ts_diff:
                        break
                    if current_ts > target_ts + 1.0:
                        break
            finally:
                decoder.close()

            if closest_frame is None:
                raise ValueError(f"Unable to find frame at timestamp {target_ts}")
            loaded_frames.append(closest_frame.to_ndarray(format="rgb24"))

        return np.array(loaded_frames)
    finally:
        if container is not None:
            container.close()


def _pyav_frames_by_timestamps(video_path: str, timestamps) -> np.ndarray:
    """Decode a compact timestamp window after one seek, preserving PyAV RGB."""

    targets = list(timestamps)
    if not targets:
        return np.array([])
    if max(targets) - min(targets) > _PYAV_BATCH_MAX_SPAN_SECONDS:
        return _pyav_frames_by_timestamps_per_seek(video_path, targets)

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        _configure_threads(stream)
        time_base = float(stream.time_base)
        earliest = min(targets)
        latest = max(targets)
        container.seek(int(earliest / time_base), stream=stream, backward=True, any_frame=False)

        closest_frames = [None] * len(targets)
        closest_diffs = [float("inf")] * len(targets)
        latest_slot = targets.index(latest)
        decoder = container.decode(video=0)
        try:
            for frame in decoder:
                current_ts = float(frame.pts * time_base)
                for slot, target_ts in enumerate(targets):
                    current_diff = abs(current_ts - target_ts)
                    if current_diff < closest_diffs[slot]:
                        closest_diffs[slot] = current_diff
                        closest_frames[slot] = frame

                if current_ts > latest and abs(current_ts - latest) > closest_diffs[latest_slot]:
                    break
                if current_ts > latest + 1.0:
                    break
        finally:
            decoder.close()

        if any(frame is None for frame in closest_frames):
            missing = [targets[i] for i, frame in enumerate(closest_frames) if frame is None]
            raise ValueError(f"Unable to find frames at timestamps {missing}")
        return np.array([frame.to_ndarray(format="rgb24") for frame in closest_frames])


def get_frames_by_indices(
    video_path: str,
    indices: list[int] | np.ndarray,
    video_backend: str = "decord",
    video_backend_kwargs: dict | None = None,
) -> np.ndarray:
    video_backend_kwargs = video_backend_kwargs or {}
    if video_backend == "decord":
        if not DECORD_AVAILABLE:
            raise ImportError("decord is not available.")
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        frames = vr.get_batch(indices)
        return frames.asnumpy()
    elif video_backend == "torchcodec":
        if not TORCHCODEC_AVAILABLE:
            raise ImportError("torchcodec is not available.")
        decoder = torchcodec.decoders.VideoDecoder(
            video_path, device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
        )
        return decoder.get_frames_at(indices=indices).data.numpy()
    elif video_backend == "opencv":
        frames = []
        cap = cv2.VideoCapture(video_path, **video_backend_kwargs)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Unable to read frame at index {idx}")
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        return frames
    elif video_backend == "pyav":
        return _pyav_frames_by_indices(video_path, indices)
    else:
        raise NotImplementedError


def get_frames_by_timestamps(
    video_path: str,
    timestamps: list[float] | np.ndarray,
    video_backend: str = "decord",
    video_backend_kwargs: dict | None = None,
) -> np.ndarray:
    """Get frames from a video at specified timestamps.
    Args:
        video_path (str): Path to the video file.
        timestamps (list[int] | np.ndarray): Timestamps to retrieve frames for, in seconds.
        video_backend (str, optional): Video backend to use. Defaults to "decord".
    Returns:
        np.ndarray: Frames at the specified timestamps.
    """
    video_backend_kwargs = video_backend_kwargs or {}
    if video_backend == "decord":
        # For some GPUs, AV format data cannot be read
        if not DECORD_AVAILABLE:
            raise ImportError("decord is not available.")
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        num_frames = len(vr)
        # Retrieve the timestamps for each frame in the video
        frame_ts = np.asarray(vr.get_frame_timestamp(range(num_frames)), dtype=np.float64)
        if frame_ts.ndim != 2 or frame_ts.shape != (num_frames, 2):
            raise ValueError(
                f"Decord returned invalid frame timestamps for {video_path}: "
                f"shape={frame_ts.shape}, expected={(num_frames, 2)}"
            )
        requested = np.asarray(timestamps, dtype=np.float64)
        # Decord reports timestamps relative to stream start, whereas PyAV frame
        # PTS values include stream.start_time. Restore that offset before nearest
        # lookup so both backends select the same physical frame (including the
        # first-frame tie rule from np.argmin / PyAV's strict '<' update).
        absolute_starts = frame_ts[:, 0] + _stream_start_seconds(video_path)
        indices = np.abs(absolute_starts[:, np.newaxis] - requested[np.newaxis, :]).argmin(axis=0)
        frames = vr.get_batch(indices)
        return frames.asnumpy()
    elif video_backend == "torchcodec":
        if not TORCHCODEC_AVAILABLE:
            raise ImportError("torchcodec is not available.")
        decoder = torchcodec.decoders.VideoDecoder(
            video_path, device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
        )
        return decoder.get_frames_played_at(seconds=timestamps).data.numpy()
    elif video_backend == "opencv":
        # Open the video file
        cap = cv2.VideoCapture(video_path, **video_backend_kwargs)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        # Retrieve the total number of frames
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Calculate timestamps for each frame
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_ts = np.arange(num_frames) / fps
        frame_ts = frame_ts[:, np.newaxis]  # Reshape to (num_frames, 1) for broadcasting
        # Map each requested timestamp to the closest frame index
        indices = np.abs(frame_ts - timestamps).argmin(axis=0)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Unable to read frame at index {idx}")
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        return frames
    elif video_backend == "pyav":
        return _pyav_frames_by_timestamps(video_path, timestamps)

    elif video_backend == "torchvision_av":
        torchvision.set_video_backend("pyav")
        loaded_frames = []
        loaded_ts = []

        reader = None
        try:
            reader = torchvision.io.VideoReader(video_path, "video")

            for target_ts in timestamps:
                # Reset reader state
                reader.seek(target_ts, keyframes_only=True)

                closest_frame = None
                closest_ts_diff = float("inf")

                for frame in reader:
                    current_ts = frame["pts"]
                    current_diff = abs(current_ts - target_ts)

                    if closest_frame is None:
                        closest_frame = frame

                    if current_diff < closest_ts_diff:
                        # Release the previous frame
                        if closest_frame is not None:
                            del closest_frame
                        closest_ts_diff = current_diff
                        closest_frame = frame
                    else:
                        # The time difference starts to increase, stop searching
                        break

                if closest_frame is not None:
                    frame_data = closest_frame["data"]
                    if isinstance(frame_data, torch.Tensor):
                        frame_data = frame_data.cpu().numpy()
                    loaded_frames.append(frame_data)
                    loaded_ts.append(closest_frame["pts"])

                    # Immediately release frame reference
                    del closest_frame

        finally:
            # Thoroughly clean resources
            if reader is not None:
                if hasattr(reader, "_c"):
                    reader._c = None
                if hasattr(reader, "container"):
                    reader.container.close()
                    reader.container = None

        frames = np.array(loaded_frames)
        return frames.transpose(0, 2, 3, 1)
    else:
        raise NotImplementedError


def get_all_frames(
    video_path: str,
    video_backend: str = "decord",
    video_backend_kwargs: dict | None = None,
    resize_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Get all frames from a video.
    Args:
        video_path (str): Path to the video file.
        video_backend (str, optional): Video backend to use. Defaults to "decord".
        video_backend_kwargs (dict, optional): Keyword arguments for the video backend.
        resize_size (tuple[int, int], optional): Resize size for the frames. Defaults to None.
    """
    video_backend_kwargs = video_backend_kwargs or {}
    if video_backend == "decord":
        if not DECORD_AVAILABLE:
            raise ImportError("decord is not available.")
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        frames = vr.get_batch(range(len(vr))).asnumpy()
    elif video_backend == "torchcodec":
        if not TORCHCODEC_AVAILABLE:
            raise ImportError("torchcodec is not available.")
        decoder = torchcodec.decoders.VideoDecoder(
            video_path, device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
        )
        frames = decoder.get_frames_at(indices=range(len(decoder)))
        return frames.data.numpy(), frames.pts_seconds.numpy()
    elif video_backend == "pyav":
        container = av.open(video_path)
        _configure_threads(container.streams.video[0])
        frames = []
        for frame in container.decode(video=0):
            frame = frame.to_ndarray(format="rgb24")
            frames.append(frame)
        frames = np.array(frames)
    elif video_backend == "torchvision_av":
        # set backend and reader
        torchvision.set_video_backend("pyav")
        reader = torchvision.io.VideoReader(video_path, "video")
        frames = []
        for frame in reader:
            frames.append(frame["data"].numpy())
        frames = np.array(frames)
        frames = frames.transpose(0, 2, 3, 1)
    else:
        raise NotImplementedError(f"Video backend {video_backend} not implemented")
    # resize frames if specified
    if resize_size is not None:
        frames = [cv2.resize(frame, resize_size) for frame in frames]
        frames = np.array(frames)
    return frames
