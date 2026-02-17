from __future__ import annotations

import onnxruntime as ort


def _build_session_options(
    intra_op_num_threads: int = 0,
    inter_op_num_threads: int = 0,
) -> ort.SessionOptions:
    """Create ONNX Runtime session options tuned for inference."""
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    if intra_op_num_threads > 0:
        session_options.intra_op_num_threads = intra_op_num_threads
    if inter_op_num_threads > 0:
        session_options.inter_op_num_threads = inter_op_num_threads

    return session_options


def _build_providers(
    *,
    device: str,
    prefer_gpu: bool,
    cuda_device_id: int,
) -> list[str | tuple[str, dict[str, str]]]:
    """Resolve providers with CUDA preference and CPU fallback."""
    available = ort.get_available_providers()

    use_cuda = prefer_gpu
    resolved_device_id = cuda_device_id
    normalized_device = str(device).lower()
    if normalized_device.startswith("cuda"):
        use_cuda = True
        _, _, device_id = normalized_device.partition(":")
        if device_id.isdigit():
            resolved_device_id = int(device_id)

    providers: list[str | tuple[str, dict[str, str]]] = []
    if use_cuda and "CUDAExecutionProvider" in available:
        providers.append(
            (
                "CUDAExecutionProvider",
                {
                    "device_id": str(resolved_device_id),
                    "do_copy_in_default_stream": "1",
                },
            )
        )
    providers.append("CPUExecutionProvider")
    return providers


def create_inference_session(
    checkpoint_path: str,
    *,
    device: str = "cpu",
    prefer_gpu: bool = True,
    cuda_device_id: int = 0,
    intra_op_num_threads: int = 0,
    inter_op_num_threads: int = 0,
) -> ort.InferenceSession:
    session_options = _build_session_options(
        intra_op_num_threads=intra_op_num_threads,
        inter_op_num_threads=inter_op_num_threads,
    )
    providers = _build_providers(
        device=device,
        prefer_gpu=prefer_gpu,
        cuda_device_id=cuda_device_id,
    )
    return ort.InferenceSession(
        checkpoint_path,
        sess_options=session_options,
        providers=providers,
    )
