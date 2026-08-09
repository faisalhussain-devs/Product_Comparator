from dataclasses import dataclass
from pathlib import Path

import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


HF_REPO_ID = "Faisal191/aspect-classifier"

RANKER_TOKENIZER_SUBFOLDER = "RankingReviews"
RANKER_MODEL_FILE = "RankingReviews/model.onnx"

ABSA_TOKENIZER_SUBFOLDER = "Domain_trained_encoder"
ABSA_MODEL_FILE = "HABSA/Habsa_v6_fp32.onnx"


@dataclass
class ONNXModel:
    tokenizer: object
    session: ort.InferenceSession


def get_execution_providers() -> list:
    available = ort.get_available_providers()

    if "CUDAExecutionProvider" in available:
        return [
            (
                "CUDAExecutionProvider",
                {
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "do_copy_in_default_stream": True,
                },
            ),
            "CPUExecutionProvider",
        ]

    return ["CPUExecutionProvider"]


def create_session(model_path: str) -> ort.InferenceSession:
    session_options = ort.SessionOptions()

    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    return ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=get_execution_providers(),
    )


def load_model(
    tokenizer_subfolder: str,
    model_file: str,
) -> ONNXModel:
    tokenizer = AutoTokenizer.from_pretrained(
        HF_REPO_ID,
        subfolder=tokenizer_subfolder,
    )

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=model_file,
        repo_type="model",
    )

    session = create_session(model_path)

    return ONNXModel(
        tokenizer=tokenizer,
        session=session,
    )


def load_ranker() -> ONNXModel:
    return load_model(
        tokenizer_subfolder=RANKER_TOKENIZER_SUBFOLDER,
        model_file=RANKER_MODEL_FILE,
    )


def load_absa() -> ONNXModel:
    return load_model(
        tokenizer_subfolder=ABSA_TOKENIZER_SUBFOLDER,
        model_file=ABSA_MODEL_FILE,
    )