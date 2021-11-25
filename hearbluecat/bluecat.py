import numpy as np
from typing import Tuple

import torch
from torch import Tensor

import hearbluecat.torchcrepe
import hearbluecat.wav2vec2
import hearbluecat.torchopenl3


class BlueCat(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.modelA = hearbluecat.torchcrepe.load_model()
        self.modelB = hearbluecat.wav2vec2.load_model()
        self.modelC = hearbluecat.torchopenl3.load_model()
        
        self.sample_rate = 16000
        self.scene_embedding_size = 9216
        self.timestamp_embedding_size = 9216


def load_model(model_file_path: str = ""):
    if model_file_path != "":
        print("Model weights file passed, but it is not used by HearBlueCat.")
    model = BlueCat()
    return model


def get_timestamp_embeddings(
    audio: Tensor,
    model: torch.nn.Module
) -> Tuple[Tensor, Tensor]:
    """
    This function returns embeddings at regular intervals centered at timestamps. Both
    the embeddings and corresponding timestamps (in milliseconds) are returned.

    Args:
        audio: n_sounds x n_samples of mono audio in the range [-1, 1].
        model: Loaded model.
        hop_size_samples: Hop size in samples.

    Returns:
        - Tensor: embeddings, A float32 Tensor with shape (n_sounds, n_timestamps,
            model.timestamp_embedding_size).
        - Tensor: timestamps, Centered timestamps in milliseconds corresponding
            to each embedding in the output. Shape: (n_sounds, n_timestamps).
    """

    # Assert audio is of correct shape
    if audio.ndim != 2:
        raise ValueError(
            "audio input tensor must be 2D with shape (n_sounds, num_samples)"
        )

    embeddingsA = hearbluecat.torchcrepe.get_timestamp_embeddings(audio, model.modelA)
    embeddingsB = hearbluecat.wav2vec2.get_timestamp_embeddings(audio, model.modelB)
    embeddingsC = hearbluecat.torchopenl3.get_timestamp_embeddings(audio, model.modelC)

    print("Shapes of individial model embeddings: ", embeddingsA[0].shape,embeddingsB[0].shape,embeddingsC[0].shape)

    # HACK: clip the last frames off the two longest embeddings, by the length of the shortest
    shortest_idx = np.argmin([embeddingsA[0].shape[1], embeddingsB[0].shape[1], embeddingsC[0].shape[1]])
    new_frame_count = [embeddingsA, embeddingsB, embeddingsC][shortest_idx][0].shape[1]

    embeddingsA = (embeddingsA[0][:, :new_frame_count, :], embeddingsA[1][:, :new_frame_count])
    embeddingsB = (embeddingsB[0][:, :new_frame_count, :], embeddingsB[1][:, :new_frame_count])
    embeddingsC = (embeddingsC[0][:, :new_frame_count, :], embeddingsC[1][:, :new_frame_count])

    print("Shapes of individial model embeddings after end clipping: ", embeddingsA[0].shape,embeddingsB[0].shape,embeddingsC[0].shape)

    embedA = embeddingsA[0].flatten()
    embedB = embeddingsB[0].flatten()
    embedC = embeddingsC[0].flatten()
    
    print("A:", embeddingsA[1][0])
    print("B:", embeddingsB[1][0])
    print("C:", embeddingsC[1][0])

    print("Shapes of invidial model embeddings after flattening: ", embedA.shape, embedB.shape, embedC.shape)

    embeddings = torch.cat((embedA, embedB, embedC))

    embeddings = embeddings.reshape(embeddingsA[0].shape[0], embeddingsA[0].shape[1], 9216)

    timestamps = embeddingsA[1]

    return embeddings, timestamps


def get_scene_embeddings(
    audio: Tensor,
    model: torch.nn.Module,
) -> Tensor:
    """
    This function returns a single embedding for each audio clip. In this baseline
    implementation we simply summarize the temporal embeddings from
    get_timestamp_embeddings() using torch.mean().

    Args:
        audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in
            a batch will be padded/trimmed to the same length.
        model: Loaded model.

    Returns:
        - embeddings, A float32 Tensor with shape
            (n_sounds, model.scene_embedding_size).
    """
    embeddings, _ = get_timestamp_embeddings(
        audio, model
    )
    embeddings = torch.mean(embeddings, dim=1)
    return embeddings