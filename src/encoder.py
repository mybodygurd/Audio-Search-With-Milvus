import torch
from speechbrain.inference.speaker import EncoderClassifier
from sklearn.preprocessing import normalize


class FeatureExtractor:
    def __init__(self, model_name: str):
        self.model = EncoderClassifier.from_hparams(
            source=model_name,
            savedir="pretrained_models/ecapa"
        )

    def __call__(self, chunk):
        with torch.no_grad():
            emb = self.model.encode_batch(chunk)
        emb_np = emb.squeeze().numpy()
        return normalize(emb_np.reshape(1, -1), norm="l2").flatten()


