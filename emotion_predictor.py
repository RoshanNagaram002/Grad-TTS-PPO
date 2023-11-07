import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from abc import ABC, abstractmethod
from speechbrain.pretrained.interfaces import foreign_class
from torchaudio.transforms import Resample

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EmotionPredictor(ABC):
    """
    Abstract class for speech prediction models.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def predict_emotion(self, wav):
        pass


    @abstractmethod
    def predict_emotion_batch(self, wav):
        pass

class SuperbPredictor(EmotionPredictor):
    def __init__(self):
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-er")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")
        # self.model.save_pretrained("./superb_emotion_predictor")
        # self.feature_extractor.save_pretrained("./super_emotion_predictor")
        # self.model = Wav2Vec2ForSequenceClassification.from_pretrained("./superb_emotion_predictor")
        # self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("./super_emotion_predictor")
        self.transform = Resample(22050, 16000).to(device)

    def predict_emotion(self, wav):
        wav = self.transform(wav)
        inputs = self.feature_extractor(wav, sampling_rate=16000, padding=True, return_tensors="pt")
        logits = self.model(**inputs).logits
        return torch.nn.functional.softmax(logits)
    
    def predict_emotion_batch(self, wav_batch):
        wav_batch = self.transform(wav_batch)
        inputs = self.feature_extractor(wav_batch.tolist(), sampling_rate=16000, padding=True, return_tensors="pt")
        logits = self.model(**inputs).logits
        return torch.nn.functional.softmax(logits, dim=1)

    def scores_to_emotion(self, scores):
        return [self.model.config.id2label[i.item()] for i in torch.argmax(scores, dim=1)]