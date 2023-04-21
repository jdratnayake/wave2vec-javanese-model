from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

api_token = "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"
model_name = "indonesian-nlp/wav2vec2-indonesian-javanese-sundanese"
cache_dir = "model_cache"

processor = Wav2Vec2Processor.from_pretrained(model_name, use_auth_token=api_token, cache_dir=cache_dir)
model = Wav2Vec2ForCTC.from_pretrained(model_name, use_auth_token=api_token, cache_dir=cache_dir)
