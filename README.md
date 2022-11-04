# Vietnamese Self-Supervised Learning Wav2Vec2 model

## Model

We use wav2vec2 architecture for doing Self-Supervised learning

<img src="https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/wav2vec2.png" width=50% height=50%>

## Data

Our self-supervised model is pre-trained on a massive audio set of 13k hours of Vietnamese youtube audio, which includes:
  - Clean audio
  - Noise audio
  - Conversation
  - Multi-gender and dialects
  

## Download

We have already upload our pre-trained model to the Huggingface. The base model trained 35 epochs and the large model trained 20 epochs in about 30 days using TPU V3-8.

 - [Based version](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vi) ~ 95M params
 - [Large version](https://huggingface.co/nguyenvulebinh/wav2vec2-large-vi) ~ 317M params

## Usage

```python
from transformers import Wav2Vec2ForPreTraining, Wav2Vec2Processor

model_name = 'nguyenvulebinh/wav2vec2-base-vi'
# model_name = 'nguyenvulebinh/wav2vec2-large-vi'

model = Wav2Vec2ForPreTraining.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

```

Since our model has the same architecture as the English wav2vec2 version, you can use [this notebook](https://colab.research.google.com/drive/1FjTsqbYKphl9kL-eILgUc-bl4zVThL8F?usp=sharing) for more information on how to fine-tune the model.

## Finetuned version

You can find a finetuned version of base model on VLSP 2020 ASR dataset [here](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h). New finetuned version will be update soon (I hope 🤗).

## Acknowledgment

- I would like to thank Google for the support of Cloud credits and TPU quota. 
- Special thanks to my colleagues at [VietAI](https://vietai.org/) and [VAIS](https://vais.vn/) for their advice.

## Contact 

nguyenvulebinh@gmail.com / binh@vietai.org

[![Follow](https://img.shields.io/twitter/follow/nguyenvulebinh?style=social)](https://twitter.com/intent/follow?screen_name=nguyenvulebinh)


