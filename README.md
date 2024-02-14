# Pytorch-Deep-Learning
만들면서 배우는 파이토치 딥러닝 : 12가지 모델로 알아보는 딥러닝 응용법

## Contents
1. 화상 분류와 전이학습 (VGG) | 24.02.01 - 24.02.04

2. 물체 인식 (SDD) | 24.02.05 - 24.02.14

3. 시맨틸 분할 (PSPNet) | 24.02.15 - 

4. 자세 추정 (OpenPose)

5. GAN을 활용한 화상 생성 (DCGAN, Self-Attention GAN)

6. GAN을 활용한 이상 감지 (AnoGAN, Efficient GAN)

7. 자연어 처리를 활용한 감정 분석 (Transformer)

8. 자연어 처리를 활용한 감정 분석 (BERT)

9. 동영상 분류 (3DCNN, ECO)

## Original Source
https://github.com/YutaroOgawa/pytorch_advanced

## Modification
* 2-2-3_Dataset_DataLoader
  * utils\data_augementation.py:246
    "mode = random.choice(self.sample_options)"
  * ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (6, ) + inhomogeneous part.
  * 원인) ndarray의 구간별 길이가 일정하지 않기 때문이다.
  * 해결방법) sample_options을 Numpy배열로 바꿔주고, dype = object를 추가해준다.
  * 변경 적용)    
    ```python
    class RandomSampleCrop(object):
    	def __init__(self):
    		self.sample_options = np.array([
    			None,
    			(0.1, None),
    			(0.3, None),
    			(0.7, None),
    			(0.9, None),
    			(None, None),
    		], dtype=object)
    ```
  
