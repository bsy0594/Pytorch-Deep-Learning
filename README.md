# 🖥️ Pytorch-Deep-Learning
<aside>
💡 만들면서 배우는 파이토치 딥러닝 : 12가지 모델로 알아보는 딥러닝 응용법 💡

</aside>

## 📝 Contents

1. 화상 분류와 전이학습 (VGG)
    
    > 24.02.01 - 24.02.04
    > 
2. 물체 인식 (SSD) 
    
    > 24.02.05 - 24.02.14
    > 
3. 시맨틸 분할 (PSPNet) 
    
    > 24.02.15 -
    > 
4. 자세 추정 (OpenPose)
    
    > 24
    > 
5. GAN을 활용한 화상 생성 (DCGAN, Self-Attention GAN)
    
    > 24
    > 
6. GAN을 활용한 이상 감지 (AnoGAN, Efficient GAN)
    
    > 24
    > 
7. 자연어 처리를 활용한 감정 분석 (Transformer)
    
    > 24
    > 
8. 자연어 처리를 활용한 감정 분석 (BERT)
    
    > 24
    > 
9. 동영상 분류 (3DCNN, ECO)
    
    > 24
    >
## 🌈 Original Source
©️ https://github.com/YutaroOgawa/pytorch_advanced

## 🛠️ Modification
✅ 2-2-3_Dataset_DataLoader
  * 에러 메세지
    * ValueError: setting an array element with a sequence.  
  * 오류 발생 위치
    * utils\data_augementation.py:246
    * "mode = random.choice(self.sample_options)"
  * 원인
    * ndarray의 구간별 길이가 일정하지 않기 때문이다.
  * 해결방법
    * sample_options을 Numpy배열로 바꿔주고, dype = object를 추가해준다.
  * 변경사항
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
  
✅ 2-8_SSD_inference
  * 에러 메세지
    * RuntimeError: Legacy autograd function with non-static forward method is deprecated. Please use new-style autograd function with static forward method.
  * 오류 발생 위치
    * utils\ssd_model.py:809
    * "return self.detect(output([0], output[1], output[2])"
  * 원인
    * Pytorch 버전 업그레이드로 autograd function(forward function)을 수정해야 한다.
  * 해결방법
    * Detect 클래스의 init 메소드를 삭제하고, forward 메소드를 staticmethod로 변경 한 뒤, self를 모두 ctx로 변경한다.
    * SSD 클래스의 init 메소드 중 Detect 호출 부분을 수정한다.
  * 변경사항    
    ```python
    class Detect(Function):
    	@staticmethod
    	def forward(ctx, loc_data, conf_data, dbox_list):
    		ctx.softmax = nn.Softmax(dim=-1)
    		ctx.conf_thresh = 0.01
    		ctx.top_k = 200
    		ctx.nms_thresh = 0.45
    		# 이런 식으로 self를 모두 ctx로 바꿔주면 된다.
    ```
    ```python
    class SSD(nn.Module):
    	def __init__(self, phase, cfg):
    		# 앞의 내용 생략
    		if phase == 'inference':
    		    	self.detect = Detect.apply # 원래는 Detect()
    ```
