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
  
* 2-8_SSD_inference
  * utils\ssd_model.py:809
    "return self.detect(output([0], output[1], output[2])"
  * RuntimeError: Legacy autograd function with non-static forward method is deprecated. Please use new-style autograd function with static forward method.
  * 원인) Pytorch 버전 업그레이드로 autograd function(forward function)을 수정해야 한다.
  * 해결방법) Detect 클래스의 init 메소드 삭제 후 forward 메소드를 staticmethod로 변경 후 forward의 self를 모두 ctx로 변경한다. SSD 클래스의 init 메소드를 수정한다.
  * 변경 적용)    
    ```python
    class Detect(Function):
    @staticmethod
    def forward(ctx, loc_data, conf_data, dbox_list):
        ctx.softmax = nn.Softmax(dim=-1)
        ctx.conf_thresh = 0.01
        ctx.top_k = 200
        ctx.nms_thresh = 0.45
        
        num_batch = loc_data.size(0)
        num_dbox = loc_data.size(1) 
        num_classes = conf_data.size(2)  

        conf_data = ctx.softmax(conf_data)
        output = torch.zeros(num_batch, num_classes, ctx.top_k, 5)
        conf_preds = conf_data.transpose(2, 1)

        for i in range(num_batch):
            decoded_boxes = decode(loc_data[i], dbox_list)
            conf_scores = conf_preds[i].clone()

            for cl in range(1, num_classes):
                c_mask = conf_scores[cl].gt(ctx.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.nelement() == 0:  # nelementで要素数の合計を求める
                    continue
    
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                ids, count = nm_suppression(
                    boxes, scores, ctx.nms_thresh, ctx.top_k)
    
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),
                                                   boxes[ids[:count]]), 1)
        return output  
    ```
    ```python
    class SSD(nn.Module):
    def __init__(self, phase, cfg):
        # 앞의 내용 생략
        if phase == 'inference':
            self.detect = Detect.apply
    def forward(self, x):
        # forward 메서드는 수정 사항 없음
    ```
