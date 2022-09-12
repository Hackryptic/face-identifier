import argparse
import os
import torch

# test_options.py
# 얼굴 인식 테스트 과정에서 사용하는 argument 옵션들을 정의한 파일
# written by Hackryptic

# argument 옵션을 정의한 클래스
class TestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
                
        self.initialize()
        
    def initialize(self):
        # knn 학습 데이터의 경로
        self.parser.add_argument('--ref_path', type=str, default="./sample_data/ref", help='path to reference dataset')
        # 저장된 모델 파라미터 파일의 경로
        self.parser.add_argument('--param_path', type=str, default="./saved_model/casia_webface_pretrained.pt", help='path to saved parameters')
        # Backbone(Feature Extractor) 모델의 종류 (mobileNetv2 or inceptionresnetv1)
        self.parser.add_argument('--model_type', type=str, default="inceptionresnetv1", help='model type (mobilenetv2 or inceptionresnetv1')
        # 테스트 할 이미지의 경로
        self.parser.add_argument('--test_image', type=str, default="./sample_data/test/1/1.png", help='image to test')
        # Rejecttion 여부를 선택
        self.parser.add_argument('--rejection', type=bool, default="True", help='rejection on/off')
        # gpu id 선택
        self.parser.add_argument('--gpu_ids', type=str, default=1, help='gpu ids') 
                                  
                                    
                                      
    def mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path)
          
    def parse(self):
                  
        self.opt = self.parser.parse_args()

        print(self.opt)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
         
          
        return self.opt
