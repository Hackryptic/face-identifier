import argparse
import os
import torch

# train_options.py
# 얼굴 인식 사전학습 과정에서 사용하는 argument 옵션들을 정의한 파일
# written by Hackryptic

# argument 옵션을 정의한 클래스

class TrainOptions():
    def __init__(self, is_train):
        self.parser = argparse.ArgumentParser()
        self.is_train = is_train

        self.initialize()

    def initialize(self):
        # 학습할 데이터의 경로
        self.parser.add_argument('--dataroot', type=str, default="/mnt/ssd/casia_faces", help='path to dataset')
        # 모델의 종류 (mobileNetv2 or inceptionresnetv1)
        self.parser.add_argument('--model_type', type=str, default="inceptionresnetv1", help='path to dataset')
        # 데이터 로딩 배치 사이즈
        self.parser.add_argument('--loading_batch_size', type=int, default=512, help='loading batch size')
        # 실제 학습 배치 사이즈 (블럭에서 튜플로 데이터를 뽑는 과정에서의 배치 사이즈)
        self.parser.add_argument('--tuple_batch_size', type=int, default=64, help='tuple batch size')
        # gpu-id 지정
        self.parser.add_argument('--gpu_ids', type=str, default=0, help='gpu ids')
        # 실험 이름 지정 (결과물은 실험의 이름으로 된 디렉터리 생성 후 저장)
        self.parser.add_argument('--name', type=str, default='2209121128', help='name of the experiment.')
        # adam optimzer의 beta1 값
        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        # adam optimzer의 beta2 값
        self.parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        # epoch 횟수
        self.parser.add_argument('--epoch', type=int, default=1000, help='epoch')
        #  decay 시작 epoch
        self.parser.add_argument('--decay_start_epoch', type=int, default=500, help='decay start epoch')
        # decay factor
        self.parser.add_argument('--decay_factor', type=float, default=0.98, help='decay factor')
        # weight update 간격 지정
        self.parser.add_argument('--update_step', type=int, default=1, help='update step')
        # Learning rate
        self.parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate')
        # triplet loss 및 quadruplet loss의 margin 값
        self.parser.add_argument('--margin', type=float, default=0.5, help='Margin (alpha)')
        # quadruplet loss의 두 번째 margin 값
        self.parser.add_argument('--margin2', type=float, default=0.25, help='Margin2 (alpha2)')
        # 블럭 안에 들어갈 클래스의 개수
        self.parser.add_argument('--class_per_block', type=int, default=100, help='classes per block')
        # 블럭 안에 들어갈 클래스 당 이미지의 개수
        self.parser.add_argument('--image_per_class', type=int, default=5, help='images per class')
        # epoch 당 step의 횟수 (모든 데이터를 1 epoch으로 지정할 시 많은 시간이 소요
        self.parser.add_argument('--epoch_size', type=int, default=1, help='epoch size (number of batch)')
        # ramdom sampling에서 hard sampling으로 전환할 시점을 지정
        self.parser.add_argument('--adapt_start_epoch', type=int, default=900, help='adapt start epoch') 




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

        # save
        
        file_name = './options/{}_opt.txt'.format(self.opt.name)
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        
        return self.opt
