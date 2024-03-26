from ultralytics.nn.modules.conv import RFB2
from ultralytics.nn.modules.conv import PyramidPooling
from ultralytics.nn.modules.conv import FFM
import torch

if __name__ == '__main__':
    # block = RFB2(256*3, 256, 4, d=[2, 3])  # [256*3 ,80 ,80] -> [256, 80, 80]
    # block = PyramidPooling(256, k=[2, 4, 6, 12])  # [256, 80, 80] -> [512, 80, 80]
    block = FFM(256*2, 256, k=3, is_cat=False)  # [512, 80, 80] -> [256, 80, 80]
    input = torch.rand(1, 512, 80, 80)
    output = block(input)
    print(input.size(), output.size())