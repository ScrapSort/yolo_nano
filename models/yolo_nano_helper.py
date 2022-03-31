'''
author: lingteng qiu

'''
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
from models.darknet import YOLOLayer,to_cpu

#ACTIVATE_FUNC
ACTIVATE={
	"relu":nn.ReLU,
	"relu6":nn.ReLU6,
	"leaky":nn.LeakyReLU
}
ARCHITECTURE ={
	'layer3':[['EP',150,325,2],['PEP',325,132,325],['PEP',325,124,325],['PEP',325,141,325],
			  ['PEP',325,140,325],['PEP',325,137,325],['PEP',325,135,325],['PEP',325,133,325],
			  ['PEP',325,140,325]],
	'layer4':[['EP',325,545,2],['PEP',545,276,545],['conv1x1',545,230],['EP',230,489,1],['PEP',489,213,469],
			  ['conv1x1',469,189]],
}
YOLO_ARCH = {
	"small": [(116, 90), (156, 198), (373, 326)],
	"middle":[(30, 61), (62, 45), (59, 119)],
	"large":[(10, 13), (16, 30), (33, 23)]
}


def conv1x1(input_channels, output_channels, stride=1, bn=True):
	# 1x1 convolution without padding
	if bn == True:
		return nn.Sequential(
			nn.Conv2d(
				input_channels, output_channels, kernel_size=1,
				stride=stride, bias=False),
			nn.BatchNorm2d(output_channels),
			nn.ReLU6(inplace=True)
		)
	else:
		return nn.Conv2d(
				input_channels, output_channels, kernel_size=1,
				stride=stride, bias=False)


def conv3x3(input_channels, output_channels, stride=1, bn=True):
	# 3x3 convolution with padding=1
	if bn == True:
		return nn.Sequential(
			nn.Conv2d(
				input_channels, output_channels, kernel_size=3,
				stride=stride, padding=1, bias=False),
			nn.BatchNorm2d(output_channels),
			nn.ReLU6(inplace=True)
		)
	else:
		nn.Conv2d(
				input_channels, output_channels, kernel_size=3,
				stride=stride, padding=1, bias=False)

def sepconv3x3(input_channels, output_channels, stride=1, expand_ratio=1):
	return nn.Sequential(
		# pw
		nn.Conv2d(
			input_channels, input_channels * expand_ratio,
			kernel_size=1, stride=1, bias=False),
		nn.BatchNorm2d(input_channels * expand_ratio),
		nn.ReLU6(inplace=True),
		# dw
		nn.Conv2d(
			input_channels * expand_ratio, input_channels * expand_ratio, kernel_size=3, 
			stride=stride, padding=1, groups=input_channels * expand_ratio, bias=False),
		nn.BatchNorm2d(input_channels * expand_ratio),
		nn.ReLU6(inplace=True),
		# pw-linear
		nn.Conv2d(
			input_channels * expand_ratio, output_channels,
			kernel_size=1, stride=1, bias=False),
		nn.BatchNorm2d(output_channels)
	)

class EP(nn.Module):
	def __init__(self, input_channels, output_channels, stride=1):
		super(EP, self).__init__()
		self.input_channels = input_channels
		self.output_channels = output_channels
		self.stride = stride
		self.use_res_connect = self.stride == 1 and input_channels == output_channels

		self.sepconv = sepconv3x3(input_channels, output_channels, stride=stride)
		
	def forward(self, x):
		if self.use_res_connect:
			return x + self.sepconv(x)
		
		return self.sepconv(x)

class PEP(nn.Module):
	def __init__(self, input_channels, output_channels, x, stride=1):
		super(PEP, self).__init__()
		self.input_channels = input_channels
		self.output_channels = output_channels
		self.stride = stride
		self.use_res_connect = self.stride == 1 and input_channels == output_channels

		self.conv = conv1x1(input_channels, x)
		self.sepconv = sepconv3x3(x, output_channels, stride=stride)
		
	def forward(self, x):		 
		out = self.conv(x)
		out = self.sepconv(out)
		if self.use_res_connect:
			return out + x

		return out


class FCA(nn.Module):
	def __init__(self, channels, reduction_ratio):
		super(FCA, self).__init__()
		self.channels = channels
		self.reduction_ratio = reduction_ratio

		hidden_channels = channels // reduction_ratio
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(
			nn.Linear(channels, hidden_channels, bias=False),
			nn.ReLU6(inplace=True),
			nn.Linear(hidden_channels, channels, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		b, c, _, _ = x.size()
		out = self.avg_pool(x).view(b, c)
		out = self.fc(out).view(b, c, 1, 1)
		out = x * out.expand_as(x)
		return out



class YoloNano2(nn.Module):
	'''
	Paper Structure Arch
	return three scale feature,int the paper :(52)4,(26)2,(13)1.
	each channel in here is only 75. for voc 2007 because: (num_class+5)*anchor because voc has 20 classes include background so the
	channel in here is 75
	'''
	def __init__(self,num_class = 20,num_anchor = 3,img_size = 416):
		__FUNC = {
			'EP': EP,
			'PEP': PEP,
			'conv1x1':conv1x1
		}
		super(YoloNano,self).__init__()
		self.num_class =num_class
		self.num_anchor = num_anchor
		self.img_size = img_size
		self.out_channel = (num_class+5)*num_anchor
		self.seen = 0

		self.layer0 = nn.Sequential(conv3x3(3,12,1),conv3x3(12,24,2))
		self.layer1 = nn.Sequential(PEP(24,7,24),
									EP(24,70,2),
									PEP(70,25,70),
									PEP(70,24,70),
									EP(70,150,2),
									PEP(150,56,150),
									conv1x1(150,150,1,1,1,use_relu=True)
									)
		self.attention = FCA(150,8)
		self.layer2 = nn.Sequential(PEP(150,73,150),
									  PEP(150,71,150),
									  PEP(150,75,150))
		layer3=[]
		for e in ARCHITECTURE['layer3']:
			layer3.append((__FUNC[e[0]](*e[1:])))
		self.layer3 = nn.Sequential(*layer3)
		layer4 = []
		for e in ARCHITECTURE['layer4']:
			layer4.append((__FUNC[e[0]](*e[1:])))
		self.layer4 = nn.Sequential(*layer4)

		#all below this u can change by your self
		self.layer5 = nn.Sequential(PEP(430,113,325),PEP(325,99,207),conv1x1(207,98,use_relu=True))
		#
		self.compress = conv1x1(189,105,use_relu = True)
		self.compress2 = conv1x1(98,47,use_relu=True)



		#Yolo_Layer using to regress the x,y,w,h
		self.scale_4 = nn.Sequential(
			PEP(197,58,122),
			PEP(122,52,87),
			PEP(87,47,93),
			nn.Conv2d(93,self.out_channel,kernel_size=1,stride = 1,padding =0,bias=True)
		)
		self.scale_2 = nn.Sequential(
			EP(98,183,1),
			nn.Conv2d(183,self.out_channel,kernel_size=1,stride = 1,padding =0,bias=True)
		)
		self.scale_1 = nn.Sequential(EP(189,462,1),nn.Conv2d(462,self.out_channel,kernel_size=1,stride = 1,padding =0,bias=True))




		#yolo0 : big_anchor
		#yolo1 : mid_anchor
		#yolo2 : small_anchor
		self.yolo0	= YOLOLayer(YOLO_ARCH['small'], self.num_class,self.img_size )
		self.yolo1	= YOLOLayer(YOLO_ARCH['middle'], self.num_class,self.img_size )
		self.yolo2	= YOLOLayer(YOLO_ARCH['large'], self.num_class,self.img_size )

		self.yolo_layers = [self.yolo0,self.yolo1,self.yolo2]


	def forward(self, x, targets=None, img_scores = None,gt_mix_index = None):


		img_dim = x.shape[2]
		loss = 0
		yolo_outputs = []

		x = self.layer0(x)
		x = self.layer1(x)
		x = self.attention(x)
		x_1 = self.layer2(x)
		x_2 = self.layer3(x_1)
		x_3 = self.layer4(x_2)
		x = self.compress(x_3)
		x = F.interpolate(x,scale_factor=2,mode = 'bilinear',align_corners=True)

		x = torch.cat([x,x_2],dim=1)
		x_4 = self.layer5(x)
		x = self.compress2(x_4)
		x = F.interpolate(x,scale_factor=2,mode = 'bilinear',align_corners=True)
		x = torch.cat([x,x_1],dim=1)

		x_scale_4 = self.scale_4(x)
		x_scale_2 = self.scale_2(x_4)
		x_scale_1 = self.scale_1(x_3)


		layer_0_x, layer_loss = self.yolo0(x_scale_1,targets,img_dim,img_scores = img_scores,gt_mix_index=gt_mix_index)
		loss += layer_loss
		yolo_outputs.append(layer_0_x)
		layer_1_x, layer_loss = self.yolo1(x_scale_2,targets,img_dim,img_scores=img_scores,gt_mix_index=gt_mix_index)
		loss += layer_loss
		yolo_outputs.append(layer_1_x)
		layer_2_x, layer_loss = self.yolo2(x_scale_4,targets,img_dim,img_scores=img_scores,gt_mix_index=gt_mix_index)
		loss += layer_loss
		yolo_outputs.append(layer_2_x)

		yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
		return yolo_outputs if targets is None else (loss, yolo_outputs)


class YoloNano(nn.Module):
	def __init__(self, num_class = 20, num_anchor = 3, img_size = 416):

		super(YoloNano, self).__init__()
		self.num_classes = num_class
		self.image_size = img_size
		self.num_anchors = 3
		self.yolo_channels = (self.num_classes + 5) * self.num_anchors
		
		anchors52 = [[10,13], [16,30], [33,23]] # 52x52
		anchors26 = [[30,61], [62,45], [59,119]] # 26x26
		anchors13 = [[116,90], [156,198], [373,326]] # 13x13
		
		# image:  416x416x3
		self.conv1 = conv3x3(3, 12, stride=1) # output: 416x416x12
		self.conv2 = conv3x3(12, 24, stride=2) # output: 208x208x24
		self.pep1 = PEP(24, 24, 7, stride=1) # output: 208x208x24
		self.ep1 = EP(24, 70, stride=2) # output: 104x104x70
		self.pep2 = PEP(70, 70, 25, stride=1) # output: 104x104x70
		self.pep3 = PEP(70, 70, 24, stride=1) # output: 104x104x70
		self.ep2 = EP(70, 150, stride=2) # output: 52x52x150
		self.pep4 = PEP(150, 150, 56, stride=1) # output: 52x52x150
		self.conv3 = conv1x1(150, 150, stride=1) # output: 52x52x150
		self.fca1 = FCA(150, 8) # output: 52x52x150
		self.pep5 = PEP(150, 150, 73, stride=1) # output: 52x52x150
		self.pep6 = PEP(150, 150, 71, stride=1) # output: 52x52x150
		
		self.pep7 = PEP(150, 150, 75, stride=1) # output: 52x52x150
		self.ep3 = EP(150, 325, stride=2) # output: 26x26x325
		self.pep8 = PEP(325, 325, 132, stride=1) # output: 26x26x325
		self.pep9 = PEP(325, 325, 124, stride=1) # output: 26x26x325
		self.pep10 = PEP(325, 325, 141, stride=1) # output: 26x26x325
		self.pep11 = PEP(325, 325, 140, stride=1) # output: 26x26x325
		self.pep12 = PEP(325, 325, 137, stride=1) # output: 26x26x325
		self.pep13 = PEP(325, 325, 135, stride=1) # output: 26x26x325
		self.pep14 = PEP(325, 325, 133, stride=1) # output: 26x26x325
		
		self.pep15 = PEP(325, 325, 140, stride=1) # output: 26x26x325
		self.ep4 = EP(325, 545, stride=2) # output: 13x13x545
		self.pep16 = PEP(545, 545, 276, stride=1) # output: 13x13x545
		self.conv4 = conv1x1(545, 230, stride=1) # output: 13x13x230
		self.ep5 = EP(230, 489, stride=1) # output: 13x13x489
		self.pep17 = PEP(489, 469, 213, stride=1) # output: 13x13x469
		
		self.conv5 = conv1x1(469, 189, stride=1) # output: 13x13x189
		self.conv6 = conv1x1(189, 105, stride=1) # output: 13x13x105
		# upsampling conv6 to 26x26x105
		# concatenating [conv6, pep15] -> pep18 (26x26x430)
		self.pep18 = PEP(430, 325, 113, stride=1) # output: 26x26x325
		self.pep19 = PEP(325, 207, 99, stride=1) # output: 26x26x325
		
		self.conv7 = conv1x1(207, 98, stride=1) # output: 26x26x98
		self.conv8 = conv1x1(98, 47, stride=1) # output: 26x26x47
		# upsampling conv8 to 52x52x47
		# concatenating [conv8, pep7] -> pep20 (52x52x197)
		self.pep20 = PEP(197, 122, 58, stride=1) # output: 52x52x122
		self.pep21 = PEP(122, 87, 52, stride=1) # output: 52x52x87
		self.pep22 = PEP(87, 93, 47, stride=1) # output: 52x52x93
		self.conv9 = conv1x1(93, self.yolo_channels, stride=1, bn=False) # output: 52x52x yolo_channels
		self.yolo_layer52 = YOLOLayer(anchors52, self.num_classes, img_dim=self.image_size)

		# conv7 -> ep6
		self.ep6 = EP(98, 183, stride=1) # output: 26x26x183
		self.conv10 = conv1x1(183, self.yolo_channels, stride=1, bn=False) # output: 26x26x yolo_channels
		self.yolo_layer26 = YOLOLayer(anchors26, self.num_classes, img_dim=self.image_size)

		# conv5 -> ep7
		self.ep7 = EP(189, 462, stride=1) # output: 13x13x462
		self.conv11 = conv1x1(462, self.yolo_channels, stride=1, bn=False) # output: 13x13x yolo_channels
		self.yolo_layer13 = YOLOLayer(anchors13, self.num_classes, img_dim=self.image_size)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				m.weight = nn.init.xavier_normal_(m.weight, gain=0.02)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.normal_(m.weight.data, 1.0, 0.02)
				m.bias.data.zero_()

	def forward(self, x, targets=None, img_scores = None,gt_mix_index = None):
		loss = 0
		yolo_outputs = []
		image_size = x.size(2)

		out = self.conv1(x)
		out = self.conv2(out)
		out = self.pep1(out)
		out = self.ep1(out)
		out = self.pep2(out)
		out = self.pep3(out)
		out = self.ep2(out)
		out = self.pep4(out)
		out = self.conv3(out)
		out = self.fca1(out)
		out = self.pep5(out)
		out = self.pep6(out)
		
		out_pep7 = self.pep7(out)
		out = self.ep3(out_pep7)
		out = self.pep8(out)
		out = self.pep9(out)
		out = self.pep10(out)
		out = self.pep11(out)
		out = self.pep12(out)
		out = self.pep13(out)
		out = self.pep14(out)

		out_pep15 = self.pep15(out)
		out = self.ep4(out_pep15)
		out = self.pep16(out)
		out = self.conv4(out)
		out = self.ep5(out)
		out = self.pep17(out)

		out_conv5 = self.conv5(out)
		out = F.interpolate(self.conv6(out_conv5), scale_factor=2)
		out = torch.cat([out, out_pep15], dim=1)
		out = self.pep18(out)
		out = self.pep19(out)
		
		out_conv7 = self.conv7(out)
		out = F.interpolate(self.conv8(out_conv7), scale_factor=2)
		out = torch.cat([out, out_pep7], dim=1)
		out = self.pep20(out)
		out = self.pep21(out)
		out = self.pep22(out)
		out_conv9 = self.conv9(out)
		#import pdb; pdb.set_trace()
		temp, layer_loss = self.yolo_layer52(out_conv9, targets, image_size)
		loss += layer_loss
		yolo_outputs.append(temp)

		out = self.ep6(out_conv7)
		out_conv10 = self.conv10(out)
		temp, layer_loss = self.yolo_layer26(out_conv10, targets, image_size)
		loss += layer_loss
		yolo_outputs.append(temp)

		out = self.ep7(out_conv5)
		out_conv11 = self.conv11(out)
		temp, layer_loss = self.yolo_layer13(out_conv11, targets, image_size)
		loss += layer_loss
		yolo_outputs.append(temp)

		yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))

		return yolo_outputs if targets is None else (loss, yolo_outputs)


if __name__ == '__main__':
	x = torch.randn(1,3,416,416)
	backbone = YoloNano()
	out = backbone(x)
	backbone.state_dict()
	torch.save(backbone.state_dict(),"xixi_a.pth")


