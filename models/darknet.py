from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils.parse_config import *
from utils.utils import build_targets, to_cpu, non_max_suppression

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_modules(module_defs):
	"""
	Constructs module list of layer blocks from module configuration in module_defs
	"""
	hyperparams = module_defs.pop(0)
	output_filters = [int(hyperparams["channels"])]
	module_list = nn.ModuleList()
	for module_i, module_def in enumerate(module_defs):
		modules = nn.Sequential()

		if module_def["type"] == "convolutional":
			bn = int(module_def["batch_normalize"])
			filters = int(module_def["filters"])
			kernel_size = int(module_def["size"])
			pad = (kernel_size - 1) // 2
			modules.add_module(
				f"conv_{module_i}",
				nn.Conv2d(
					in_channels=output_filters[-1],
					out_channels=filters,
					kernel_size=kernel_size,
					stride=int(module_def["stride"]),
					padding=pad,
					bias=not bn,
				),
			)
			if bn:
				modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
			if module_def["activation"] == "leaky":
				modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

		elif module_def["type"] == "maxpool":
			kernel_size = int(module_def["size"])
			stride = int(module_def["stride"])
			if kernel_size == 2 and stride == 1:
				modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
			maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
			modules.add_module(f"maxpool_{module_i}", maxpool)

		elif module_def["type"] == "upsample":
			upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
			modules.add_module(f"upsample_{module_i}", upsample)

		elif module_def["type"] == "route":
			layers = [int(x) for x in module_def["layers"].split(",")]
			filters = sum([output_filters[1:][i] for i in layers])
			modules.add_module(f"route_{module_i}", EmptyLayer())

		elif module_def["type"] == "shortcut":
			filters = output_filters[1:][int(module_def["from"])]
			modules.add_module(f"shortcut_{module_i}", EmptyLayer())

		elif module_def["type"] == "yolo":

			anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
			# Extract anchors
			anchors = [int(x) for x in module_def["anchors"].split(",")]


			anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]



			anchors = [anchors[i] for i in anchor_idxs]
			num_classes = int(module_def["classes"])
			img_size = int(hyperparams["height"])

			# Define detection layer
			yolo_layer = YOLOLayer(anchors, num_classes, img_size)


			modules.add_module(f"yolo_{module_i}", yolo_layer)
		# Register module list and number of output filters
		module_list.append(modules)
		output_filters.append(filters)

	return hyperparams, module_list


class Upsample(nn.Module):
	""" nn.Upsample is deprecated """

	def __init__(self, scale_factor, mode="nearest"):
		super(Upsample, self).__init__()
		self.scale_factor = scale_factor
		self.mode = mode

	def forward(self, x):
		x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
		return x


class EmptyLayer(nn.Module):
	"""Placeholder for 'route' and 'shortcut' layers"""

	def __init__(self):
		super(EmptyLayer, self).__init__()


class YOLOLayer2(nn.Module):
	"""Detection layer"""

	def __init__(self, anchors, num_classes, img_dim=416):
		super(YOLOLayer, self).__init__()
		self.anchors = anchors
		self.num_anchors = len(anchors)
		self.num_classes = num_classes
		self.ignore_thres = 0.5
		self.mse_loss = nn.MSELoss(reduction='none')
		self.bce_loss = nn.BCELoss(reduction='none')
		self.obj_scale = 1
		self.noobj_scale = 100
		self.metrics = {}
		self.img_dim = img_dim
		self.grid_size = 0	# grid size

	def compute_grid_offsets(self, grid_size, cuda=True):
		self.grid_size = grid_size
		g = self.grid_size
		FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
		self.stride = self.img_dim / self.grid_size
		# Calculate offsets for each grid
		self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
		self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
		self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
		self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
		self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

	def forward(self, x, targets=None, img_dim=None,img_scores=None,gt_mix_index = None):

		# Tensors for cuda support
		FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
		LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
		ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

		self.img_dim = img_dim
		num_samples = x.size(0)
		grid_size = x.size(2)


		prediction = (
			x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
			.permute(0, 1, 3, 4, 2)
			.contiguous()
		)

		# Get outputs
		x = torch.sigmoid(prediction[..., 0])  # Center x
		y = torch.sigmoid(prediction[..., 1])  # Center y
		w = prediction[..., 2]	# Width
		h = prediction[..., 3]	# Height
		pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
		pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

		# If grid size does not match current we compute new offsets
		if grid_size != self.grid_size:
			self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

		# Add offset and scale with anchors
		pred_boxes = FloatTensor(prediction[..., :4].shape)
		pred_boxes[..., 0] = x.data + self.grid_x
		pred_boxes[..., 1] = y.data + self.grid_y
		pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
		pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h


		output = torch.cat(
			(
				pred_boxes.view(num_samples, -1, 4) * self.stride,
				pred_conf.view(num_samples, -1, 1),
				pred_cls.view(num_samples, -1, self.num_classes),
			),
			-1,
		)
		if targets is None:
			return output, 0
		else:
			iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf,batches_weights,obj_mask_mix_index = build_targets(
				pred_boxes=pred_boxes,
				pred_cls=pred_cls,
				target=targets,
				anchors=self.scaled_anchors,
				ignore_thres=self.ignore_thres,
				img_scores =img_scores,
				gt_mix_index = gt_mix_index
			)

			# Loss : Mask outputs to ignore non-existing objects (except with conf. loss)

			sum_weights1,sum_weights2 = batches_weights
			obj_mask_1,obj_mask_2 = obj_mask_mix_index


			loss_x_1 = torch.mean(self.mse_loss(x[obj_mask_1], tx[obj_mask_1])*sum_weights1[obj_mask_1])
			loss_y_1 = torch.mean(self.mse_loss(y[obj_mask_1], ty[obj_mask_1])*sum_weights1[obj_mask_1])
			loss_w_1 = torch.mean(self.mse_loss(w[obj_mask_1], tw[obj_mask_1])*sum_weights1[obj_mask_1])
			loss_h_1 = torch.mean(self.mse_loss(h[obj_mask_1], th[obj_mask_1])*sum_weights1[obj_mask_1])
			if obj_mask_2 is not None:
				loss_x_2 = torch.mean(self.mse_loss(x[obj_mask_2], tx[obj_mask_2])*sum_weights2[obj_mask_2])
				loss_y_2 = torch.mean(self.mse_loss(y[obj_mask_2], ty[obj_mask_2])*sum_weights2[obj_mask_2])
				loss_w_2 = torch.mean(self.mse_loss(w[obj_mask_2], tw[obj_mask_2])*sum_weights2[obj_mask_2])
				loss_h_2 = torch.mean(self.mse_loss(h[obj_mask_2], th[obj_mask_2])*sum_weights2[obj_mask_2])
			else:
				loss_x_2 = 0.
				loss_y_2 = 0.
				loss_w_2 = 0.
				loss_h_2 = 0.

			loss_x = loss_x_1+loss_x_2
			loss_y = loss_y_1+loss_y_2
			loss_w = loss_w_1+loss_w_2
			loss_h = loss_h_1+loss_h_2



			loss_conf_obj_1 = torch.mean(self.bce_loss(pred_conf[obj_mask_1], tconf[obj_mask_1])*sum_weights1[obj_mask_1])
			if obj_mask_2 is not None:
				loss_conf_obj_2 = torch.mean(self.bce_loss(pred_conf[obj_mask_2], tconf[obj_mask_2])*sum_weights2[obj_mask_2])
			else:
				loss_conf_obj_2 = 0.
			loss_conf_obj = loss_conf_obj_1+loss_conf_obj_2

			loss_conf_noobj = torch.mean(self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask]))
			loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj

			#loss_cls_1 = torch.mean(self.bce_loss(pred_cls[obj_mask_1], tcls[obj_mask_1])*sum_weights1[obj_mask_1])
			loss_cls_1 = torch.mean(self.bce_loss(pred_cls[obj_mask_1], tcls[obj_mask_1]))
			if obj_mask_2 is not None:
				#loss_cls_2 = torch.mean(self.bce_loss(pred_cls[obj_mask_2], tcls[obj_mask_2])*sum_weights2[obj_mask_2])
				loss_cls_2 = torch.mean(self.bce_loss(pred_cls[obj_mask_2], tcls[obj_mask_2]))
			else:
				loss_cls_2 = 0.
			loss_cls = loss_cls_1+loss_cls_2
			total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

			# Metrics
			cls_acc = 100 * class_mask[obj_mask].mean()
			conf_obj = pred_conf[obj_mask].mean()
			conf_noobj = pred_conf[noobj_mask].mean()
			conf50 = (pred_conf > 0.5).float()
			iou50 = (iou_scores > 0.5).float()
			iou75 = (iou_scores > 0.75).float()
			detected_mask = conf50 * class_mask * tconf
			precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
			recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
			recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

			self.metrics = {
				"loss": to_cpu(total_loss).item(),
				"x": to_cpu(loss_x).item(),
				"y": to_cpu(loss_y).item(),
				"w": to_cpu(loss_w).item(),
				"h": to_cpu(loss_h).item(),
				"conf": to_cpu(loss_conf).item(),
				"cls": to_cpu(loss_cls).item(),
				"cls_acc": to_cpu(cls_acc).item(),
				"recall50": to_cpu(recall50).item(),
				"recall75": to_cpu(recall75).item(),
				"precision": to_cpu(precision).item(),
				"conf_obj": to_cpu(conf_obj).item(),
				"conf_noobj": to_cpu(conf_noobj).item(),
				"grid_size": grid_size,
			}

			return output, total_loss


class YOLOLayer(nn.Module):
	# detection layer
	def __init__(self, anchors, num_classes, img_dim=416):
		super(YOLOLayer, self).__init__()
		self.anchors = anchors
		self.num_anchors = len(anchors)
		self.num_classes = num_classes
		self.ignore_thres = .5
		self.mse_loss = nn.MSELoss()
		self.bce_loss = nn.BCELoss()
		self.obj_scale = 1
		self.noobj_scale = 100
		self.metrics = {}
		self.img_dim = img_dim
		self.grid_size = 0

	def compute_grid_offsets(self, grid_size, cuda=True):
		self.grid_size = grid_size
		g = self.grid_size
		FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
		self.stride = self.img_dim / self.grid_size
		# Calculate offsets for each grid
		self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
		self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
		self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
		self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
		self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

	def forward(self, x, targets=None, img_dim=None):
		FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
		LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
		ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

		self.img_dim = img_dim

		num_samples = x.size(0)
		grid_size = x.size(2)

		prediction = (
			x.view(num_samples, self.num_anchors, self.num_classes+5, grid_size, grid_size)
			.permute(0,1,3,4,2)
			.contiguous()
		)

		x = torch.sigmoid(prediction[..., 0]) # center x
		y = torch.sigmoid(prediction[..., 1]) # center y
		w = prediction[..., 2] # width
		h = prediction[..., 3] # Height
		pred_conf = torch.sigmoid(prediction[..., 4]) # Conf
		pred_cls = torch.sigmoid(prediction[..., 5:]) # Cls Pred

		if grid_size != self.grid_size:
			self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

		# Add offset and scale with anchors
		pred_boxes = FloatTensor(prediction[..., :4].shape)
		pred_boxes[..., 0] = x.data + self.grid_x
		pred_boxes[..., 1] = y.data + self.grid_y
		pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
		pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h
		#print(pred_boxes.size())
		#print(pred_boxes.view(num_samples, -1, 4).size())

		output = torch.cat(
			(
				pred_boxes.view(num_samples, -1, 4) * self.stride,
				pred_conf.view(num_samples, -1, 1),
				pred_cls.view(num_samples, -1, self.num_classes),
			),
			-1,
		)

		if targets is None:
			return output, 0
		else:
			iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
				pred_boxes=pred_boxes,
				pred_cls=pred_cls,
				target=targets,
				anchors=self.scaled_anchors,
				ignore_thres=self.ignore_thres,
			)

			#obj_mask = obj_mask.to(torch.bool)
			#noobj_mask = noobj_mask.to(torch.bool)

			# Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
			loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
			loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
			loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
			loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
			#print(pred_conf[obj_mask], tconf[obj_mask])
			loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
			loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
			loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
			loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
			total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

			# Metrics
			cls_acc = 100 * class_mask[obj_mask].mean()
			conf_obj = pred_conf[obj_mask].mean()
			conf_noobj = pred_conf[noobj_mask].mean()
			conf50 = (pred_conf > 0.5).float()
			iou50 = (iou_scores > 0.5).float()
			iou75 = (iou_scores > 0.75).float()
			detected_mask = conf50 * class_mask * tconf
			precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
			recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
			recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

			self.metrics = {
				"loss": to_cpu(total_loss).item(),
				"x": to_cpu(loss_x).item(),
				"y": to_cpu(loss_y).item(),
				"w": to_cpu(loss_w).item(),
				"h": to_cpu(loss_h).item(),
				"conf": to_cpu(loss_conf).item(),
				"cls": to_cpu(loss_cls).item(),
				"cls_acc": to_cpu(cls_acc).item(),
				"recall50": to_cpu(recall50).item(),
				"recall75": to_cpu(recall75).item(),
				"precision": to_cpu(precision).item(),
				"conf_obj": to_cpu(conf_obj).item(),
				"conf_noobj": to_cpu(conf_noobj).item(),
				"grid_size": grid_size,
			}
		return output, total_loss


