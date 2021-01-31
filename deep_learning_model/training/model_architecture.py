import torch
import torch.nn as nn
import torchvision


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d(2))
        self.drop_out = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(256, 64)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
#         out = self.fc1(out)
        return out


class SoccerNet(nn.Module):
    def __init__(self):
        super(SoccerNet, self).__init__()
        self.stream_body = torchvision.models.resnet50(pretrained=False)
        try:
            pretrain_dict = torch.load(
                "deep_learning_model/trained_model/resnet50_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth")
        except BaseException:
            pretrain_dict = torch.load(
                "../trained_model/resnet50_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth")
        model_dict = self.stream_body.state_dict()
        pretrain_dict = {
            k: v
            for k, v in pretrain_dict.items()
            if k in model_dict and model_dict[k].size() == v.size()
        }
        model_dict.update(pretrain_dict)
        self.stream_body.load_state_dict(model_dict)

        # self.stream_body = torchvision.models.resnet34(pretrained=True) #default resnet
        # self.stream_body.fc = nn.Dropout(0.5)

        self.stream_head = ConvNet()
        self.stream_torso = ConvNet()
        self.stream_legs = ConvNet()

        # self.linear = nn.Sequential(nn.Linear(1280,512), nn.ReLU(),
        # nn.Linear(512,128), nn.ReLU(), nn.Linear(128,25)) #default resnet
        self.linear = nn.Sequential(
            nn.Linear(
                1768, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(
                512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(
                128, 25))

    def forward(self, x):
        h = x.shape[2]
        body = x
        head = x[:, :, :h // 3, :]
        torso = x[:, :, h // 3:2 * h // 3, :]
        legs = x[:, :, 2 * h // 3:h, :]
        out_body = self.stream_body(body)
        out_head = self.stream_head(head)
        out_torso = self.stream_torso(torso)
        out_legs = self.stream_legs(legs)
        linear_input = torch.cat(
            (out_body, out_head, out_torso, out_legs), dim=1)
        out = self.linear(linear_input)
        return out


class SoccerNet_category_id(nn.Module):
    def __init__(self):
        super(SoccerNet_category_id, self).__init__()
        self.stream_body = ConvNet()
        self.linear = nn.Sequential(
            nn.Linear(
                256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(
                128, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(
                32, 5))

    def forward(self, x):
        body = x
        out_body = self.stream_body(body)
        out = self.linear(out_body)
        return out


class SoccerNet_player_id(nn.Module):
    def __init__(self):
        super(SoccerNet_player_id, self).__init__()
        self.stream_body = torchvision.models.resnet50(pretrained=False)
        try:
            pretrain_dict = torch.load(
                "deep_learning_model/trained_model/resnet50_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth")
        except BaseException:
            pretrain_dict = torch.load(
                "../trained_model/resnet50_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth")
        model_dict = self.stream_body.state_dict()
        pretrain_dict = {
            k: v
            for k, v in pretrain_dict.items()
            if k in model_dict and model_dict[k].size() == v.size()
        }
        model_dict.update(pretrain_dict)
        self.stream_body.load_state_dict(model_dict)

        self.stream_head = ConvNet()
        self.stream_torso = ConvNet()
        self.stream_legs = ConvNet()

        self.linear = nn.Sequential(
            nn.Linear(
                1768, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(
                512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(
                128, 11))

    def forward(self, x):
        h = x.shape[2]
        body = x
        head = x[:, :, :h // 3, :]
        torso = x[:, :, h // 3:2 * h // 3, :]
        legs = x[:, :, 2 * h // 3:h, :]
        out_body = self.stream_body(body)
        out_head = self.stream_head(head)
        out_torso = self.stream_torso(torso)
        out_legs = self.stream_legs(legs)
        linear_input = torch.cat(
            (out_body, out_head, out_torso, out_legs), dim=1)
        out = self.linear(linear_input)
        return out
