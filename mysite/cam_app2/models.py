# cam_app2/models.py

import os
import uuid

from django.conf import settings
from django.db import models
from django.shortcuts import render

from wagtail.core.models import Page
from wagtail.admin.edit_handlers import FieldPanel, MultiFieldPanel

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

# ─── GLOBAL SETUP ────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSIFY_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

SEG_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# ─── CLASSIFIER LOADER ───────────────────────────────────────

_GLOBAL_CLS_MODEL = None

def get_model():
    global _GLOBAL_CLS_MODEL
    if _GLOBAL_CLS_MODEL is None:
        base = tv_models.resnet18(pretrained=False)
        for p in base.parameters():
            p.requires_grad = False
        num_ftrs = base.fc.in_features

        class MTR(nn.Module):
            def __init__(self):
                super().__init__()
                self.base_model     = nn.Sequential(*list(base.children())[:-1])
                self.shared_fc      = nn.Sequential(
                    nn.Linear(num_ftrs, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                )
                self.freshness_head = nn.Linear(256, 3)
                self.produce_head   = nn.Linear(256, 4)

            def forward(self, x):
                x = self.base_model(x).view(x.size(0), -1)
                x = self.shared_fc(x)
                return self.freshness_head(x), self.produce_head(x)

        model = MTR().to(DEVICE)
        model.load_state_dict(torch.load(
            os.path.join(settings.BASE_DIR, "food_multi_task_model.pth"),
            map_location=DEVICE
        ))
        model.eval()
        _GLOBAL_CLS_MODEL = model

    return _GLOBAL_CLS_MODEL

# ─── SEGMENTATION LOADER ────────────────────────────────────

_SEG_MODEL = None

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        def block(i,o):
            return nn.Sequential(
                nn.Conv2d(i,o,3,padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(o,o,3,padding=1), nn.ReLU(inplace=True),
            )
        self.encoder1 = block(3,64)
        self.pool1    = nn.MaxPool2d(2)
        self.encoder2 = block(64,128)
        self.pool2    = nn.MaxPool2d(2)
        self.middle   = block(128,256)
        self.up1      = nn.ConvTranspose2d(256,128,2,stride=2)
        self.decoder1 = block(256,128)
        self.up2      = nn.ConvTranspose2d(128,64,2,stride=2)
        self.decoder2 = block(128,64)
        self.out      = nn.Conv2d(64,1,1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        m  = self.middle(self.pool2(e2))
        d1 = self.decoder1(torch.cat([self.up1(m), e2], dim=1))
        d2 = self.decoder2(torch.cat([self.up2(d1), e1], dim=1))
        return torch.sigmoid(self.out(d2))


def get_seg_model():
    global _SEG_MODEL
    if _SEG_MODEL is None:
        seg = UNet().to(DEVICE)
        seg.load_state_dict(torch.load(
            os.path.join(settings.BASE_DIR, "unet_model.pth"),
            map_location=DEVICE
        ))
        seg.eval()
        _SEG_MODEL = seg
    return _SEG_MODEL

# ─── WAGTAIL PAGE ────────────────────────────────────────────

class ImagePage(Page):
    template   = "cam_app2/image.html"
    max_count  = 1

    name_title    = models.CharField(max_length=100, blank=True, default="")
    name_subtitle = models.TextField(blank=True, default="")

    content_panels = Page.content_panels + [
        MultiFieldPanel([
            FieldPanel("name_title"),
            FieldPanel("name_subtitle"),
        ], heading="Page Options"),
    ]

    def serve(self, request):
        context = {
            "my_uploaded_file_names": [],
            "my_result_file_names":   [],
            "my_advice":             [],
        }

        files = request.FILES.getlist("file_data")
        if not files:
            return render(request, self.template, context)

        cls_model = get_model()
        seg_model = get_seg_model()
        produce_labels = ['apple','banana','potato','carrot']

        for uploaded in files:
            # ── Save upload ──────────────────────────────
            uid   = uuid.uuid4().hex
            stem  = os.path.splitext(uploaded.name)[0]
            ext   = uploaded.name.split('.')[-1]
            fn    = f"{stem}_{uid}.{ext}"
            rel_up= os.path.join("uploadedPics", fn)
            abs_up= os.path.join(settings.MEDIA_ROOT, rel_up)
            os.makedirs(os.path.dirname(abs_up), exist_ok=True)
            with open(abs_up, "wb+") as dst:
                for chunk in uploaded.chunks():
                    dst.write(chunk)
            context["my_uploaded_file_names"].append(settings.MEDIA_URL + rel_up)

            pil_img = Image.open(abs_up).convert("RGB")

            # ── Defaults ────────────────────────────────
            produce, quality, advice = "Unknown", "Fresh", "Ready to eat 👍"

            # ── Classification ───────────────────────────
            x_cls = CLASSIFY_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                fres, prod = cls_model(x_cls)

                # produce label
                prod_p = torch.softmax(prod,1)[0].cpu().numpy()*100
                pi     = int(prod_p.argmax())
                produce= produce_labels[pi].title()

                # freshness probs
                f_p, sr_p, r_p = (torch.softmax(fres,1)[0].cpu().numpy()*100)

                # three-way only:
                if   r_p >= 70:
                    quality, advice = "Rotten",          "Not safe to eat... :("
                elif r_p >= 40:
                    quality, advice = "Slightly Rotten", "Don't eat the rotten parts!"
                else:
                    quality, advice = "Fresh",           "Safe to eat! :D"

            # ── Overlay text ───────────────────────────
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            cv2.putText(cv_img, f"{produce}: {quality}", (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(cv_img, advice, (10,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # ── Segmentation ────────────────────────────
            x_seg = SEG_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                mask_pred = seg_model(x_seg)[0,0].cpu().numpy()
                mask      = (mask_pred > 0.5).astype(np.uint8)

            mask = cv2.resize(mask,
                              (cv_img.shape[1], cv_img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
            overlay = cv_img.copy()
            overlay[mask==1] = (0,0,255)
            cv_img = cv2.addWeighted(overlay, 0.5, cv_img, 0.5, 0)

            # ── Save result ─────────────────────────────
            rel_res = os.path.join("Result", fn)
            abs_res = os.path.join(settings.MEDIA_ROOT, rel_res)
            os.makedirs(os.path.dirname(abs_res), exist_ok=True)
            cv2.imwrite(abs_res, cv_img)

            context["my_result_file_names"].append(settings.MEDIA_URL + rel_res)
            context["my_advice"].append(advice)

        # ── Must return here ─────────────────────────
        return render(request, self.template, context)
