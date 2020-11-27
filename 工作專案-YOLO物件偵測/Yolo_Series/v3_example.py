# %%
import imgaug.augmenters as iaa
from yolov3 import Yolo
from utils.tools import get_class_weight

# %% use CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# %%
label_class=["raccoon"]
yolo = Yolo(label_class=label_class)

# %%
label_class = ["A_Liang"]

anchors=[[0.16864347, 0.42071678],
         [0.18529212, 0.47019412],
         [0.11586914, 0.25918466],
         [0.15172164, 0.37613554],
         [0.13290392, 0.32469645]]

yolo = Yolo(label_class=label_class, anchors=anchors)

# %%
label_class = [i for i in range(10)]
label_class = list(map(str, label_class))

anchors=[[0.1564507 , 0.15965215],
         [0.15097858, 0.13304265],
         [0.1371249 , 0.1284751 ],
         [0.11841729, 0.12273861],
         [0.10507934, 0.09904802],
         [0.08685152, 0.08555991],
         [0.06917795, 0.07387701],
         [0.05559911, 0.05470573],
         [0.03865267, 0.04000309]]

yolo = Yolo(label_class=label_class, anchors=anchors)

# %%
weight_path = "C:/Users/flag/OneDrive - 國立陽明大學/接案/血壓計影像判讀/master/weights/BPdigit_darkner53_yolov3.h5"
yolo.create_model(
    pretrained_weights=weight_path
    )
yolo.model.summary()

# %%
seq = iaa.Sequential([
    # iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect BBs
    iaa.Affine(
        # translate_px={"x": 40, "y": 60},
        scale=(0.6, 0.9),
        rotate=(-20, 20)
    ), # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
    # iaa.Crop(px=(0, 20)), # crop images from each side by 0 to 16px (randomly chosen)
    # iaa.Crop(px=(0, 100)),
    #iaa.Crop(percent=(0, 0.3)),
    # iaa.Flipud(0.5),
    iaa.Fliplr(0.5),
])

# %%
img_path = "ex_data/img"
label_path = "ex_data/label"
data, label = yolo.read_file(
    img_path,
    label_path,
    # augmenter=seq,
    # aug_times=5,
    shuffle=False)

# %%
img_path = "ex_data/A_liang/img"
label_path = "ex_data/A_liang/ann"
data, label3 = yolo.read_file(
    img_path,
    label_path,
    shuffle=False)

yolo.grid_num = 26
data, label2 = yolo.read_file(
    img_path,
    label_path,
    shuffle=False)

yolo.grid_num = 13
data, label1 = yolo.read_file(
    img_path,
    label_path,
    shuffle=False)

# %%
data_path = "ex_data/json"
yolo.grid_num = 52
data, label52 = yolo.read_file(data_path,
                               label_format="labelme",
                               shuffle=False)

yolo.grid_num = 26
data, label26 = yolo.read_file(data_path,
                               label_format="labelme",
                               shuffle=False)

yolo.grid_num = 13
data, label13 = yolo.read_file(data_path,
                               label_format="labelme",
                               shuffle=False)

# %%
for i in range(len(data)):
    yolo.vis_img(data[i], label52[i])

# %%
binary_weight = get_class_weight(
    label[:, :, :, 4:5],
    method='binary',
    )
print(binary_weight)
# %%
from tensorflow.keras.optimizers import SGD, Adam

loss_weight = {"xy":1,
               "wh":1,
               "conf":5,
               "pr":1}

yolo.grid_num = 13
loss13 = yolo.loss(0.2, loss_weight=loss_weight)
metric13 = [yolo.metrics("obj"),
            yolo.metrics("iou"),
            yolo.metrics("class")]

yolo.grid_num = 26
loss26 = yolo.loss(0.2, loss_weight=loss_weight)
metric26 = [yolo.metrics("obj"),
            yolo.metrics("iou"),
            yolo.metrics("class")]

yolo.grid_num = 52
loss52 = yolo.loss(0.2, loss_weight=loss_weight)
metric52 = [yolo.metrics("obj"),
            yolo.metrics("iou"),
            yolo.metrics("class")]

yolo.model.compile(optimizer = Adam(lr=1e-4),
                   #optimizer = SGD(lr=1e-10, momentum=0.9, decay=5e-4),
                   loss = [loss13, loss26, loss52],
                   metrics = [metric13, metric26, metric52]
                   )
# %%
train_history = yolo.model.fit(data,
                               [label13, label26, label52],
                               epochs=30,
                               batch_size=1,
                               verbose=1,
                               #validation_split=0.2,
                               )
# %%
prediction = yolo.model.predict(data)

# %%
for i in range(len(data)):
    yolo.vis_img(
        data[i],
        prediction[2][i],
        prediction[1][i],
        use_nms=True, nms_threshold=0.1,
        merge_threshold=0.1
        )
# %%