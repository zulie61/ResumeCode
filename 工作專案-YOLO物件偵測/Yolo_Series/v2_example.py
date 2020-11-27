# %%
import imgaug.augmenters as iaa
from yolov2 import Yolo
from utils.tools import get_class_weight
# %%
label_class=["raccoon"]
yolo = Yolo(label_class=label_class)

# %%
label_class = ["A_Liang"]
yolo = Yolo(label_class=label_class,
    anchors=[[0.16864347, 0.42071678],
            [0.18529212, 0.47019412],
            [0.11586914, 0.25918466],
            [0.15172164, 0.37613554],
            [0.13290392, 0.32469645]])

# %%
label_class = [i for i in range(10)]
label_class = list(map(str, label_class))
yolo = Yolo(label_class=label_class)

# %%
weight_path = "../ODT_weights/A_liang_20201019_perfect.h5"
yolo.create_model(
    # pretrained_weights=weight_path
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
data, label = yolo.read_file(
    img_path,
    label_path,
    shuffle=False)
# %%
data_path = "ex_data/json"
data, label = yolo.read_file(data_path,
                             label_format="labelme")

# %%
for i in range(len(data)):
    yolo.vis_img(data[i], label[i])

# %%
binary_weight = get_class_weight(
    label[:, :, :, 4:5],
    method='binary',
    )
print(binary_weight)
# %%
from tensorflow.keras.optimizers import SGD, Adam

yolo.model.compile(optimizer = Adam(lr=1e-4),
                   #optimizer = SGD(lr=1e-10, momentum=0.9, decay=5e-4),
                   loss = yolo.loss(
                       1,
                       loss_weight={"xy":1,
                                    "wh":1,
                                    "conf":5,
                                    "pr":1}),
                   metrics = [yolo.metrics("obj"),
                              yolo.metrics("iou"),
                              yolo.metrics("class")]
                   )
# %%
train_history = yolo.model.fit(data,
                               label,
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
        data[i], prediction[i],
        use_nms=True, nms_threshold=0.0)
# %%
