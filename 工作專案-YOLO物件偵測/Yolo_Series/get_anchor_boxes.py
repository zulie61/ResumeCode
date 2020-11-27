# %%
from yolov1_5 import Yolo
from utils.kmeans import kmeans, iou_dist, euclidean_dist

# %%
label_class = [i for i in range(10)]
label_class = list(map(str, label_class))
yolo = Yolo(label_class=label_class)

# %%
data_path = "ex_data/json"
data, label = yolo.read_file(label_path=data_path,
                             label_format="labelme")

# %%
for i in range(len(data)):
    yolo.vis_img(data[i], label[i])

# %%
all_boxes = label[label[..., 4] == 1][..., 2:4]
anchor_boxes = kmeans(
    all_boxes,
    n_cluster=5,
    dist_func=iou_dist,
    stop_dist=0.01)
print(anchor_boxes)

# %%
anchor_boxes = kmeans(
    all_boxes,
    n_cluster=5,
    dist_func=euclidean_dist,
    stop_dist=0.01)
print(anchor_boxes)
# %%
