# Upper Bound Tracking

## Installation
A dedicated conda environment is recommended. You can set it up as follows:
```
sudo apt install g++ wget cmake git protobuf-compiler libsqlite3-dev -y
git clone https://github.com/dolokov/upper_bound_tracking && cd upper_bound_tracking
conda create --name ubt python=3.7 && conda activate ubt
conda install pip
pip install .
# if framework yolox used:
conda install cudatoolkit # hotfix for tf bug https://github.com/tensorflow/tensorflow/issues/45930
cd src/ubt/object_detection/YOLOX && python setup.py install
```

## Getting Started
### How to integrate UBT
```
from ubt.tracking.ocsort import ubocsort
upper_bound = 4
tracker = ubocsort.UpperBoundOCSort(upper_bound)
while frame is not None:
  _, frame = video_reader.read()
  detections = object_detector.detect(frame) # detections is np.array and has shape [N, 5], each entry [x, y, w, h, score]
  tracks = tracker.update(detections, frame.shape) # tracks are [M, 5], each entry [track_id, x, y, w, h]
```

### How use UBT for your data
<details>
  <summary>Create Project</summary>
 
First create a new project. A project has a name and a set of keypoint names. It can contain multiple videos.
```
python3.7 -m ubt.be.project -name MiceTop -manager MyName -keypoint_names nose,tail_base,left_ear,right_ear,left_front_paw,right_front_paw,left_back_paw,right_back_paw 
```
Note, that the keypoint detection uses horizontal and vertical flipping for data augmentation while training, which might violate some label maps. This is automatically fixed by dynamically switching labels of pairs of classes that are simliar expect left and right in the name. (e.g. `left_ear` and `right_ear` are switched, `l_ear` and `r_ear` are not).

</details>

<details>
  <summary>Add Video</summary>
  
  Then add a video to your project with ID 1. It will write every frame of the video to your local disk for later annotation.
```
python3.7 -m ubt.be.video -add_project 1 -add_video /path/to/video.mp4
```
  
</details>

<details>
  <summary>Label Frames</summary>
  
Fixed Multitracker tracks objects and keypoints and therefore offers two annotation tools for drawing bounding boxes and setting predefined, project dependent keypoints.
``` 
python3.7 -m ubt.app.app
```
Go to the url `http://localhost:8888/home`. You should see a list of your projects and videos. You then can start each annotation tool with a link for the specific tool and video you want annotate. Please note that you should have an equal number of labeled images for both tasks. We recommend to annotate at least 150 frames, but the more samples the better the detections.
  
 </details>
<details>
  <summary>Train Object Detector</summary>
  
  Prepare the training dataset.
``` 
  python3 -m ubt.object_detection.cvt2VOC --train_video_ids 1,2 --test_video_ids 3 --database /path/to/data.db --outdir /path/to/data_dir
``` 
  
Prepare the training configuration. We recommend YoloX-S or YoloX-M. Update your `data_dir` in `yolox_voc_s.py`.

  Download the official pretrained model from [the official YoloX repo](https://github.com/Megvii-BaseDetection/YOLOX#standard-models).
  
Then training can be started. The batchsize (-b) should be chosen based on the available GPU memory for training. 
```  
upper_bound_tracking/src/ubt/object_detection/YOLOX$ python -m tools.train -f exps/example/yolox_voc/yolox_voc_s.py -d 1 -b 16 --fp16 -c /path/to/pretrained/yolox_s.pth
```
</details>

<details>
  <summary>Run Tracking</summary>

  Now you can call the actual tracking algorithm. If not provided with pretrained models for object detection and keypoint models, it will train those based on annotations of the supplied video ids.
```
python3.7 -m ubt.tracking --video /path/to/target_video.mp4 --objectdetection_model /path/to/yolox_checkpoint_dir --upper_bound 4
  ```
</details>

<details>
  <summary>Run Tracking with Keypoints</summary>
  
  You also can use a combined MOT/Keypoint tracking approach. If you have created a project inside UBT and have chosen a keypoint method, it will start training the keypoint model prior to tracking. Alternatively you can use a pretrained keypoint detection network. 
```
python3.7 -m ubt.tracking --video /path/to/target_video.mp4 --objectdetection_model /path/to/yolox_checkpoint_dir --project 1 --keypoint_model /path/to/keypoint.h5 --keypoint_method psp --upper_bound 4
```
  
</details>

### How to evaluate on Mouse Data
1) download labeled bounding box and tracking data [here](https://drive.google.com/file/d/1ABpsj560McDTLVuvpNGqwfMqKiEczUyC/view?usp=share_link)
2) import data `python3 -m ubt.be.migrate --mode import --zip /path/to/labeled_data.zip`
3) download pretrained detection network [here](https://drive.google.com/file/d/1O-cA7tCmOVY8v-K8EtWECDb6mJsks-QB/view?usp=sharing)
4) download videos 
5) prepare evaluation [plan](https://github.com/dolokov/upper_bound_tracking/blob/main/src/ubt/TrackEval/configs/ubocsort.json)
6) [evaluate](https://github.com/dolokov/upper_bound_tracking/blob/main/src/ubt/TrackEval/scripts/run_multitracker.py) against GT tracking data

## Motivation
Multiple Object Tracking (MOT) is defined in an open world: for each frame it is unknown how many objects are currently observed, and therefore can be visible at the same time. Also the total number of objects seen in the complete video is unknown.

In this specific use case, requirements differ slightly. Animals are filmed in enclosured environments in fixed and known numbers. These biological studies expect a limited set of very long tracks, each track corresponding to the complete movement of one animal in parts of the video. Tracking algorithms will produce fragmented tracks that have to be merged manually after the tracking process.

This leads to the definition of Upper Bound Tracking, that tries to track multiple animals with a known upper bound u ∈ N of the video v as the maximum number of indivudual animals filmed at the same time. Therefore a new tracking algorithm was developed to improve fragmentation that exploits the upper bound of videos, called Upper Bound Tracker (UBT). It needs, besides the RGB video stream, the upper bound u ∈ N. It is inspired by the V-IoU tracker and extended by careful consideration before creating new tracks to never have more than u tracks at the same time. Additionally a novel reidentification step is introduced, that updates an inactive and probably lost track to the position of a new detection if u tracks are already present. By introducing the upper bound u, tracking algorithms can exploit the provided additional knowledge to improve matching or reidentification.

## Arguments
There are several options for object detection, keypoint estimation and tracking. Each combination might give different results and can be easily changed.

`--project_id` ID of the project. Each project has a unique label map.

`--video` path of the MP4 video that should be tracked.

`--train_video_ids` list of video ids that are trained on (eg 1,2,3)

`--test_video_ids` list of video ids that are tested on (eg 3,4)

`--data_dir` directory to save all data and database. defaults to ~/data/multitracker

`--tracking_method` options: DeepSORT, VIoU, UpperBound

`--upper_bound` upper bound number of animals observed

`--keypoint_method` options: none, hourglass2, hourglass4, hourglass8, vgg16, efficientnet, efficientnetLarge, psp. defaults to hourglass2. option none tracks objects without keypoints.

`--keypoint_resolution` resolution used in keypoint detection. defaults to 224x224

`--track_tail` length of drawn tail for all animals in visualization

`--delete_all_checkpoints` delete all checkpoints from directory ~/checkpoints/multitracker

Each predicted bounding box and keypoint comes with its own confidence score indicating how sure the algorithm is the object or keypoint to actually be there. We filter these predictions based on two thresholds, that can be changed:

`--min_confidence_boxes` minimum confidence for an detected animal bounding box, defaults to 0.5

`--min_confidence_keypoints` minimum confidence for an detected limb keypoint, defaults to 0.5
