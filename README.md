# Upper Bound Tracking

sudo apt-get install libsqlite3-dev

python3 -m ubt.be.migrate --mode import --zip /path/to/labeled_detections.zip

## Installation
A dedicated conda environment is recommended. You can set it up as follows:
'''
sudo apt install g++ wget cmake git protobuf-compiler -y
git clone https://github.com/dolokov/upper_bound_tracking && cd upper_bound_tracking
conda create --name multitracker python=3.7 && conda activate multitracker
conda install pip
pip install .
conda install cudatoolkit # hotfix for tf bug https://github.com/tensorflow/tensorflow/issues/45930
cd src/ubt/object_detection/YOLOX && python setup.py install
'''

## Getting Started
### How to integrate UBT
'''

'''

## Motivation
Multiple Object Tracking (MOT) is defined in an open world: for each frame it is unknown how many objects are currently observed, and therefore can be visible at the same time. Also the total number of objects seen in the complete video is unknown.

In this specific use case, requirements differ slightly. Animals are filmed in enclosured environments in fixed and known numbers. These biological studies expect a limited set of very long tracks, each track corresponding to the complete movement of one animal in parts of the video. Tracking algorithms will produce fragmented tracks that have to be merged manually after the tracking process.

This leads to the definition of Upper Bound Tracking, that tries to track multiple animals with a known upper bound u ∈ N of the video v as the maximum number of indivudual animals filmed at the same time. Therefore a new tracking algorithm was developed to improve fragmentation that exploits the upper bound of videos, called Upper Bound Tracker (UBT). It needs, besides the RGB video stream, the upper bound u ∈ N. It is inspired by the V-IoU tracker and extended by careful consideration before creating new tracks to never have more than u tracks at the same time. Additionally a novel reidentification step is introduced, that updates an inactive and probably lost track to the position of a new detection if u tracks are already present. By introducing the upper bound u, tracking algorithms can exploit the provided additional knowledge to improve matching or reidentification.

