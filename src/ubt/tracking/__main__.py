
"""
    main program to track animals and their corresponding limbs on a video file

    
    python3.7 -m ubt.tracking --project_id 1 --train_video_ids 1 --test_video_ids 1 --upper_bound 4 --video /home/alex/data/ubt/projects/1/videos/2020-11-25_08-47-15_22772819_rec-00.00.00.000-00.10.20.916-seg1.avi --keypoint_method none --objectdetection_model /home/alex/github/upper_bound_tracking/src/ubt/object_detection/YOLOX/YOLOX_outputs/yolox_voc_m/last_epoch_ckpt.pth 
             --sketch_file /home/alex/data/ubt/projects/7/13/sketch.png 

    === 1) docker build ===
    $ cd upper_bound_tracking && sudo docker build -t ubt -f Dockerfile .
    $ chmod +x run_docker.sh
    

    == 2) docker start ==
    sudo ./run_docker.sh

    == 3) docker train ==

    create dataset
        root@9133fa063773:/home/alex/github/upper_bound_tracking/src# python -m ubt.object_detection.cvt2VOC --train_video_ids 1,3,4 --test_video_ids 4,6 --database /home/alex/data/ubt/data.db

    download pretrained network
        $ wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth -P /home/alex/data/ubt/object_detection

    start training
        root@9133fa063773:/home/alex/github/upper_bound_tracking/src/ubt/object_detection/YOLOX# python -m tools.train -f exps/example/yolox_voc/yolox_voc_s.py -d 1 -b 16 --fp16 -c /home/alex/data/ubt/object_detection/yolox_s.pth
    
    == 4) start tracking ==
    root@dd07d98dec1a:/home/alex/github/upper_bound_tracking/src# sh ubt/TrackEval/scripts/run_eval_tracking_viou.sh && sh ubt/TrackEval/scripts/run_eval_tracking_sort.sh && sh ubt/TrackEval/scripts/run_eval_tracking_upperbound.sh

    == 5) start evaluation
        python3.7 -m ubt.TrackEval.scripts.run_multitracker
        python3.7 ubt/TrackEval/scripts/plot_eval_results.py --csv /home/alex/data/ubt/evaluation/evaluation.csv --out /tmp/eval

    == close docker session ==> [CTRL] + [D]

    TODO: upper bound with less and more than real
    TODO: DLC
    TODO: ROI detections
    TODO: errors in eval GT_IDs
"""

from tqdm import tqdm 
import os
import numpy as np 

import subprocess
from glob import glob 
from random import shuffle 
import time 
from datetime import datetime
import cv2 as cv 
import h5py
import json 
import shutil
from collections import deque 

from natsort import natsorted
from ubt import util 
from ubt.keypoint_detection import model
from ubt.tracking import inference 

#import tensorflow as tf 
#tf.get_logger().setLevel('INFO')
#tf.get_logger().setLevel('ERROR')

from ubt import autoencoder

from ubt.tracking.viou import viou_tracker
from ubt.tracking import upperbound_tracker
from ubt.tracking.deep_sort.deep_sort import tracker as deepsort_tracker
from ubt.tracking import sort as sort_tracker
from ubt.tracking.ocsort import ocsort, ubocsort

from ubt.tracking.deep_sort import deep_sort_app
from ubt.tracking.keypoint_tracking import tracker as keypoint_tracking
from ubt.tracking.deep_sort.application_util import visualization

def train_yolox():
    raise NotImplementedError("You have to train a yolox model first!")

def main(args, showing = bool(1)):
    os.environ['UBT_DATA_DIR'] = args.data_dir
    from ubt.be import dbconnection
    
    if args.minutes>0 or args.video_resolution is not None:
        tpreprocessvideostart = time.time()
        sscale = args.video_resolution if args.video_resolution is not None else ''
        smins = str(args.minutes) if args.minutes>0 else ''
        fvo = args.video[:-4] + sscale + smins + args.video[-4:]
        if not os.path.isfile(fvo):
            commands = ['ffmpeg']
            if args.minutes>0:
                commands.extend(['-t',str(int(60.*args.minutes))])
            commands.extend(['-i',args.video])
            if args.video_resolution is not None:
                commands.extend(['-vf','scale=%s'%args.video_resolution.replace('x',':')])
            commands.extend([fvo])
            print('[*] preprocess video', ' '.join(commands))
            subprocess.call(commands)
            tpreprocessvideoend = time.time()
            print('[*] preprocessing of video to %s took %f seconds' % (fvo, tpreprocessvideoend-tpreprocessvideostart))
        args.video = fvo

    tstart = time.time()
    config = model.get_config(project_id = args.project_id, data_dir=args.data_dir)
    config['project_id'] = args.project_id
    config['video'] = args.video
    config['keypoint_model'] = args.keypoint_model
    config['keypoint_method'] = args.keypoint_method
    config['autoencoder_model'] = args.autoencoder_model 
    config['objectdetection_model'] = args.objectdetection_model
    config['output_video'] = args.output_video
    config['file_tracking_results'] = args.output_tracking_results
    config['train_video_ids'] = args.train_video_ids
    config['test_video_ids'] = args.test_video_ids
    config['minutes'] = args.minutes
    #config['upper_bound'] = db.get_video_fixednumber(args.video_id) 
    #config['upper_bound'] = None
    config['upper_bound'] = args.upper_bound
    config['n_blocks'] = 4
    config['tracking_method'] = args.tracking_method
    config['tracking_hyperparameters'] = args.tracking_hyperparameters
    config['track_tail'] = args.track_tail
    config['sketch_file'] = args.sketch_file
    config['use_all_data4train'] = args.use_all_data4train
    config['yolox_exp'] = args.yolox_exp
    config['yolox_name'] = args.yolox_name
    config['min_confidence_boxes'] = args.min_confidence_boxes
    
    config['object_detection_backbone'] = args.objectdetection_method
    config = model.update_config_object_detection(config)
    config['object_detection_resolution'] = [int(r) for r in args.objectdetection_resolution.split('x')]
    config['keypoint_resolution'] = [int(r) for r in args.keypoint_resolution.split('x')]
    config['img_height'], config['img_width'] = config['keypoint_resolution'][::-1]
    config['kp_backbone'] = args.keypoint_method
    if 'hourglass' in args.keypoint_method:
        config['kp_num_hourglass'] = int(args.keypoint_method[9:])
        config['kp_backbone'] = 'efficientnetLarge'
    
    if args.inference_objectdetection_batchsize > 0:
        config['inference_objectdetection_batchsize'] = args.inference_objectdetection_batchsize
    if args.inference_keypoint_batchsize > 0:
        config['inference_keypoint_batchsize'] = args.inference_keypoint_batchsize

    if args.delete_all_checkpoints:
        if os.path.isdir(os.path.expanduser('~/checkpoints/ubt')):
            shutil.rmtree(os.path.expanduser('~/checkpoints/ubt'))
    
    if args.data_dir:
        db = dbconnection.DatabaseConnection(file_db=os.path.join(args.data_dir,'data.db'))
        config['data_dir'] = args.data_dir 
        try:
            config['kp_data_dir'] = os.path.join(args.data_dir , 'projects/%i/data' % config['project_id'])
            config['kp_roi_dir'] = os.path.join(args.data_dir , 'projects/%i/data_roi' % config['project_id'])
        except:
            pass
        try:
            config['keypoint_names'] = db.get_keypoint_names(config['project_id'])
        except:
            config['keypoint_names'] = None 

        try:
            config['project_name'] = db.get_project_name(config['project_id'])
        except:
            config['project_name'] = None 

    # <train models>
    # 1) animal bounding box finetuning -> trains and inferences 
    if 'objectdetection_model' in config and config['objectdetection_model'] is not None:
        detection_model = inference.load_object_detector(config)

    else:
        config['objectdetection_max_steps'] = 30000
        # train object detector
        now = str(datetime.now()).replace(' ','_').replace(':','-').split('.')[0]
        checkpoint_directory_object_detection = os.path.expanduser('~/checkpoints/ubt/bbox/vids%s-%s' % (config['train_video_ids'], now))
        object_detect_restore = None 
        detection_model = None
        config['object_detection_model'] = checkpoint_directory_object_detection
        train_yolox()


    ## crop bbox detections and train keypoint estimation on extracted regions
    #point_classification.calculate_keypoints(config, detection_file_bboxes)
    
    # 2) train autoencoder for tracking appearence vector
    if config['autoencoder_model'] is None and config['tracking_method'] == 'DeepSORT':
        config_autoencoder = autoencoder.get_autoencoder_config()
        config_autoencoder['project_id'] = config['project_id']
        config_autoencoder['video_ids'] = natsorted(list(set([int(iid) for iid in config['train_video_ids'].split(',')]+[int(iid) for iid in config['test_video_ids'].split(',')])))
        config_autoencoder['project_name'] = config['project_name']
        config_autoencoder['data_dir'] = config['data_dir']
        config['autoencoder_model'] = autoencoder.train(config_autoencoder)
    print('[*] trained autoencoder model',config['autoencoder_model'])

    # 4) train keypoint estimator model
    if config['keypoint_model'] is None and not config['kp_backbone'] == 'none':
        config['kp_max_steps'] = 25000
        assert len(config['train_video_ids'])>0, "You must provide at least one video id for training ('--train_video_ids 1,2')!"
        assert len(config['test_video_ids'])>0, "You must provide at least one video id for testing (can be part of train, but part of test only recommended)  ('--test_video_ids 3,4')!"
        from ubt.keypoint_detection import roi_segm
        config['keypoint_model'] = roi_segm.train(config)
    print('[*] trained keypoint_model',config['keypoint_model'])
    # </train models>

    # load trained autoencoder model for Deep Sort Tracking 
    encoder_model = None 
    if config['tracking_method']=='DeepSORT':
        encoder_model = inference.load_autoencoder_feature_extractor(config)

    # load trained keypoint model
    if config['kp_backbone'] == 'none':
        keypoint_model = None
    else:
        keypoint_model = inference.load_keypoint_model(config['keypoint_model'])
    # </load models>

    
    #print(4*'\n',config)
    
    ttrack_start = time.time()
    
    output_video = None 
    if 'output_video' in config:
        output_video = config['output_video']
    output_video = run(config, detection_model, encoder_model, keypoint_model, args.min_confidence_boxes, args.min_confidence_keypoints, output_video=output_video , showing = showing )
    
    ttrack_end = time.time()
    if showing:
        tmpe = False
        if output_video[-4:] == '.mp4':
            nname = 'e.mp4'  
            tmpe = True 
        else:
            nname = '.mp4'
        video_file_out = output_video.replace('.%s'%output_video.split('.')[-1], nname)
        convert_video_h265(output_video, video_file_out)
        if tmpe:
            os.rename(video_file_out, video_file_out.replace('e.mp4','.mp4'))

        print('[*] done tracking after %f minutes. outputting file' % float(int((ttrack_end-ttrack_start)*10.)/10.),video_file_out)
    
def convert_video_h265(video_in, video_out):
    import subprocess 
    if os.path.isfile(video_out):
        os.remove(video_out)
    subprocess.call(['ffmpeg','-i',video_in, '-c:v','libx265','-preset','ultrafast',video_out])
    os.remove(video_in)


  
def run(config, detection_model, encoder_model, keypoint_model, min_confidence_boxes, min_confidence_keypoints, tracker = None, output_video = None, showing = True):
    if 'UpperBound' == config['tracking_method']:
        assert 'upper_bound' in config and config['upper_bound'] is not None and int(config['upper_bound'])>0, "ERROR: Upper Bound Tracking requires the argument --upper_bound to bet set (eg --upper_bound 4)"
    #config['upper_bound'] = None # ---> force VIOU tracker
    video_reader = cv.VideoCapture( config['video'] )
    
    Wframe  = int(video_reader.get(cv.CAP_PROP_FRAME_WIDTH))
    Hframe = int(video_reader.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    crop_dim = None
    total_frame_number = int(video_reader.get(cv.CAP_PROP_FRAME_COUNT))
    total_frame_number = 2000 # WARNING! just for eval
    fps = int(video_reader.get(cv.CAP_PROP_FPS))
    
    print('[*] total_frame_number',total_frame_number,'Hframe,Wframe',Hframe,Wframe,'fps',fps)
    
    if output_video is None:
        video_file_out = inference.get_video_output_filepath(config)
    else:
        video_file_out = output_video

    if config['file_tracking_results'] is None:
        config['file_tracking_results'] = video_file_out.replace('.%s'%video_file_out.split('.')[-1],'.csv')
    # setup CSV for object tracking and keypoints
    os.makedirs(os.path.split(config['file_tracking_results'])[0], exist_ok=True)
    if os.path.isfile(config['file_tracking_results']): os.remove(config['file_tracking_results'])

    print('[*] writing csv file to', config['file_tracking_results'])
    with open( config['file_tracking_results'], 'w') as ff:
        ff.write('video_id,frame_id,track_id,center_x,center_y,x1,y1,x2,y2,time_since_update\n')

    if 'keypoint_method' in config and not config['keypoint_method'] == 'none':
        path_csv_keypoints = config['file_tracking_results'].replace('.csv','_keypoints.csv')
        file_csv_keypoints = open( path_csv_keypoints, 'w') 
        file_csv_keypoints.write('video_id,frame_id,keypoint_class,keypoint_x,keypoint_y\n')
        print(f"[*] writing CSV with keypoints to {path_csv_keypoints}")
    # find out if video is part of the db and has video_id
    try:
        db.execute("select id from videos where name == '%s'" % config['video'].split('/')[-1])
        video_id = int([x for x in db.cur.fetchall()][0])
    except:
        video_id = -1
    print('      video_id',video_id)

    if showing:
        if os.path.isfile(video_file_out): os.remove(video_file_out)
        import skvideo.io
        video_writer = skvideo.io.FFmpegWriter(video_file_out, outputdict={
            '-vcodec': 'libx264',  #use the h.264 codec
            '-crf': '0',           #set the constant rate factor to 0, which is lossless
            '-preset':'veryslow'   #the slower the better compression, in princple, try 
                                    #other options see https://trac.ffmpeg.org/wiki/Encode/H.264
        }) 

        visualizer = visualization.Visualization([Wframe, Hframe], update_ms=5, config=config)
        print('[*] writing video file %s' % video_file_out)
        
    ## initialize tracker for boxes and keypoints
    args = None 
    if 'tracking_hyperparameters' in config and config['tracking_hyperparameters'] is not None:
        config['tracking_hyperparameters'] = os.path.expanduser(config['tracking_hyperparameters'])
        if os.path.isfile(config['tracking_hyperparameters']):
            with open(config['tracking_hyperparameters'],'r') as f:
                args = json.load(f)

    if config['tracking_method'] == 'UpperBound':
        tracker = upperbound_tracker.UpperBoundTracker(config)
    elif config['tracking_method'] == 'DeepSORT':
        tracker = deepsort_tracker.DeepSORTTracker(config)
    elif config['tracking_method'] == 'VIoU':
        tracker = viou_tracker.VIoUTracker(config)
    elif config['tracking_method'] == 'SORT':
        tracker = sort_tracker.Sort(config)
    elif config['tracking_method'] == 'Byte':
        if args is None:
            args = { # original authors values
                'track_thresh': 0.25,
                'track_buffer': 50,
                'mot20': False,
                'match_thresh': 0.3
            }
            if 0: args.update({'mot20': True})
            if 1: args.update({'track_thresh': 0.7})

        from collections import namedtuple
        args = namedtuple('byte_args', args.keys())(*args.values())
        from ubt.tracking.byte_tracker import byte_tracker
        tracker = byte_tracker.BYTETracker(args, frame_rate=fps)
    elif config['tracking_method'] == 'OCSORT':
        if args is None:
            args = {
                'det_thresh': 0.6, 
                'max_age': 30, 
                'min_hits': 3, 
                'iou_threshold':0.3, 
                'delta_t': 3, 
                'asso_func': "iou", 
                'inertia':0.2, 
                'use_byte': False
            }
        tracker = ocsort.OCSort(args['det_thresh'], max_age=args['max_age'], min_hits=args['min_hits'], 
            iou_threshold=args['iou_threshold'], delta_t=args['delta_t'], asso_func=args['asso_func'], 
            inertia=args['inertia'], use_byte=args['use_byte'])
    elif config['tracking_method'] == 'UpperBoundOCSORT':
        if args is None:
            args = {
                'det_thresh': 0.6, 
                'max_age': 30, 
                'min_hits': 3, 
                'iou_threshold':0.3, 
                'delta_t': 3, 
                'asso_func': "iou", 
                'inertia':0.2, 
                'use_byte': False,
                'upper_bound': int(config['upper_bound']),
                'maximum_nearest_reassign_track_distance': 1.2,
                'min_missed_steps_before_reassign': 8
            }
        args['maximum_nearest_reassign_track_distance'] *= min([Hframe,Wframe])

        tracker = ubocsort.UpperBoundOCSort(args['det_thresh'], args['upper_bound'], 
            maximum_nearest_reassign_track_distance = args['maximum_nearest_reassign_track_distance'],
            min_missed_steps_before_reassign = args['min_missed_steps_before_reassign'],
            max_age=args['max_age'], min_hits=args['min_hits'], 
            iou_threshold=args['iou_threshold'], delta_t=args['delta_t'], asso_func=args['asso_func'], 
            inertia=args['inertia'], use_byte=args['use_byte'])



    keypoint_tracker = keypoint_tracking.KeypointTracker()

    frame_idx = -1
    frame_buffer = deque()
    detection_buffer = deque()
    keypoint_buffer = deque()
    results = []
    running = True 
    scale = None 
    
    tbenchstart = time.time()
    # fill up initial frame buffer for batch inference
    for ib in range(config['inference_objectdetection_batchsize']-1):
        ret, frame = video_reader.read()
        #frame = cv.resize(frame,None,fx=0.5,fy=0.5)
        frame_buffer.append(frame[:,:,::-1]) # trained on TF RGB, cv2 yields BGR

    #while running: #video_reader.isOpened():
    for frame_idx in tqdm(range(total_frame_number)):
        #frame_idx += 1 
        config['count'] = frame_idx
        if frame_idx == 10:
            tbenchstart = time.time()

        # fill up frame buffer as you take something from it to reduce lag 
        timread0 = time.time()
        if video_reader.isOpened():
            ret, frame = video_reader.read()
            #frame = cv.resize(frame,None,fx=0.5,fy=0.5)
            if frame is not None:
                frame_buffer.append(frame[:,:,::-1]) # trained on TF RGB, cv2 yields BGR
            else:
                running = False
                #file_csv.close()
                if 'keypoint_method' in config and not config['keypoint_method'] == 'none':
                    file_csv_keypoints.close()
                #return True  
        else:
            running = False 
            #file_csv.close()
            if 'keypoint_method' in config and not config['keypoint_method'] == 'none':
                file_csv_keypoints.close()
            #return True 
        timread1 = time.time()
        # frame_idx % 1000 == 0

        if running:
            tobdet0 = time.time()
            
            ## object detection
            frames_tensor = np.array([frame]).astype(np.float32)
            detections = inference.detect_batch_bounding_boxes(config, detection_model, frames_tensor, min_confidence_boxes, encoder_model = encoder_model)[0]
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            features = np.array([d.feature for d in detections])
            if 'roi' in config and config['roi'] is not None: ## filter out detections inside certain rectangle to test large scale occlusion
                xb,yb,x2b,y2b = [float(c) for c in config['roi'].split(',')]
                wb, hb = x2b-xb, y2b-yb 
                filtered_detections, filtered_boxes, filtered_scores, filtered_features = [],[],[], []
                for detection, box, score, feature in zip(detections,boxes,scores,features):
                    xd,yd,wd,hd = box 
                    cxd, cyd = xd+wd/2., yd+hd/2. # center of detection box
                    # blackout_rectangle is [xb,yb,wb,hb]
                    if not (xb <= cxd <= xb+wb and yb <= cyd <= yb+hb):
                        filtered_boxes.append(box); filtered_scores.append(score); filtered_features.append(feature); filtered_detections.append(detection)
                    
                boxes, scores, features, detections = filtered_boxes, filtered_scores, filtered_features, filtered_detections

            tobdet1 = time.time()
            tobtrack0 = time.time()

            ## keypoint detection
            if keypoint_model is not None:
                from ubt.keypoint_detection import roi_segm

                t_kp_inf_start = time.time()
                if crop_dim is None:
                    crop_dim = roi_segm.get_roi_crop_dim(config['data_dir'], config['project_id'], config['test_video_ids'].split(',')[0],Hframe)
                keypoints = inference.inference_batch_keypoints(config, keypoint_model, crop_dim, frames_tensor, [detections], min_confidence_keypoints)[0]
                
            # Update tracker
            tracker.step({'img':frame,'detections':[detections, boxes, scores, features], 'frame_idx': frame_idx, 'file_tracking_results': config['file_tracking_results']})
            tobtrack1 = time.time() 
            tkptrack0 = time.time()
            if keypoint_model is not None:
                #keypoints = keypoint_buffer.popleft()
                # update tracked keypoints with new detections
                tracked_keypoints = keypoint_tracker.update(keypoints)
            else:
                keypoints = tracked_keypoints = []
            tkptrack1 = time.time()

            # Store results.        
            for track in tracker.tracks:
                bbox = track.to_tlwh()
                center0, center1, _, _ = util.tlhw2chw(bbox)
                _unmatched_steps = -1
                if hasattr(track,'time_since_update'):
                    _unmatched_steps = track.time_since_update
                elif hasattr(track,'steps_unmatched'):
                    _unmatched_steps = track.steps_unmatched
                else:
                    raise Exception("ERROR: can't find track's attributes time_since_update or unmatched_steps")

                result = [video_id, frame_idx, track.track_id, center0, center1, bbox[0], bbox[1], bbox[2], bbox[3], _unmatched_steps]
                with open( config['file_tracking_results'], 'a') as ff:
                    ff.write(','.join([str(r) for r in result])+'\n')
                results.append(result)
            
            if 'keypoint_method' in config and not config['keypoint_method'] == 'none':
                results_keypoints = []
                for kp in tracked_keypoints:
                    try:
                        result_keypoint = [video_id, frame_idx, kp.history_class[-1],kp.position[0],kp.position[1]]
                        file_csv_keypoints.write(','.join([str(r) for r in result_keypoint])+'\n')
                        results_keypoints.append(result_keypoint)
                    except Exception as e:
                        print(e)
            
            #print('[%i/%i] - %i detections. %i keypoints' % (config['count'], total_frame_number, len(detections), len(keypoints)))
            tvis0 = time.time()
            if showing:
                out = deep_sort_app.visualize(visualizer, frame, tracker, detections, keypoint_tracker, keypoints, tracked_keypoints, crop_dim, results, sketch_file=config['sketch_file'])
                video_writer.writeFrame(cv.cvtColor(out, cv.COLOR_BGR2RGB))
            tvis1 = time.time()

            if int(frame_idx) == 1010:
                tbenchend = time.time()
                print('[*] 1000 steps took',tbenchend-tbenchstart,'seconds')
                step_dur_ms = 1000.*(tbenchend-tbenchstart)/1000.
                fps = 1. / ( (tbenchend-tbenchstart)/1000. )
                print('[*] one time step takes on average',step_dur_ms,'ms',fps,'fps')

            if showing:
                cv.imshow("tracking visualization", cv.resize(out,None,None,fx=0.75,fy=0.75))
                cv.waitKey(1)
        
            if 0:
                dur_imread = timread1 - timread0
                dur_obdet = tobdet1 - tobdet0
                dur_obtrack = tobtrack1 - tobtrack0
                dur_kptrack = tkptrack1 - tkptrack0
                dur_vis = tvis1 - tvis0
                dur_imread, dur_obdet, dur_obtrack, dur_kptrack, dur_vis = [1000. * dur for dur in [dur_imread, dur_obdet, dur_obtrack, dur_kptrack, dur_vis]]
                print('imread',dur_imread,'obdetect',dur_obdet, 'obtrack',dur_obtrack, 'kptrack',dur_kptrack, 'vis',dur_vis)
    return video_file_out

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--objectdetection_model', required=True,default=None)
    parser.add_argument('--video', required=True,default=None)
    parser.add_argument('--output_video', required=False,default=None)
    parser.add_argument('--output_tracking_results', required=False,default=None)
    parser.add_argument('--keypoint_model', required=False,default=None,help='path to trained keypoint detection model .h5')
    parser.add_argument('--autoencoder_model', required=False,default=None)
    parser.add_argument('--project_id', required=False,type=int)
    parser.add_argument('--train_video_ids', default='')
    parser.add_argument('--test_video_ids', default='')
    parser.add_argument('--minutes', required=False,default=0.0,type=float,help="cut the video to the first n minutes, eg 2.5 cuts after the first 150seconds.")
    parser.add_argument('--min_confidence_boxes', required=False,default=0.75,type=float)
    parser.add_argument('--min_confidence_keypoints', required=False,default=0.5,type=float)
    parser.add_argument('--inference_objectdetection_batchsize', required=False,default=0,type=int)
    parser.add_argument('--inference_keypoint_batchsize', required=False,default=0,type=int)
    parser.add_argument('--track_tail', required=False,default=100,type=int,help="How many steps back in the past should the path of each animal be drawn? -1 -> draw complete path")
    parser.add_argument('--sketch_file', required=False,default=None, help="Black and White Sketch of the frame without animals")
    parser.add_argument('--yolox_exp', default='~/github/upper_bound_tracking/src/ubt/object_detection/YOLOX/exps/example/yolox_voc/yolox_voc_s.py')
    parser.add_argument('--yolox_name', default='yolox_s')
    parser.add_argument('--tracking_method', required=False,default='UpperBound',type=str,help="Tracking Algorithm to use: [DeepSORT, VIoU, UpperBound, SORT] defaults to VIoU")
    parser.add_argument('--tracking_hyperparameters', default=None, help="json file containing hyperparameters")
    parser.add_argument('--objectdetection_method', required=False,default="fasterrcnn", help="Object Detection Algorithm to use [fasterrcnn, ssd] defaults to fasterrcnn") 
    parser.add_argument('--objectdetection_resolution', required=False, default="640x640", help="xy resolution for object detection. coco pretrained model only available for 320x320, but smaller resolution saves time")
    parser.add_argument('--keypoint_resolution', required=False, default="224x224",help="patch size to analzye keypoints of individual animals")
    parser.add_argument('--keypoint_method', required=False,default="none", help="Keypoint Detection Algorithm to use [none, hourglass2, hourglass4, hourglass8, vgg16, efficientnet, efficientnetLarge, psp]. defaults to none. recommended psp") 
    parser.add_argument('--upper_bound', required=False,default=0,type=int)
    parser.add_argument('--data_dir', required=False, default = '~/data/ubt')
    parser.add_argument('--delete_all_checkpoints', required=False, action="store_true")
    parser.add_argument('--video_resolution', default=None, help='resolution the video is downscaled to before processing to reduce runtime, eg 640x480. default no downscaling')
    parser.add_argument('--use_all_data4train', action='store_true')
    parser.add_argument('--roi', default = None, help = "rectangle in form x1,y1,x2,y2 of Region of Interest. detections outside of ROI are filtered out")
    args = parser.parse_args()
    assert args.tracking_method in ['Byte','DeepSORT', 'VIoU', 'UpperBound', 'SORT','OCSORT','UpperBoundOCSORT']
    assert args.objectdetection_method in ['fasterrcnn', 'ssd']
    assert args.keypoint_method in ['none', 'hourglass2', 'hourglass4', 'hourglass8', 'vgg16', 'efficientnet', 'efficientnetLarge', 'psp']
    args.yolox_exp = os.path.expanduser(args.yolox_exp)
    args.data_dir = os.path.expanduser(args.data_dir)
    main(args)