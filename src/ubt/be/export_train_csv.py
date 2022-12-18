"""
    export training images to CSV files and sample files

"""

from glob import glob 
import os 
import shutil 
import numpy as np 
import cv2 as cv 
import zipfile
import time 
import pickle
from ubt.be import dbconnection

from ubt.object_detection import finetune 

def main(args):
    args['dir_out'] = os.path.expanduser(args['dir_out'])
    args['data_dir'] = os.path.expanduser(args['data_dir'])
    args['csv_out_boxes'] = os.path.expanduser(args['csv_out_boxes'])
    args['csv_out_keypoints'] = os.path.expanduser(args['csv_out_keypoints'])

    if os.path.isfile(args['csv_out_boxes']): os.remove(args['csv_out_boxes'])
    if os.path.isfile(args['csv_out_keypoints']): os.remove(args['csv_out_keypoints'])
    
    os.makedirs(args['dir_out'],exist_ok=True)

    project_id=args['project_id']
    video_ids = args['video_ids']
    
    # load source database
    db = dbconnection.DatabaseConnection(file_db = os.path.join(args['data_dir'], 'data.db'))

    # hacky: make sure that all the frames are written to disk
    #finetune.get_bbox_data({'project_id': project_id, 'data_dir': args['data_dir']}, ','.join([str(i) for i in args['video_ids']]), abort_early = True)

    ## fetch project, video, bounding boxes and keypoints data from source database
    db.execute("select * from projects where id = %i;" % project_id)
    project_data = db.cur.fetchone()
    video_data = {}
    box_data = {}
    keypoints_data = {}
    labeled_files = []
    
    
    for video_id in video_ids:
        db.execute("select * from videos where id = %i;" % video_id)
        video_data[video_id] = [x for x in db.cur.fetchall()][0]
        
        video_name = os.path.split( video_data[video_id][1] )[1]
        db.execute('select * from bboxes where video_id = %i;' % video_id)
        box_data[video_id] = [x for x in db.cur.fetchall()]

        db.execute('select * from keypoint_positions where video_id = %i;' % video_id)
        keypoints_data[video_id] = [x for x in db.cur.fetchall()]

        with open(args['csv_out_boxes'],'a+') as f:
            f.write('video_name,video_id,frame,x1,y1,x2,y2\n')
            for _d in box_data[video_id]:
                _, _, frame_idx, individual_id, x1, y1, x2, y2, is_visible = _d
                frame_file = os.path.join( args['data_dir'], 'projects', str(project_id), str(video_id), 'frames', 'train', '%05d.png' % int(frame_idx) )
                labeled_files.append( frame_file )
                f.write(f"{video_name},{video_id},{frame_idx},{x1},{y1},{x2},{y2}\n")

        with open(args['csv_out_keypoints'],'a+') as f:
            f.write('video_name,video_id,frame,individual_id,keypoint_name,x1,y1\n')
            
            for _d in keypoints_data[video_id]:
                labid, video_id, frame_idx, keypoint_name, individual_id, keypoint_x, keypoint_y, is_visible = _d
                frame_file = os.path.join( args['data_dir'], 'projects', str(project_id), str(video_id), 'frames', 'train', '%05d.png' % int(frame_idx) )
                labeled_files.append( frame_file )
                f.write(f"{video_name},{video_id},{frame_idx},{individual_id},{keypoint_name},{keypoint_x}, {keypoint_y}\n")

    labeled_files = list(set(labeled_files))

                    
    ## copy files to output directory
    print('[*] copying %i files' % len(labeled_files))
    for f in labeled_files:
        fo = args['dir_out'] + '/projects/' + f.split('/projects/')[1]
        dd = os.path.split(fo)[0]
        if not os.path.isdir(dd): os.makedirs(dd)
        shutil.copy(f, fo)


if __name__ == '__main__':
    main({
        'data_dir': '~/data/ubt',
        'dir_out': '/tmp/traindata',
        'project_id': 1,
        'video_ids': [1,3,4,6],
        'csv_out_boxes': '/tmp/traindata/boxes.csv',
        'csv_out_keypoints': '/tmp/traindata/keypoints.csv'
    })