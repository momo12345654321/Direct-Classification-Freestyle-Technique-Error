# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import cv2
import mmcv
import numpy as np
import os
import os.path as osp
import pickle
import shutil
import torch
import warnings
from scipy.optimize import linear_sum_assignment


try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    def inference_detector(*args, **kwargs):
        pass

    def init_detector(*args, **kwargs):
        pass
    warnings.warn(
        'Failed to import `inference_detector` and `init_detector` from `mmdet.apis`. '
        'Make sure you can successfully import these if you want to use related features. '
    )

try:
    from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result
except (ImportError, ModuleNotFoundError):
    def init_pose_model(*args, **kwargs):
        pass

    def inference_top_down_pose_model(*args, **kwargs):
        pass

    def vis_pose_result(*args, **kwargs):
        pass

    warnings.warn(
        'Failed to import `init_pose_model`, `inference_top_down_pose_model`, `vis_pose_result` from '
        '`mmpose.apis`. Make sure you can successfully import these if you want to use related features. '
    )


try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.75
FONTCOLOR = (255, 255, 255)  # BGR, white
THICKNESS = 1
LINETYPE = 1


def parse_args():
    parser = argparse.ArgumentParser(description='PoseC3D demo')
    parser.add_argument('video', help='video file/url')
    parser.add_argument('out_filename', nargs='?', help='output video; omit to save only the pkl')
    parser.add_argument(
        '--config',
        default='configs/posec3d/slowonly_r50_ntu120_xsub/joint.py',
        help='skeleton action recognition config file path')
    parser.add_argument(
        '--checkpoint',
        default='https://download.openmmlab.com/mmaction/pyskl/ckpt/posec3d/slowonly_r50_ntu120_xsub/joint.pth',
        help='skeleton action recognition checkpoint file/url')
    parser.add_argument(
        '--det-config',
        default='demo/faster_rcnn_r50_fpn_1x_coco-person.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/'
                 'faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--pose-config',
        default='demo/hrnet_w32_coco_256x192.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default='https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument(
        '--rotation',
        choices=['cw', 'ccw'],
        required=True,
        help='rotate frames 90 degrees so the swimmer is head-up for detection and pose: '
             'cw if the swimmer moves right-to-left, ccw if left-to-right')
    parser.add_argument(
        '--label-map',
        default='tools/data/label_map/nturgbd_120.txt',
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side',
        type=int,
        default=480,
        help='specify the short-side length of the image')
    parser.add_argument(
        '--output-pkl',
        type=str,
        required=True,
        help='path for saving the numerical pose annotation')
    args = parser.parse_args()
    return args


def frame_extraction(video_path, short_side, rotation):
    """Extract frames given video_path.

    Args:
        video_path (str): The video_path.
    """
    # Load the video, extract frames into ./tmp/video_name
    target_dir = osp.join('./tmp', osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    vid = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    new_h, new_w = None, None
    while flag:
        if new_h is None:
            h, w, _ = frame.shape
            new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.inf))

        frame = mmcv.imresize(frame, (new_w, new_h))

        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)

        # The models read the rotated (head-up) frame; `frames` keeps the original for the video.
        cv2.imwrite(frame_path, np.ascontiguousarray(np.rot90(frame, -1 if rotation == 'cw' else 1)))
        cnt += 1
        flag, frame = vid.read()

    return frame_paths, frames


def detection_inference(args, frame_paths):
    """Detect human boxes given frame paths.

    Args:
        args (argparse.Namespace): The arguments.
        frame_paths (list[str]): The paths of frames to do detection inference.

    Returns:
        list[np.ndarray]: The human detection results.
    """
    model = init_detector(args.det_config, args.det_checkpoint, args.device)
    assert model is not None, ('Failed to build the detection model. Check if you have installed mmcv-full properly. '
                               'You should first install mmcv-full successfully, then install mmdet, mmpose. ')
    assert model.CLASSES[0] == 'person', 'We require you to use a detector trained on COCO'
    results = []
    print('Performing Human Detection for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        result = inference_detector(model, frame_path)
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][:, 4] >= args.det_score_thr]
        #frames rotated
        result = result[(result[:, 3] - result[:, 1]) > (result[:, 2] - result[:, 0])]
        results.append(result)
        prog_bar.update()
    return results


def pose_inference(args, frame_paths, det_results):
    model = init_pose_model(args.pose_config, args.pose_checkpoint,
                            args.device)
    ret = []
    print('Performing Human Pose Estimation for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for f, d in zip(frame_paths, det_results):
        # Align input format
        d = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        ret.append(pose)
        prog_bar.update()
    return ret

def unrotate(xy, rotation, h, w):
    """Map (..., 2) points from the rotated frame back to the original h x w frame."""
    x, y = xy[..., 0], xy[..., 1]
    if rotation == 'cw':  # original (x, y) -> rotated (h-1-y, x)
        return np.stack([y, h - 1 - x], axis=-1)
    return np.stack([w - 1 - y, x], axis=-1)  # ccw: original (x, y) -> rotated (y, w-1-x)

def dist_ske(ske1, ske2):
    dist = np.linalg.norm(ske1[:, :2] - ske2[:, :2], axis=1) * 2
    diff = np.abs(ske1[:, 2] - ske2[:, 2])
    return np.sum(np.maximum(dist, diff))


def pose_tracking(pose_results, max_tracks=2, thre=30):
    tracks, num_tracks = [], 0
    num_joints = None
    for idx, poses in enumerate(pose_results):
        if len(poses) == 0:
            continue
        if num_joints is None:
            num_joints = poses[0]['keypoints'].shape[0]
        track_proposals = [t for t in tracks if t['data'][-1][0] > idx - thre]
        n, m = len(track_proposals), len(poses)
        scores = np.zeros((n, m))
        overlap = np.zeros((n, m), dtype=bool)

        for i in range(n):
            last = track_proposals[i]['data'][-1][1]
            for j in range(m):
                scores[i][j] = dist_ske(last['keypoints'], poses[j]['keypoints'])
                a, b = last['bbox'], poses[j]['bbox']
                overlap[i][j] = a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

        row, col = linear_sum_assignment(np.where(overlap, scores, scores.sum() + 1))
        matched=set()
        for r, c in zip(row, col):
            if overlap[r][c]:
                track_proposals[r]['data'].append((idx, poses[c]))
                matched.add(c)
        # Every pose that did not join a track starts a new one.
        for j in range(m):
            if j not in matched:
                num_tracks += 1
                new_track = dict(data=[])
                new_track['track_id'] = num_tracks
                new_track['data'] = [(idx, poses[j])]
                tracks.append(new_track)
    if num_joints is None:
        return None, None
    tracks.sort(key=lambda x: -len(x['data']))
    result = np.zeros((max_tracks, len(pose_results), num_joints, 3), dtype=np.float16)
    for i, track in enumerate(tracks[:max_tracks]):
        for item in track['data']:
            idx, pose = item
            result[i, idx] = pose['keypoints']
    return result[..., :2], result[..., 2]


def main():
    args = parse_args()

    frame_paths, original_frames = frame_extraction(args.video,
                                                    args.short_side,
                                                    args.rotation)
    num_frame = len(frame_paths)
    h, w, _ = original_frames[0].shape


    # Get Human detection results
    det_results = detection_inference(args, frame_paths)
    torch.cuda.empty_cache()

    pose_results = pose_inference(args, frame_paths, det_results)
    torch.cuda.empty_cache()

    video_id = osp.splitext(osp.basename(args.video))[0]

    fake_anno = dict(
        frame_dir=video_id,
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)

    # Map keypoints and boxes back to the original frame (used by tracking, the pkl and the video).
    for poses in pose_results:
        for pose in poses:
            pose['keypoints'][:, :2] = unrotate(pose['keypoints'][:, :2], args.rotation, h, w)
            corners = unrotate(pose['bbox'][:4].reshape(2, 2), args.rotation, h, w)
            pose['bbox'][:4] = np.concatenate([corners.min(0), corners.max(0)])

    keypoint, keypoint_score = pose_tracking(
        pose_results,
        max_tracks=1
    )

    fake_anno['keypoint'] = keypoint
    fake_anno['keypoint_score'] = keypoint_score


    output_dir = osp.dirname(args.output_pkl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_pkl, 'wb') as f:
        pickle.dump(fake_anno, f)

    print(f'Saved numerical pose annotation to: {args.output_pkl}')
    print(f"Keypoint shape: {fake_anno['keypoint'].shape}")
    print(f"Keypoint-score shape: {fake_anno['keypoint_score'].shape}")

    if args.out_filename is None:
        shutil.rmtree(osp.dirname(frame_paths[0]))
        return


    
    action_label = ''

    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint,
                                 args.device)
    # Draw the tracked swimmer saved to the pkl (frames without a swimmer have score 0, so nothing is drawn).
    vis_frames = [
        vis_pose_result(pose_model, original_frames[i],
                        [dict(keypoints=np.concatenate([keypoint[0, i], keypoint_score[0, i, :, None]], -1))])
        for i in range(num_frame)
    ]
    for frame in vis_frames:
        cv2.putText(frame, action_label, (10, 30), FONTFACE, FONTSCALE,
                    FONTCOLOR, THICKNESS, LINETYPE)

    vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames], fps=24)
    vid.write_videofile(args.out_filename, remove_temp=True)

    tmp_frame_dir = osp.dirname(frame_paths[0])
    shutil.rmtree(tmp_frame_dir)


if __name__ == '__main__':
    main()
