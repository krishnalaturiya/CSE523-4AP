"""
hand_body_movement_detection.py

Detects hand movement + body/arm movement metrics per second.
Outputs a structured JSON file with all extracted data.

Usage:
    python hand_body_movement_detection.py "https://www.youtube.com/watch?v=..." 2-10 88-98 15-25
    
Input format:
    [YouTube_URL] [start1-end1] [start2-end2] ...
    
Example:
    python hand_body_movement_detection.py "https://www.youtube.com/watch?v=xyz" 0-30 45-90 120-150
"""

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import tempfile

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.solutions import hands, pose

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GESTURE] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("gesture")


def _build_hands_detector():
    return hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.1,
        min_tracking_confidence=0.1,
    )


def _build_pose_detector():
    return pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def preprocess_frame(frame, brightness_increase=60, zoom_factor=1.3):
    """Brighten and zoom-crop for better MediaPipe detection accuracy."""
    h, w = frame.shape[:2]

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    l_channel = cv2.add(l_channel, brightness_increase)
    l_channel = np.clip(l_channel, 0, 255)
    lab = cv2.merge([l_channel, a, b])
    brightened = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    new_h = int(h / zoom_factor)
    new_w = int(w / zoom_factor)
    start_y = (h - new_h) // 2
    start_x = (w - new_w) // 2
    zoomed = brightened[start_y:start_y + new_h, start_x:start_x + new_w]
    return cv2.resize(zoomed, (w, h), interpolation=cv2.INTER_LINEAR)


def get_body_bounding_box(pose_landmarks, frame_h, frame_w, margin=0.3):
    if not pose_landmarks or not pose_landmarks.landmark:
        return None

    xs = [lm.x for lm in pose_landmarks.landmark if lm.visibility > 0.3]
    ys = [lm.y for lm in pose_landmarks.landmark if lm.visibility > 0.3]

    if not xs or not ys:
        return None

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_margin = (x_max - x_min) * margin
    y_margin = (y_max - y_min) * margin

    return (
        max(0, x_min - x_margin),
        max(0, y_min - y_margin),
        min(1.0, x_max + x_margin),
        min(1.0, y_max + y_margin),
    )


def crop_frame_to_bbox(frame, bbox):
    h, w = frame.shape[:2]
    x1 = int(bbox[0] * w)
    y1 = int(bbox[1] * h)
    x2 = int(bbox[2] * w)
    y2 = int(bbox[3] * h)
    return frame[y1:y2, x1:x2]


def _hand_centroid(hand_landmarks):
    """Calculate centroid of hand landmarks (21 points)."""
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_body_metrics(pose_landmarks):
    if not pose_landmarks or not pose_landmarks.landmark:
        return None

    lm = pose_landmarks.landmark

    left_shoulder  = (lm[11].x, lm[11].y)
    right_shoulder = (lm[12].x, lm[12].y)
    left_hip       = (lm[23].x, lm[23].y)
    right_hip      = (lm[24].x, lm[24].y)
    left_elbow     = (lm[13].x, lm[13].y)
    right_elbow    = (lm[14].x, lm[14].y)
    left_wrist     = (lm[15].x, lm[15].y)
    right_wrist    = (lm[16].x, lm[16].y)

    body_center = (
        (left_shoulder[0] + right_shoulder[0] + left_hip[0] + right_hip[0]) / 4,
        (left_shoulder[1] + right_shoulder[1] + left_hip[1] + right_hip[1]) / 4,
    )

    shoulder_center = (
        (left_shoulder[0] + right_shoulder[0]) / 2,
        (left_shoulder[1] + right_shoulder[1]) / 2,
    )
    hip_center = (
        (left_hip[0] + right_hip[0]) / 2,
        (left_hip[1] + right_hip[1]) / 2,
    )

    avg_elbow_distance = (
        _euclidean(left_elbow, body_center) + _euclidean(right_elbow, body_center)
    ) / 2

    return {
        'shoulder_width':      float(_euclidean(left_shoulder, right_shoulder)),
        'shoulder_height_diff': float(abs(left_shoulder[1] - right_shoulder[1])),
        'hip_width':           float(_euclidean(left_hip, right_hip)),
        'torso_tilt':          float(_euclidean(shoulder_center, hip_center)),
        'avg_elbow_distance':  float(avg_elbow_distance),
        'avg_arm_height':      float((left_wrist[1] + right_wrist[1]) / 2),
        'arm_spread':          float(_euclidean(left_elbow, right_elbow)),
        'arm_velocity':        0.0,
        'body_center_velocity': 0.0,
        'hand_distance':       0.0,
        'hand_velocity':       0.0,
        'body_center':         body_center,
    }


def compute_hand_metrics(hand_landmarks_list):
    """
    Compute hand metrics from detected hand landmarks.
    
    Args:
        hand_landmarks_list: List of hand landmark objects
        
    Returns:
        dict with hand_distance and hand_velocity (both 0.0 if no hands)
    """
    if not hand_landmarks_list or len(hand_landmarks_list) == 0:
        return {
            'hand_distance': 0.0,
            'hand_velocity': 0.0,
        }
    
    # Get centroids of all detected hands
    hand_centroids = [_hand_centroid(hand) for hand in hand_landmarks_list]
    
    # Calculate distance between hands (if 2 hands detected)
    if len(hand_centroids) >= 2:
        hand_distance = _euclidean(hand_centroids[0], hand_centroids[1])
    else:
        # Single hand or no hands - distance is 0
        hand_distance = 0.0
    
    return {
        'hand_distance': float(hand_distance),
        'hand_velocity': 0.0,  # Will be computed per-frame
    }


def download_youtube(url: str, output_dir: str) -> str:
    """Download YouTube video using yt-dlp."""
    log.info(f"Downloading: {url}")
    output_template = os.path.join(output_dir, "%(id)s.%(ext)s")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--print", "after_move:filepath",
        url,
    ]
    log.info(f"yt-dlp cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"yt-dlp failed (rc={result.returncode}): {result.stderr[:500]}")
        raise RuntimeError(f"yt-dlp failed:\n{result.stderr}")
    lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    if not lines:
        raise RuntimeError("yt-dlp did not print the output filepath.")
    log.info(f"Downloaded to: {lines[-1]}")
    return lines[-1]


def analyze_segment_full(
    video_path: str,
    start_sec: float,
    end_sec: float,
    segment_label: str,
) -> list:
    """Analyze a video segment and return per-second metrics."""
    log.info(f"analyze_segment_full: {segment_label} [{start_sec}s – {end_sec}s]")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error(f"Cannot open video: {video_path}")
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info(f"Video opened: fps={fps:.2f}, total_frames={total_frames}")
    if fps <= 0:
        fps = 25.0
        log.warning("FPS undetectable, defaulting to 25.0")

    results_per_second = []

    with _build_hands_detector() as hand_detector, _build_pose_detector() as pose_detector:
        log.info("MediaPipe detectors initialised")
        current_second = int(start_sec)

        while current_second < int(end_sec):
            sec_start = current_second
            frame_start = int(sec_start * fps)
            frame_end = int(min(current_second + 1, end_sec) * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

            frame_body_metrics_list = []
            frame_hand_metrics_list = []
            max_hands_seen = 0
            prev_body_metrics = None
            prev_hand_distance = None

            for _ in range(frame_end - frame_start):
                ret, frame = cap.read()
                if not ret:
                    break

                h, w = frame.shape[:2]
                preprocessed = preprocess_frame(frame)
                rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
                pose_result = pose_detector.process(rgb)

                if not pose_result.pose_landmarks:
                    continue

                body_bbox = get_body_bounding_box(pose_result.pose_landmarks, h, w, margin=0.3)
                if not body_bbox:
                    continue

                body_metrics = compute_body_metrics(pose_result.pose_landmarks)
                if body_metrics:
                    if prev_body_metrics:
                        body_metrics['arm_velocity'] = float(
                            abs(body_metrics['avg_elbow_distance'] -
                                prev_body_metrics['avg_elbow_distance']) * fps
                        )
                        body_metrics['body_center_velocity'] = float(
                            _euclidean(body_metrics['body_center'],
                                       prev_body_metrics['body_center']) * fps
                        )
                    frame_body_metrics_list.append(body_metrics)
                    prev_body_metrics = body_metrics

                cropped = crop_frame_to_bbox(preprocessed, body_bbox)
                if cropped.size > 0:
                    hand_result = hand_detector.process(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                    if hand_result.multi_hand_landmarks:
                        max_hands_seen = max(max_hands_seen, len(hand_result.multi_hand_landmarks))
                        
                        # Compute hand metrics
                        hand_metrics = compute_hand_metrics(hand_result.multi_hand_landmarks)
                        
                        # Compute hand velocity
                        if prev_hand_distance is not None:
                            hand_metrics['hand_velocity'] = float(
                                abs(hand_metrics['hand_distance'] - prev_hand_distance) * fps
                            )
                        
                        frame_hand_metrics_list.append(hand_metrics)
                        prev_hand_distance = hand_metrics['hand_distance']

            # Aggregate metrics for this second
            if frame_body_metrics_list:
                avg_metrics = {
                    k: float(np.mean([m[k] for m in frame_body_metrics_list]))
                    for k in [
                        'shoulder_width', 'shoulder_height_diff', 'hip_width',
                        'torso_tilt', 'avg_elbow_distance', 'avg_arm_height',
                        'arm_spread', 'arm_velocity', 'body_center_velocity',
                    ]
                }
            else:
                avg_metrics = {
                    k: 0.0 for k in [
                        'shoulder_width', 'shoulder_height_diff', 'hip_width',
                        'torso_tilt', 'avg_elbow_distance', 'avg_arm_height',
                        'arm_spread', 'arm_velocity', 'body_center_velocity',
                    ]
                }
            
            # Aggregate hand metrics
            if frame_hand_metrics_list:
                hand_avg_metrics = {
                    'hand_distance': float(np.mean([m['hand_distance'] for m in frame_hand_metrics_list])),
                    'hand_velocity': float(np.mean([m['hand_velocity'] for m in frame_hand_metrics_list])),
                }
            else:
                hand_avg_metrics = {
                    'hand_distance': 0.0,
                    'hand_velocity': 0.0,
                }

            results_per_second.append({
                'second': sec_start,
                'hands_detected': max_hands_seen,
                **avg_metrics,
                **hand_avg_metrics,
            })
            log.info(
                f"  s={sec_start}: pose_frames={len(frame_body_metrics_list)} "
                f"hands={max_hands_seen} "
                f"arm_vel={avg_metrics['arm_velocity']:.3f} "
                f"arm_spread={avg_metrics['arm_spread']:.3f}"
            )
            current_second += 1

    cap.release()
    log.info(f"Segment done: {len(results_per_second)} seconds processed")
    return results_per_second


def parse_timestamp_ranges(range_strings):
    """
    Parse timestamp ranges from strings like '2-10', '88-98'.
    
    Args:
        range_strings: List of strings like ['2-10', '88-98']
        
    Returns:
        List of tuples like [(2, 10), (88, 98)]
    """
    ranges = []
    for range_str in range_strings:
        parts = range_str.split('-')
        if len(parts) != 2:
            raise ValueError(f"Invalid range format: '{range_str}'. Use format like '2-10'")
        try:
            start = float(parts[0].strip())
            end = float(parts[1].strip())
            if start >= end:
                raise ValueError(f"Invalid range: start ({start}) must be less than end ({end})")
            ranges.append((start, end))
        except ValueError as e:
            raise ValueError(f"Cannot parse range '{range_str}': {e}")
    return ranges


def main():
    parser = argparse.ArgumentParser(
        description="Hand + body movement detection pipeline from YouTube video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hand_body_movement_detection.py "https://www.youtube.com/watch?v=xyz" 0-30 45-90
  python hand_body_movement_detection.py "https://www.youtube.com/watch?v=abc" 10-20 50-70 100-120 --output analysis.json
        """
    )
    parser.add_argument(
        "url",
        help="YouTube URL of the video to analyze"
    )
    parser.add_argument(
        "ranges",
        nargs="+",
        help="Time ranges to analyze (format: start-end in seconds, e.g., 2-10 88-98)"
    )
    parser.add_argument(
        "--output",
        default="results.json",
        help="Output JSON file path (default: results.json)"
    )
    parser.add_argument(
        "--video-file",
        default=None,
        metavar="PATH",
        help="Path to an already-downloaded video file. Skips the YouTube download step."
    )

    args = parser.parse_args()

    # Skip URL validation when a local file is provided — the URL is still logged
    # for metadata purposes but yt-dlp is never called.
    if not args.video_file:
        if not ("youtube.com" in args.url or "youtu.be" in args.url):
            print("[ERROR] Invalid YouTube URL provided")
            sys.exit(1)

    # Parse timestamp ranges
    try:
        ranges = parse_timestamp_ranges(args.ranges)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    log.info("=" * 60)
    log.info("HAND & BODY MOVEMENT DETECTION")
    log.info(f"URL   : {args.url}")
    log.info(f"Ranges: {', '.join([f'{int(s)}-{int(e)}s' for s, e in ranges])}")
    log.info(f"Output: {args.output}")
    log.info("=" * 60)

    # Use a pre-downloaded file if --video-file was supplied, otherwise download.
    # Sharing the file avoids a second simultaneous yt-dlp call when app.py is
    # running audio analysis at the same time.
    tmpdir = None
    if args.video_file:
        video_path = args.video_file
        log.info(f"Using pre-downloaded video: {video_path}")
        if not os.path.isfile(video_path):
            log.error(f"--video-file not found: {video_path}")
            sys.exit(1)
    else:
        try:
            tmpdir = tempfile.mkdtemp(prefix="hand_detect_")
            log.info(f"Temp dir: {tmpdir}")
            video_path = download_youtube(args.url, tmpdir)
            log.info(f"Download complete: {video_path}")
        except Exception as e:
            log.error(f"Download failed: {e}", exc_info=True)
            sys.exit(1)

    # Analyze each range
    all_results = {}

    for idx, (start, end) in enumerate(ranges, 1):
        segment_label = f"Segment_{idx}"
        print(f"\n{'='*70}")
        print(f"Processing {segment_label}: {int(start)}s – {int(end)}s")
        print(f"{'='*70}")

        try:
            log.info(f"Starting analysis for {segment_label}")
            seg_data = analyze_segment_full(video_path, start, end, segment_label)
            all_results[segment_label] = {
                'time_range': {
                    'start': int(start),
                    'end': int(end)
                },
                'seconds': seg_data
            }
            log.info(f"{segment_label} done: {len(seg_data)} seconds")
        except Exception as e:
            log.error(f"Analysis failed for {segment_label}: {e}", exc_info=True)
            sys.exit(1)

    # Save results
    output_data = {
        'metadata': {
            'video_url': args.url,
            'ranges_analyzed': len(ranges),
            'output_file': args.output
        },
        'segments': all_results
    }

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        log.info(f"Results saved to: {args.output}")
        log.info("Analysis complete!")
    except Exception as e:
        log.error(f"Failed to save results: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Cleanup
        if tmpdir:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()