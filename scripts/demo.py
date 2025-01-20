import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import gc
import sys
import ffmpeg  # Import ffmpeg
import logging  # Import logging
from sam2.build_sam import build_sam2_video_predictor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

color = [(255, 0, 0)]

def save_mask(mask, frame_idx, obj_id, output_folder):
    """Save the mask as a black and white image."""
    mask_img = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)
    mask_img[mask] = 255  # Set mask to white
    mask_filename = os.path.join(output_folder, f'mask_frame_{frame_idx:08d}_obj_{obj_id}.png')
    cv2.imwrite(mask_filename, mask_img)
    logging.debug(f"Saved mask: {mask_filename}")

def load_txt(gt_path):
    with open(gt_path, 'r') as f:
        gt = f.readlines()
    prompts = {}
    for fid, line in enumerate(gt):
        try:
            x, y, w, h = map(float, line.strip().split(','))
            x, y, w, h = int(x), int(y), int(w), int(h)
            prompts[fid] = ((x, y, x + w, y + h), 0)
        except ValueError as e:
            logging.error(f"Error parsing line {fid}: {line.strip()} - {e}")
    logging.info(f"Loaded {len(prompts)} prompts from {gt_path}")
    return prompts

def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")

def prepare_frames_or_path(video_path):
    if video_path.endswith(".mp4") or osp.isdir(video_path):
        return video_path
    else:
        raise ValueError("Invalid video_path format. Should be .mp4 or a directory of jpg frames.")

def main(args):
    logging.info("Starting main function")
    model_cfg = determine_model_cfg(args.model_path)
    logging.info(f"Using model config: {model_cfg}")
    
    predictor = build_sam2_video_predictor(model_cfg, args.model_path, device="cuda:0")
    frames_or_path = prepare_frames_or_path(args.video_path)
    prompts = load_txt(args.txt_path)

    frame_rate = 30
    loaded_frames = []
    
    if args.save_to_video:
        if osp.isdir(args.video_path):
            frames = sorted([osp.join(args.video_path, f) for f in os.listdir(args.video_path) if f.endswith((".jpg", ".jpeg", ".JPG", ".JPEG"))])
            loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
            height, width = loaded_frames[0].shape[:2]
            logging.info(f"Loaded {len(loaded_frames)} frames from directory")
        else:
            cap = cv2.VideoCapture(args.video_path)
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            loaded_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                loaded_frames.append(frame)
            cap.release()
            height, width = loaded_frames[0].shape[:2]
            logging.info(f"Loaded {len(loaded_frames)} frames from video")
            if len(loaded_frames) == 0:
                raise ValueError("No frames were loaded from the video.")

    if args.save_to_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.video_output_path, fourcc, frame_rate, (width, height))
        logging.info(f"Output video will be saved to {args.video_output_path}")

    # Ensure the output folders for masks and frames exist
    masks_output_folder = os.path.join(args.frames_output_folder, "masks")
    frames_output_folder = os.path.join(args.frames_output_folder, "frames")
    os.makedirs(masks_output_folder, exist_ok=True)
    os.makedirs(frames_output_folder, exist_ok=True)
    logging.info(f"Masks will be saved to {masks_output_folder}")
    logging.info(f"Frames will be saved to {frames_output_folder}")

    # Create a separate video for segmented objects if specified
    if args.create_segmented_video and args.save_to_video:
        segmented_video_output_path = os.path.join(args.frames_output_folder, "segmented_video.mp4")
        # Initialize ffmpeg process for writing RGBA video
        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', s='{}x{}'.format(width, height), pix_fmt='rgba')
            .output(segmented_video_output_path, vcodec='libx264', pix_fmt='yuv420p', crf=18)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
        logging.info(f"Segmented video will be saved to {segmented_video_output_path}")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)
        if 0 not in prompts:
            logging.error("Prompt for frame 0 not found.")
            return
        bbox, track_label = prompts[0]
        _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)
        logging.info("Initialized predictor state and added initial box")

        frame_count = 0
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            logging.debug(f"Processing frame {frame_idx} with {len(object_ids)} objects")
            mask_to_vis = {}
            bbox_to_vis = {}

            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                non_zero_indices = np.argwhere(mask)
                if len(non_zero_indices) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                bbox_to_vis[obj_id] = bbox
                mask_to_vis[obj_id] = mask

                # Save the mask for the current object
                save_mask(mask, frame_idx, obj_id, masks_output_folder)  # Save to masks folder

            if args.save_to_video:
                img = loaded_frames[frame_idx]
                # Save the current frame to the frames folder
                frame_filename = os.path.join(frames_output_folder, f'frame_{frame_idx:08d}.png')
                cv2.imwrite(frame_filename, img)  # Save the frame
                logging.debug(f"Saved frame {frame_idx} to {frame_filename}")

                if args.create_segmented_video:
                    segmented_frame = np.zeros((height, width, 4), dtype=np.uint8)  # Create an RGBA frame
                    for obj_id, mask in mask_to_vis.items():
                        segmented_frame[mask, :3] = img[mask]        # Assign RGB values
                        segmented_frame[mask, 3] = 255
                    # Write the RGBA frame to ffmpeg
                    process.stdin.write(segmented_frame.tobytes())
                    logging.debug(f"Wrote segmented frame {frame_idx} to ffmpeg")

                for obj_id, mask in mask_to_vis.items():
                    mask_img = np.zeros((height, width, 3), np.uint8)
                    mask_img[mask] = color[(obj_id + 1) % len(color)]
                    img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)

                for obj_id, bbox in bbox_to_vis.items():
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color[obj_id % len(color)], 2)

                out.write(img)
                logging.debug(f"Wrote annotated frame {frame_idx} to output video")

                frame_count += 1

        if args.save_to_video:
            out.release()
            logging.info(f"Released video writer after processing {frame_count} frames")
        if args.create_segmented_video and args.save_to_video:
            process.stdin.close()
            process.wait()  # Wait for the ffmpeg process to finish
            logging.info("Closed ffmpeg process")

    del predictor, state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()
    logging.info("Released all resources and cleared caches")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="Input video path or directory of frames.")
    parser.add_argument("--txt_path", required=True, help="Path to ground truth text file.")
    parser.add_argument("--model_path", default="sam2/checkpoints/sam2.1_hiera_base_plus.pt", help="Path to the model checkpoint.")
    parser.add_argument("--video_output_path", default="demo.mp4", help="Path to save the output video.")
    parser.add_argument("--frames_output_folder", default="out", help="Path to save the masks and frames.")
    parser.add_argument("--create_segmented_video", default=True, action='store_true', help="Create a video with only segmented objects.")
    parser.add_argument("--save_to_video", default=True, type=lambda x: (str(x).lower() in ['true','1', 'yes']), help="Save results to a video.")
    args = parser.parse_args()
    main(args)
