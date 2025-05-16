vim run_rmbg.py

import os
import shutil
from gradio_client import Client, handle_file

# 初始化客户端
client = Client("http://localhost:7860/")

# 定义输入和输出目录
input_dir = "Xiang_Float_After_Tomorrow_Head_SPLITED"
output_dir_rmbg = "Xiang_Float_After_Tomorrow_Head_SPLITED_RMBG"
output_dir_mask = "Xiang_Float_After_Tomorrow_Head_SPLITED_RMBG_MASK"

# 创建输出目录（如果不存在）
os.makedirs(output_dir_rmbg, exist_ok=True)
os.makedirs(output_dir_mask, exist_ok=True)

# 遍历输入目录中的所有mp4文件
for filename in os.listdir(input_dir):
    if filename.endswith(".mp4"):
        input_path = os.path.join(input_dir, filename)

        try:
            print(f"正在处理: {filename}")

            # 调用API处理视频
            result = client.predict(
                input_video_path={"video": handle_file(input_path)},
                api_name="/video"
            )

            # 获取结果文件路径
            rmbg_video_path = result[0]["video"]
            mask_video_path = result[1]["video"]

            # 构建输出路径（保持相同文件名）
            output_path_rmbg = os.path.join(output_dir_rmbg, filename)
            output_path_mask = os.path.join(output_dir_mask, filename)

            # 复制结果文件到目标目录
            shutil.copy(rmbg_video_path, output_path_rmbg)
            shutil.copy(mask_video_path, output_path_mask)

            print(f"成功保存: {filename}")

        except Exception as e:
            print(f"处理 {filename} 时出错: {str(e)}")

print("所有文件处理完成！")


from PIL import Image, ImageDraw
import numpy as np
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
from moviepy.video.fx.all import resize
from pathlib import Path
import os

def get_sorted_files(directory, extension):
    """Return sorted list of files with given extension in directory"""
    try:
        files = sorted(Path(directory).glob(f'*.{extension}'))
        return [str(f) for f in files]
    except Exception as e:
        print(f"Error reading {directory}: {e}")
        return []

def create_mask_video(input_video_path, output_video_path, canvas_size, bbox_position, bbox_size):
    """
    Create mask video where:
    - White areas remain white
    - Non-white areas become black
    - Content is positioned within specified bbox on canvas
    """
    # Load input video
    video_clip = VideoFileClip(input_video_path)

    # Process each frame to convert non-white to black
    def process_frame(frame):
        pil_img = Image.fromarray(frame)
        img_array = np.array(pil_img)

        # Create mask where pixels are not white
        mask = ~(np.all(img_array == [255, 255, 255], axis=-1))

        # Set non-white pixels to black
        img_array[mask] = [0, 0, 0]

        return img_array

    processed_clip = video_clip.fl_image(process_frame)

    # Create white background canvas
    white_bg = ImageClip(np.array(Image.new('RGB', canvas_size, color='white')))
    white_bg = white_bg.set_duration(video_clip.duration)

    # Resize video to fit bbox while maintaining aspect ratio
    bbox_w, bbox_h = bbox_size
    video_w, video_h = processed_clip.size
    ratio = min(bbox_w / video_w, bbox_h / video_h)
    new_w = int(video_w * ratio)
    new_h = int(video_h * ratio)
    resized_video = processed_clip.fx(resize, width=new_w, height=new_h)

    # Calculate position (centered in bbox)
    x, y = bbox_position
    pos_x = x + (bbox_w - new_w) // 2
    pos_y = y + (bbox_h - new_h) // 2

    # Composite video on white background
    final_clip = CompositeVideoClip([
        white_bg,
        resized_video.set_position((pos_x, pos_y))
    ])

    # Write output
    final_clip.write_videofile(
        output_video_path,
        fps=video_clip.fps,
        codec='libx264',
        audio=False,
        threads=4,
        preset='fast',
        bitrate='8000k'
    )

    # Close clips
    video_clip.close()
    processed_clip.close()
    final_clip.close()

def create_placed_video(input_video_path, output_video_path, canvas_size, bbox_position, bbox_size):
    """
    Create placed video where:
    - White areas become gray
    - Non-white areas remain unchanged
    - Content is positioned within specified bbox on gray canvas
    """
    # Load input video
    video_clip = VideoFileClip(input_video_path)

    # Process each frame to convert white to gray
    def process_frame(frame):
        pil_img = Image.fromarray(frame)
        img_array = np.array(pil_img)

        # Create mask where pixels are white
        mask = np.all(img_array == [255, 255, 255], axis=-1)

        # Set white pixels to gray
        img_array[mask] = [128, 128, 128]

        return img_array

    processed_clip = video_clip.fl_image(process_frame)

    # Create gray background canvas
    gray_bg = ImageClip(np.array(Image.new('RGB', canvas_size, color='#808080')))
    gray_bg = gray_bg.set_duration(video_clip.duration)

    # Resize video to fit bbox while maintaining aspect ratio
    bbox_w, bbox_h = bbox_size
    video_w, video_h = processed_clip.size
    ratio = min(bbox_w / video_w, bbox_h / video_h)
    new_w = int(video_w * ratio)
    new_h = int(video_h * ratio)
    resized_video = processed_clip.fx(resize, width=new_w, height=new_h)

    # Calculate position (centered in bbox)
    x, y = bbox_position
    pos_x = x + (bbox_w - new_w) // 2
    pos_y = y + (bbox_h - new_h) // 2

    # Composite video on gray background
    final_clip = CompositeVideoClip([
        gray_bg,
        resized_video.set_position((pos_x, pos_y))
    ])

    # Write output
    final_clip.write_videofile(
        output_video_path,
        fps=video_clip.fps,
        codec='libx264',
        audio_codec='aac',
        threads=4,
        preset='fast',
        bitrate='8000k'
    )

    # Close clips
    video_clip.close()
    processed_clip.close()
    final_clip.close()

def process_all_videos(input_folder, mask_output_folder, placed_output_folder):
    """
    Process all MP4 files in the input folder
    """
    # Create output folders if they don't exist
    os.makedirs(mask_output_folder, exist_ok=True)
    os.makedirs(placed_output_folder, exist_ok=True)

    # Get sorted list of MP4 files
    video_files = get_sorted_files(input_folder, 'mp4')

    # Canvas settings
    canvas_size = (1920, 1080)
    bbox_position = (850, 100)
    bbox_size = (200, 200)

    for i, input_video in enumerate(video_files):
        # Generate file number with leading zeros
        file_num = f"{i:02d}"

        # Generate output filenames
        mask_output = os.path.join(mask_output_folder, f"{file_num}_mask.mp4")
        placed_output = os.path.join(placed_output_folder, f"{file_num}_placed.mp4")

        print(f"Processing {input_video} -> {mask_output} and {placed_output}")

        # Create mask video (white remains, non-white becomes black)
        create_mask_video(input_video, mask_output, canvas_size, bbox_position, bbox_size)

        # Create placed video (white becomes gray, non-white remains)
        create_placed_video(input_video, placed_output, canvas_size, bbox_position, bbox_size)

if __name__ == "__main__":
    # Configuration parameters
    input_folder = "Xiang_Float_After_Tomorrow_Head_SPLITED_placed_videos_BiRefNet"
    mask_output_folder = "Xiang_Float_After_Tomorrow_Head_SPLITED_BiRefNet_mask_videos"
    placed_output_folder = "Xiang_Float_After_Tomorrow_Head_SPLITED_BiRefNet_placed_videos"

    # Process all videos
    process_all_videos(input_folder, mask_output_folder, placed_output_folder)


from PIL import Image, ImageDraw
import numpy as np
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
from moviepy.video.fx.all import resize
from pathlib import Path
import os

def get_sorted_files(directory, extension):
    """Return sorted list of files with given extension in directory"""
    try:
        files = sorted(Path(directory).glob(f'*.{extension}'))
        return [str(f) for f in files]
    except Exception as e:
        print(f"Error reading {directory}: {e}")
        return []

def create_mask_video(input_video_path, mask_video_path, output_video_path, canvas_size, bbox_position, bbox_size):
    """
    Create mask video using reference mask to determine black areas:
    - Where mask is white -> output black
    - Where mask is black -> output white
    - Content is positioned within specified bbox on white canvas
    """
    # Load input video and corresponding mask video
    video_clip = VideoFileClip(input_video_path)
    mask_clip = VideoFileClip(mask_video_path)

    # Make sure video and mask have same duration
    duration = min(video_clip.duration, mask_clip.duration)
    video_clip = video_clip.subclip(0, duration)
    mask_clip = mask_clip.subclip(0, duration)

    # Create a function that processes frames with proper timing
    def make_frame(t):
        video_frame = video_clip.get_frame(t)
        mask_frame = mask_clip.get_frame(t)
        return process_frame(video_frame, mask_frame)

    # Process each frame using the mask
    def process_frame(frame, mask_frame):
        # Convert frames to numpy arrays
        video_array = np.array(frame)
        mask_array = np.array(mask_frame)

        # Create output frame (black by default)
        output = np.zeros_like(video_array)

        # Where mask is black (or near black), make output white (inverted)
        black_threshold = 10
        is_black = np.all(mask_array <= black_threshold, axis=-1)
        output[is_black] = [255, 255, 255]

        return output

    # Create processed clip
    processed_clip = video_clip.fl(lambda gf, t: make_frame(t))

    # Create white background canvas
    white_bg = ImageClip(np.array(Image.new('RGB', canvas_size, color='white')))
    white_bg = white_bg.set_duration(duration)

    # Resize video to fit bbox while maintaining aspect ratio
    bbox_w, bbox_h = bbox_size
    video_w, video_h = processed_clip.size
    ratio = min(bbox_w / video_w, bbox_h / video_h)
    new_w = int(video_w * ratio)
    new_h = int(video_h * ratio)
    resized_video = processed_clip.fx(resize, width=new_w, height=new_h)

    # Calculate position (centered in bbox)
    x, y = bbox_position
    pos_x = x + (bbox_w - new_w) // 2
    pos_y = y + (bbox_h - new_h) // 2

    # Composite video on white background
    final_clip = CompositeVideoClip([
        white_bg,
        resized_video.set_position((pos_x, pos_y))
    ])

    # Write output
    final_clip.write_videofile(
        output_video_path,
        fps=video_clip.fps,
        codec='libx264',
        audio=False,
        threads=4,
        preset='fast',
        bitrate='8000k'
    )

    # Close clips
    video_clip.close()
    mask_clip.close()
    processed_clip.close()
    final_clip.close()


def create_placed_video(input_video_path, mask_video_path, output_video_path, canvas_size, bbox_position, bbox_size):
    """
    Create placed video using reference mask to determine black areas:
    - Where mask is white -> keep original colors
    - Where mask is black -> output gray
    - Content is positioned within specified bbox on gray canvas
    """
    # Load input video and corresponding mask video
    video_clip = VideoFileClip(input_video_path)
    mask_clip = VideoFileClip(mask_video_path)

    # Make sure video and mask have same duration
    duration = min(video_clip.duration, mask_clip.duration)
    video_clip = video_clip.subclip(0, duration)
    mask_clip = mask_clip.subclip(0, duration)

    # Create a function that processes frames with proper timing
    def make_frame(t):
        video_frame = video_clip.get_frame(t)
        mask_frame = mask_clip.get_frame(t)
        return process_frame(video_frame, mask_frame)

    # Process each frame using the mask
    def process_frame(frame, mask_frame):
        # Convert frames to numpy arrays
        video_array = np.array(frame)
        mask_array = np.array(mask_frame)

        # Create output frame (copy of original)
        output = video_array.copy()

        # Where mask is black (or near black), make output gray
        black_threshold = 10
        is_black = np.all(mask_array <= black_threshold, axis=-1)
        output[is_black] = [128, 128, 128]

        return output

    # Create processed clip
    processed_clip = video_clip.fl(lambda gf, t: make_frame(t))

    # Create gray background canvas
    gray_bg = ImageClip(np.array(Image.new('RGB', canvas_size, color='#808080')))
    gray_bg = gray_bg.set_duration(duration)

    # Resize video to fit bbox while maintaining aspect ratio
    bbox_w, bbox_h = bbox_size
    video_w, video_h = processed_clip.size
    ratio = min(bbox_w / video_w, bbox_h / video_h)
    new_w = int(video_w * ratio)
    new_h = int(video_h * ratio)
    resized_video = processed_clip.fx(resize, width=new_w, height=new_h)

    # Calculate position (centered in bbox)
    x, y = bbox_position
    pos_x = x + (bbox_w - new_w) // 2
    pos_y = y + (bbox_h - new_h) // 2

    # Composite video on gray background
    final_clip = CompositeVideoClip([
        gray_bg,
        resized_video.set_position((pos_x, pos_y))
    ])

    # Write output
    final_clip.write_videofile(
        output_video_path,
        fps=video_clip.fps,
        codec='libx264',
        audio_codec='aac',
        threads=4,
        preset='fast',
        bitrate='8000k'
    )

    # Close clips
    video_clip.close()
    mask_clip.close()
    processed_clip.close()
    final_clip.close()

def process_all_videos(input_folder, mask_folder, mask_output_folder, placed_output_folder):
    """
    处理输入文件夹中的所有MP4文件
    """
    # 创建输出文件夹
    os.makedirs(mask_output_folder, exist_ok=True)
    os.makedirs(placed_output_folder, exist_ok=True)

    # 获取排序后的MP4文件列表
    video_files = get_sorted_files(input_folder, 'mp4')
    mask_files = get_sorted_files(mask_folder, 'mp4')

    # 确保视频和mask文件数量匹配
    if len(video_files) != len(mask_files):
        print(f"Warning: Number of videos ({len(video_files)}) doesn't match number of masks ({len(mask_files)})")
        min_files = min(len(video_files), len(mask_files))
        video_files = video_files[:min_files]
        mask_files = mask_files[:min_files]

    # 画布大小 (宽, 高)
    canvas_size = (1920, 1080)

    # bbox位置和大小 (x, y), (宽, 高)
    bbox_position = (850, 100)  # 中心位置示例
    bbox_size = (200, 200)

    for i, (input_video, mask_video) in enumerate(zip(video_files, mask_files)):
        # 生成序号，保持前导零
        file_num = f"{i:02d}"

        # 生成输出文件名
        mask_output = os.path.join(mask_output_folder, f"{file_num}_mask.mp4")
        placed_output = os.path.join(placed_output_folder, f"{file_num}_placed.mp4")

        print(f"Processing {input_video} with mask {mask_video}")
        print(f"  -> {mask_output} and {placed_output}")

        # 创建mask视频 (使用参考mask确定黑色区域)
        create_mask_video(input_video, mask_video, mask_output, canvas_size, bbox_position, bbox_size)

        # 创建placed视频 (使用参考mask确定黑色区域)
        create_placed_video(input_video, mask_video, placed_output, canvas_size, bbox_position, bbox_size)

if __name__ == "__main__":
    # 配置参数
    input_folder = "Xiang_Float_After_Tomorrow_Head_SPLITED_RMBG"
    mask_folder = "Xiang_Float_After_Tomorrow_Head_SPLITED_RMBG_MASK"
    mask_output_folder = "Xiang_Float_After_Tomorrow_Head_SPLITED_RMBG_mask_videos"
    placed_output_folder = "Xiang_Float_After_Tomorrow_Head_SPLITED_RMBG_placed_videos"

    # 处理所有视频
    process_all_videos(input_folder, mask_folder, mask_output_folder, placed_output_folder)
