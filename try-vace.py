conda activate system
pip install comfy-cli

comfy --here install

cd ComfyUI/custom_nodes
git clone https://github.com/Chaoses-Ib/ComfyScript.git
cd ComfyScript
pip install -e ".[default,cli]"
pip uninstall aiohttp
pip install -U aiohttp

Models autodownload to /ComfyUI/models/float
Or
https://drive.google.com/file/d/1rvWuM12cyvNvBQNCLmG4Fr2L1rpjQBF0/view?usp=sharing


from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
    with Workflow():
    image, _ = LoadImage('sam_altman_512x512.jpg')
    audio = LoadAudio('aud-sample-vs-1.wav')
    float_pipe = LoadFloatModels('float.pth')
    fpsfloat = PrimitiveFloat(25)
    image = FloatProcess(image, audio, float_pipe, 2, 1, 1, fpsfloat, 'none', True, 982045898717762)
    _ = VHSVideoCombine(image, fpsfloat, 0, 'AnimateDiff', 'video/nvenc_h264-mp4', False, True, audio, None, None)

vim run_head.py

import os
import time
import pandas as pd
import subprocess
from pathlib import Path
from itertools import zip_longest

# Configuration
SEEDS = [42]
ADJUST_IMAGE_DIR = 'adjust_images'  # Directory containing PNG images
AUDIO_DIR = 'After_Tomorrow_SPLITED_Fix'  # Directory containing MP3 files
OUTPUT_DIR = 'ComfyUI/output'
PYTHON_PATH = '/environment/miniconda3/envs/system/bin/python'

def get_sorted_files(directory, extension):
    """Return sorted list of files with given extension in directory"""
    try:
        files = sorted(Path(directory).glob(f'*.{extension}'))
        return [str(f) for f in files]
    except Exception as e:
        print(f"Error reading {directory}: {e}")
        return []

def get_latest_output_count():
    """Return the number of MP4 files in the output directory"""
    try:
        return len(list(Path(OUTPUT_DIR).glob('*audio.mp4')))
    except:
        return 0

def wait_for_new_output(initial_count):
    """Wait until a new MP4 file appears in the output directory"""
    timeout = 3000  # 5 minutes timeout
    start_time = time.time()

    while time.time() - start_time < timeout:
        current_count = get_latest_output_count()
        if current_count > initial_count:
            time.sleep(1)  # additional 1 second delay
            return True
        time.sleep(0.5)
    return False

def generate_script(image_path, audio_path, seed):
    """Generate the script with the given image/audio pair"""
    script_content = f"""from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *

with Workflow():
    image, _ = LoadImage('{image_path}')
    audio = LoadAudio('{audio_path}')
    float_pipe = LoadFloatModels('float.pth')
    fpsfloat = PrimitiveFloat(25)
    image = FloatProcess(image, audio, float_pipe, 2, 1, 1, fpsfloat, 'none', True, {seed})
    _ = VHSVideoCombine(image, fpsfloat, 0, 'AnimateDiff', 'video/nvenc_h264-mp4', False, True, audio, None, None)
"""
    return script_content

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get sorted lists of images and audio files
    image_files = get_sorted_files(ADJUST_IMAGE_DIR, 'png')
    audio_files = get_sorted_files(AUDIO_DIR, 'mp3')

    if not image_files or not audio_files:
        print("No image or audio files found. Exiting.")
        return

    # Pair them up (will stop when the shorter list is exhausted)
    file_pairs = list(zip(image_files, audio_files))

    # Main generation loop
    for seed in SEEDS:
        for image_path, audio_path in file_pairs:
            # Generate script
            script = generate_script(image_path, audio_path, seed)

            # Write script to file
            with open('run_float_process.py', 'w') as f:
                f.write(script)

            # Get current output count before running
            initial_count = get_latest_output_count()

            # Run the script
            print(f"Generating video with image: {image_path}, audio: {audio_path}, seed: {seed}")
            subprocess.run([PYTHON_PATH, 'run_float_process.py'])

            # Wait for new output
            if not wait_for_new_output(initial_count):
                print("Timeout waiting for new output. Continuing to next generation.")
                continue

if __name__ == "__main__":
    main()


git clone https://github.com/ali-vilab/VACE.git && cd VACE
pip install torch==2.5.1 torchvision==0.20.1
pip install -r requirements.txt
pip install wan@git+https://github.com/Wan-Video/Wan2.1  # If you want to use Wan2.1-based VACE.
#pip install ltx-video@git+https://github.com/Lightricks/LTX-Video@ltx-video-0.9.1 sentencepiece --no-deps # If you want to use LTX-Video-0.9-based VACE. It may conflict with Wan.

python vace/vace_wan_inference.py --model_name vace-1.3B --src_video <path-to-src-video> --src_mask <path-to-src-mask> --src_ref_images <paths-to-src-ref-images> --prompt <prompt>  # wan

huggingface-cli download  Wan-AI/Wan2.1-VACE-1.3B --local-dir models/Wan2.1-VACE-1.3B --repo-type model

huggingface-cli download ali-vilab/VACE-Benchmark --local-dir benchmarks/VACE-Benchmark --repo-type dataset

python vace/vace_wan_inference.py --model_name vace-1.3B \
 --src_video "benchmarks/VACE-Benchmark/assets/examples/outpainting/src_video.mp4" \
 --src_mask "benchmarks/VACE-Benchmark/assets/examples/outpainting/src_mask.mp4" \
 --prompt "赛博朋克风格，无人机俯瞰视角下的现代西安城墙，镜头穿过永宁门时泛起金色涟漪，城墙砖块化作数据流重组为唐代长安城。周围的街道上流动的人群和飞驰的机械交通工具交织在一起，现代与古代的交融，城墙上的灯光闪烁，形成时空隧道的效果。全息投影技术展现历史变迁，粒子重组特效细腻逼真。大远景逐渐过渡到特写，聚焦于城门特效。"

pip install moviepy==1.0.3

from PIL import Image, ImageDraw
import numpy as np
from moviepy.editor import VideoFileClip, ImageClip
import os

def create_bbox_image(canvas_size, bbox_position, bbox_size):
    """
    创建带有黑色bbox的白色背景图像

    参数:
        canvas_size: (width, height) 整个画布的大小
        bbox_position: (x, y) bbox左上角的位置
        bbox_size: (width, height) bbox的大小
    """
    # 创建白色背景图像
    image = Image.new('RGB', canvas_size, color='white')
    draw = ImageDraw.Draw(image)

    # 绘制黑色bbox
    x, y = bbox_position
    w, h = bbox_size
    draw.rectangle([x, y, x + w, y + h], fill='black')

    return image

def create_static_video(input_video_path, output_video_path, canvas_size, bbox_position, bbox_size):
    """
    创建静态视频

    参数:
        input_video_path: 输入视频路径(用于获取视频长度)
        output_video_path: 输出视频路径
        canvas_size: (width, height) 整个画布的大小
        bbox_position: (x, y) bbox左上角的位置
        bbox_size: (width, height) bbox的大小
    """
    # 创建静态图像
    image = create_bbox_image(canvas_size, bbox_position, bbox_size)

    # 获取输入视频的时长
    with VideoFileClip(input_video_path) as video:
        duration = video.duration

    # 将图像转换为视频
    image_array = np.array(image)  # 将PIL图像转换为numpy数组
    clip = ImageClip(image_array).set_duration(duration)

    # 写入输出视频文件
    clip.write_videofile(output_video_path, fps=24, codec='libx264', audio=False)

    print(f"静态视频已创建: {output_video_path}")

# 示例用法
if __name__ == "__main__":
    # 配置参数
    input_video = "AnimateDiff_00011-audio.mp4"  # 替换为你的输入视频路径
    output_video = "mask.mp4"

    # 画布大小 (宽, 高)
    canvas_size = (1920, 1080)

    # bbox位置和大小 (x, y), (宽, 高)
    bbox_position = (850, 100)  # 中心位置示例
    bbox_size = (200, 200)

    # 创建静态视频
    create_static_video(input_video, output_video, canvas_size, bbox_position, bbox_size)


from PIL import Image, ImageDraw, ImageOps
import numpy as np
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
from moviepy.video.fx.all import resize

def embed_video_in_bbox(input_video_path, output_video_path, canvas_size, bbox_position, bbox_size):
    """
    将输入视频嵌入到bbox区域内，边缘用灰色填充

    参数:
        input_video_path: 输入视频路径
        output_video_path: 输出视频路径
        canvas_size: (width, height) 整个画布的大小
        bbox_position: (x, y) bbox左上角的位置
        bbox_size: (width, height) bbox的大小
    """
    # 创建灰色背景的静态图像
    def create_gray_background():
        return Image.new('RGB', canvas_size, color='#808080')  # 灰色背景

    # 加载输入视频
    video_clip = VideoFileClip(input_video_path)

    # 调整视频大小以适应bbox
    bbox_w, bbox_h = bbox_size
    video_w, video_h = video_clip.size

    # 计算保持宽高比的缩放比例
    ratio = min(bbox_w / video_w, bbox_h / video_h)
    new_w = int(video_w * ratio)
    new_h = int(video_h * ratio)

    # 调整视频大小
    resized_video = video_clip.fx(resize, width=new_w, height=new_h)

    # 计算视频在bbox中的位置(居中)
    x, y = bbox_position
    pos_x = x + (bbox_w - new_w) // 2
    pos_y = y + (bbox_h - new_h) // 2

    # 创建灰色背景的静态视频
    gray_bg = ImageClip(np.array(create_gray_background())).set_duration(video_clip.duration)

    # 合成视频
    final_clip = CompositeVideoClip([
        gray_bg,
        resized_video.set_position((pos_x, pos_y))
    ])

    # 写入输出视频文件
    final_clip.write_videofile(
        output_video_path,
        fps=video_clip.fps,
        codec='libx264',
        audio_codec='aac',
        threads=4,
        preset='fast',
        bitrate='8000k'
    )

    # 关闭视频剪辑
    video_clip.close()
    final_clip.close()

    print(f"新视频已生成: {output_video_path}")

# 示例用法
if __name__ == "__main__":
    # 配置参数
    input_video = "AnimateDiff_00011-audio.mp4"  # 替换为你的输入视频路径
    output_video = "AnimateDiff_00011-audio_placed.mp4"

    # 画布大小 (宽, 高)
    canvas_size = (1920, 1080)

    # bbox位置和大小 (x, y), (宽, 高)
    bbox_position = (850, 100)  # 中心位置示例
    bbox_size = (200, 200)

    # 将视频嵌入bbox
    embed_video_in_bbox(input_video, output_video, canvas_size, bbox_position, bbox_size)

python vace/vace_wan_inference.py --model_name vace-1.3B \
 --src_video "AnimateDiff_00011-audio_placed.mp4" \
 --src_mask "mask.mp4" \
 --prompt "一个戴眼镜男青年在自然风光美丽的无人室外唱歌。"

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

def create_bbox_image(canvas_size, bbox_position, bbox_size):
    """
    创建带有黑色bbox的白色背景图像
    """
    image = Image.new('RGB', canvas_size, color='white')
    draw = ImageDraw.Draw(image)
    x, y = bbox_position
    w, h = bbox_size
    draw.rectangle([x, y, x + w, y + h], fill='black')
    return image

def create_static_video(input_video_path, output_video_path, canvas_size, bbox_position, bbox_size):
    """
    创建静态视频
    """
    image = create_bbox_image(canvas_size, bbox_position, bbox_size)

    with VideoFileClip(input_video_path) as video:
        duration = video.duration

    image_array = np.array(image)
    clip = ImageClip(image_array).set_duration(duration)
    clip.write_videofile(output_video_path, fps=24, codec='libx264', audio=False)

def create_gray_background(canvas_size):
    """创建灰色背景图像"""
    return Image.new('RGB', canvas_size, color='#808080')

def embed_video_in_bbox(input_video_path, output_video_path, canvas_size, bbox_position, bbox_size):
    """
    将输入视频嵌入到bbox区域内，边缘用灰色填充
    """
    # 加载输入视频
    video_clip = VideoFileClip(input_video_path)

    # 调整视频大小以适应bbox
    bbox_w, bbox_h = bbox_size
    video_w, video_h = video_clip.size

    # 计算保持宽高比的缩放比例
    ratio = min(bbox_w / video_w, bbox_h / video_h)
    new_w = int(video_w * ratio)
    new_h = int(video_h * ratio)

    # 调整视频大小
    resized_video = video_clip.fx(resize, width=new_w, height=new_h)

    # 计算视频在bbox中的位置(居中)
    x, y = bbox_position
    pos_x = x + (bbox_w - new_w) // 2
    pos_y = y + (bbox_h - new_h) // 2

    # 创建灰色背景的静态视频
    gray_bg = ImageClip(np.array(create_gray_background(canvas_size))).set_duration(video_clip.duration)

    # 合成视频
    final_clip = CompositeVideoClip([
        gray_bg,
        resized_video.set_position((pos_x, pos_y))
    ])

    # 写入输出视频文件
    final_clip.write_videofile(
        output_video_path,
        fps=video_clip.fps,
        codec='libx264',
        audio_codec='aac',
        threads=4,
        preset='fast',
        bitrate='8000k'
    )

    # 关闭视频剪辑
    video_clip.close()
    final_clip.close()

def process_all_videos(input_folder, mask_output_folder, placed_output_folder):
    """
    处理输入文件夹中的所有MP4文件
    """
    # 创建输出文件夹
    os.makedirs(mask_output_folder, exist_ok=True)
    os.makedirs(placed_output_folder, exist_ok=True)

    # 获取排序后的MP4文件列表
    video_files = get_sorted_files(input_folder, 'mp4')

    # 画布大小 (宽, 高)
    canvas_size = (1920, 1080)

    # bbox位置和大小 (x, y), (宽, 高)
    bbox_position = (850, 100)  # 中心位置示例
    bbox_size = (200, 200)

    for i, input_video in enumerate(video_files):
        # 生成序号，保持前导零
        file_num = f"{i:02d}"

        # 生成输出文件名
        mask_output = os.path.join(mask_output_folder, f"{file_num}_mask.mp4")
        placed_output = os.path.join(placed_output_folder, f"{file_num}_placed.mp4")

        print(f"Processing {input_video} -> {mask_output} and {placed_output}")

        # 创建mask视频
        create_static_video(input_video, mask_output, canvas_size, bbox_position, bbox_size)

        # 创建placed视频
        embed_video_in_bbox(input_video, placed_output, canvas_size, bbox_position, bbox_size)

if __name__ == "__main__":
    # 配置参数
    input_folder = "Xiang_Float_After_Tomorrow_Head_SPLITED"
    mask_output_folder = "Xiang_Float_After_Tomorrow_Head_SPLITED_mask_videos"
    placed_output_folder = "Xiang_Float_After_Tomorrow_Head_SPLITED_placed_videos"

    # 处理所有视频
    process_all_videos(input_folder, mask_output_folder, placed_output_folder)

en_prompt = ["Golden sunlight spills through swaying leaves,   Whispering breezes dance on rippling streams.   Moss-kissed stones hum with ancient tales,   While silver dewdrops cling to emerald veils.    Cloud shadows drift across sun-warmed hills,   Crickets weave songs through twilight's thrills.   Moonbeams trace patterns on silent ponds,   Where water lilies dream their pale blonde bonds.    Autumn's brush paints the maple's sigh,   Scarlet whispers against cobalt sky.   Frost-kissed branches etch crystal lace,   As winter's breath leaves its fleeting trace.    Dawn's first light gilds the spider's thread,   Pearls of morning on gossamer spread.   Endless cycles of earth's slow turn,   Seasons waltzing at nature's stern.",
 "The wind whispers through the trees,   Light dances on the rippling stream,   Golden leaves drift in autumn's breeze,   Silent clouds paint dreams.    Moonlight spills on the quiet lake,   Stars blink in the velvet sky,   Mist rises where the willows wake,   Night breathes a soft sigh.",
 'A starless night sky stretches endlessly, Darkness blankets the earth in silent embrace, No celestial sparks pierce the velvet void, Only the wind whispers through empty space.',
 'Silence lingers in the air, Unspoken words dissolve like mist, Empty spaces between the trees, Where echoes fade into the breeze.',
 'Golden sands slip through the breeze,   Whispering leaves let moments ease.   Shimmering streams of time flow light,   Dancing sunbeams fade from sight.',
 'The wind whispers through empty branches,   Dawn light hesitates on distant hills,   A lone leaf drifts across still waters,   Morning mist veils the silent fields.',
 'Memories weave through the branches, Whispering leaves hold the past, Golden light lingers between the trees, Silent breezes carry echoes.',
 'The wind whispers through swaying leaves, Sunlight melts into golden hues, A fleeting warmth brushes the petals, Dew trembles on morning grass.',
 'Mist swirls in endless cycles, Shadows dance in timeless rings, Dewdrops tremble on bending reeds, Twilight whispers through swaying leaves.',
 'Breeze stirs the trembling leaves, Sunlight flickers through swaying branches, Shadows dance in restless patterns, A silent pulse quickens through the air.',
 "Dappled sunlight filters through the leaves, Two shadows merge beneath the swaying boughs. The creek murmurs secrets to the stones, While golden pollen drifts on summer's breath.",
 "Golden light lingers in the air,   Soft petals drift on evening's breath.",
 'The gentle breeze empties the tranquil air, Sunlight pours through the hollowed leaves.',
 'A gentle breeze whispers through the leaves, Sunlight dances on rippling waters, Petals tremble with morning dew, Silent moments weave through the trees.',
 "Winds whisper through the trees,   Moonlight lingers on the leaves.   No questions rise with morning's glow,   Only stars that softly know.",
 "Bright mountains and clear waters   Cannot match your radiance    （注：根据要求，改写为纯自然描写，避免人称代词和人物动作）    改写版本：   Sunlit peaks and crystal streams   Pale before the dawn's first gleam",
 'Golden sunlight lingers on the winding path Autumn leaves dance in whispering breezes Two shadows merge upon the cobblestones Twilight paints the horizon in fading hues',
 'The fading light lingers low, A breeze stirs the bending reeds. Shadows stretch across the silent path, As twilight deepens in the trees.',
 'A starless night sky stretches endlessly, Darkness blankets the earth in silent embrace, No celestial sparks pierce the velvet expanse, Only the void whispers to the sleeping land.',
 'Silence lingers in the air, Unspoken words dissolve like mist. Empty spaces between the trees, Where echoes fade without reply.',
 "Promises slip through the cracks of time,   Like sand through fingers, lost in the wind's rhyme.",
 "The wind whispers through empty branches, Distant stars flicker in silent hesitation, A lone leaf trembles on the water's surface, Moonlight pools upon untouched ground.",
 'Memories weave through the whispering trees, Shadows linger where sunlight once danced. The river murmurs old melodies, While autumn leaves drift in silent trance.',
 'The wind whispers through swaying leaves,   Sunlight melts into golden hues,   A fleeting touch of warmth upon the petals,   Dew trembles on morning grass.',
 'The mist swirls in endless cycles, Shadows flicker through the veiled light, Seasons turn without beginning or end, A silent dance of fog and fading glow.',
 'The breeze quickens its restless dance, Leaves tremble without rhythm, Sunlight flickers through swaying branches, A silent storm stirs the air.',
 "Do the golden rays of dawn suffice to frame love's fleeting glow?   Can the whispering leaves alone compose the ballad of the spring?",
 'Golden light lingers in the air   Petals drift where laughter once bloomed',
 'The gentle breeze empties the tranquil air',
 'A gentle breeze whispers through the leaves, Golden sunlight dances on trembling grass, Morning dew glistens on petals unfolding, Time lingers in the fragrance of blooming flowers.',
 "Winds whisper through the trees,   Moonlight lingers on the leaves.   No questions asked by the flowing stream,   No answers given by dawn's first gleam.    Shadows dance without a care,   Stars glow softly in the air.   Tomorrow's sun may rise or fade,   Tonight the night in silence laid.",
 'Bright mountains and clear streams   pale in comparison to your radiance.',
 'Golden sunlight lingers on the winding path Autumn leaves dance in the whispering breeze Two shadows merge beneath the ancient oak Twilight descends as the river flows endlessly',
 "The fading light lingers in the air, Whispering leaves hesitate in twilight's glow, A single breeze drifts through the silent trees, As shadows stretch long across the quiet earth.",
 'Distant clouds drift across the endless blue, Sunlight dances upon the rippling stream. Two leaves tremble in the autumn breeze, Their shadows merging where the waters gleam.',
 'Golden light lingers among swaying leaves.   A gentle breeze carries whispers through the trees.',
 'The gentle breeze caresses the empty air, Sunlight pours through translucent leaves, A tranquil stream reflects the boundless sky.',
 'A fleeting moment of warmth, Sunlight dancing on dewdrops, Breezes whispering through leaves.',
 'Winds whisper through the silent trees,   Moonlight spills across the sleeping earth,   Dewdrops tremble on trembling leaves,   Stars flicker in the boundless dark.    No questions linger in the quiet air,   No answers drift with the passing clouds,   Only the endless dance of night and dawn,   Flowing like an eternal river.',
 'Sunlit peaks and crystal streams,   no sight more fair than these.',
 'The golden sunlight lingers on the winding path, Two shadows merge as twilight softly falls, Autumn leaves dance upon the silent breeze, A distant river hums its endless song.',
 'The fading light lingers in the air, Shadows stretch across the silent land.',
 'The fading light lingers,   unwilling to turn.']

cn_poems = [
    "一个戴眼镜男青年漫步在金色阳光穿透的枫林间，落叶在他脚下沙沙作响，他对着潺潺溪流歌唱，苔石上的露珠随着他的旋律轻轻颤动",
    "一个戴眼镜男青年站在微风吹拂的溪边，镜片映着粼粼波光，他随着摇曳的芦苇轻声哼唱，惊起几只白鹭掠过翡翠色的水面",
    "一个戴眼镜男青年在无星的夜空下仰头歌唱，黑暗温柔包裹着他的声音，只有旷野的风偶尔捎来几缕回响",
    "一个戴眼镜男青年站在晨雾弥漫的林间空地，未唱完的歌词如薄雾般消散，松针间漏下的阳光为他的歌声镀上金边",
    "一个戴眼镜男青年赤脚走在夕阳染红的沙滩上，他的歌声随着流沙滑落，与退潮的浪花一同消逝在粼粼波光中",
    "一个戴眼镜男青年穿过晨雾笼罩的麦田，草尖的露珠随着他的音调震颤，薄雾为他的歌声蒙上朦胧面纱",
    "一个戴眼镜男青年背靠百年老树歌唱，飘落的黄叶如金色音符环绕，树皮粗糙的纹理记录着他歌声的起伏",
    "一个戴眼镜男青年站在沾满晨露的苜蓿地，他呼出的白气与歌声融入草尖颤抖的露珠，裤管被露水浸透仍不停歌唱",
    "一个戴眼镜男青年在暮色四合的山谷歌唱，紫灰色的暮霭随着他的旋律流转，夜风将歌声送往远方",
    "一个戴眼镜男青年站在白桦林间歌唱，阳光透过树隙在他身上投下光斑，飘落的树皮像褪色的乐谱在他脚边盘旋",
    "一个戴眼镜男青年坐在溪边圆石上歌唱，水流冲刷石头的声响为他伴奏，几片枫叶载着他的音符顺流而下",
    "一个戴眼镜男青年站在野花丛中歌唱，花粉沾满衣袖，晚风将歌声揉碎在摇曳的花影里",
    "一个戴眼镜男青年漫步在空寂的松林歌唱，松针在脚下沙沙作响，穿过树冠的光柱如天然舞台灯光",
    "一个戴眼镜男青年迎着晨风歌唱，露水打湿球鞋，草叶上的露珠随着音调轻轻震颤",
    "一个戴眼镜男青年在星光下自弹自唱，萤火虫画出流动光弧，月亮在云层后静静聆听",
    "一个戴眼镜男青年面对雪山湖泊放歌，山风卷起衣摆，回声在碧蓝湖面荡起细纹",
    "一个戴眼镜男青年在铺满落叶的小径歌唱，脚步激起金红叶浪，脚步声打着自然节拍",
    "一个戴眼镜男青年在暮色中低唱，拉长的影子与芦苇丛融为一体，归巢鸟雀偶尔应和",
    "一个戴眼镜男青年在漆黑旷野歌唱，声音被黑夜温柔包裹，远处传来夜枭的回应",
    "一个戴眼镜男青年站在寂静森林歌唱，松涛声为他和声，风带走回声在树冠间流转",
    "一个戴眼镜男青年对着流逝时光歌唱，阳光如沙漏从指缝溜走，歌声试图留住指间沙",
    "一个戴眼镜男青年在月夜歌唱，落叶在水面写音符，倒影在涟漪中时隐时现",
    "一个戴眼镜男青年沿记忆之河歌唱，老树年轮记录旋律，枯枝在脚下清脆断裂",
    "一个戴眼镜男青年在晨光中歌唱，露珠折射七彩光晕，呼吸在冷空气凝成白雾",
    "一个戴眼镜男青年站在山巅歌唱，云海在脚下翻涌，朝阳给他的轮廓镀上金边",
    "一个戴眼镜男青年在暴雨前歌唱，风扯着衣襟，远处雷声打着低沉节拍",
    "一个戴眼镜男青年站在麦浪中歌唱，麦穗轻扫掌心，蝉鸣为他和声",
    "一个戴眼镜男青年在冰湖畔歌唱，呼出的白气凝结，冰层下传来湖水隐秘回响",
    "一个戴眼镜男青年在银杏道歌唱，金黄落叶如雨点落下，铺就金色地毯",
    "一个戴眼镜男青年站在悬崖歌唱，海风带着咸湿气息，浪花在崖底碎成白色音符",
    "一个戴眼镜男青年在竹林歌唱，新笋破土应和节奏，竹叶沙沙如观众掌声",
    "一个戴眼镜男青年站在向日葵田歌唱，花盘随旋律摆动，蜜蜂嗡嗡编织和弦",
    "一个戴眼镜男青年在枫红山谷歌唱，回声在峭壁碰撞，飘落枫叶如跳动火焰",
    "一个戴眼镜男青年在雪原歌唱，脚印绵延成五线谱，呼出的白气在阳光下闪烁",
    "一个戴眼镜男青年在薰衣草田歌唱，紫色波浪起伏，芬芳随歌声飘远",
    "一个戴眼镜男青年站在瀑布边歌唱，水雾打湿睫毛，彩虹在水幕间时隐时现",
    "一个戴眼镜男青年在樱花雨中歌唱，粉白花瓣落满肩头，脚下落英如柔软绒毯",
    "一个戴眼镜男青年站在沙漠绿洲歌唱，远处驼铃隐约，棕榈树影为他遮阳",
    "一个戴眼镜男青年在芦苇荡歌唱，芦花如雪片飞舞，野鸭扑翅为他伴奏",
    "一个戴眼镜男青年站在葡萄架下歌唱，藤蔓间光斑游走，熟透葡萄散发甜香",
    "一个戴眼镜男青年在结霜草原歌唱，脚步留下脆响，呼出的白气如飘散音符",
    "一个戴眼镜男青年站在涨潮海滩歌唱，浪花亲吻脚尖，海鸥鸣叫划出弧线",
    "一个戴眼镜男青年在晨露未晞的草原歌唱，蛛网上的露珠折射晨光，如天然舞台灯光"
]

len(en_prompt), len(cn_poems)

import pandas as pd
from datasets import Dataset
Dataset.from_pandas(pd.DataFrame(zip(en_prompt, cn_poems), columns = ["en", "zh"])).push_to_hub(
    "svjack/Xiang_Float_After_Tomorrow_SPLITED_EN_ZH_Caption"
)

vim run_outpainting.py

import os
import subprocess
import pandas as pd
from datasets import load_dataset
from moviepy.editor import VideoFileClip
from pathlib import Path

# 加载数据集
dataset = load_dataset("svjack/Xiang_Float_After_Tomorrow_SPLITED_EN_ZH_Caption")

# 设置文件夹路径
mask_output_folder = "Xiang_Float_After_Tomorrow_Head_SPLITED_mask_videos"
placed_output_folder = "Xiang_Float_After_Tomorrow_Head_SPLITED_placed_videos"

# 获取并排序MP4文件
def get_sorted_mp4_files(directory, extension = "mp4"):
    """Return sorted list of files with given extension in directory"""
    try:
        files = sorted(Path(directory).glob(f'*.{extension}'))
        return [str(f) for f in files]
    except Exception as e:
        print(f"Error reading {directory}: {e}")
        return []

mask_files = get_sorted_mp4_files(mask_output_folder)
placed_files = get_sorted_mp4_files(placed_output_folder)

print(len(mask_files), len(placed_files))

# 确保文件数量匹配
assert len(mask_files) == len(placed_files), "Mask and placed video counts don't match"
assert len(mask_files) == len(dataset['train']), "Video counts don't match dataset size"

# 辅助函数：获取视频时长
def get_video_duration(video_path):
    """使用moviepy获取视频时长（秒）"""
    try:
        with VideoFileClip(video_path) as video:
            return video.duration
    except Exception as e:
        print(f"获取视频时长失败: {video_path}, 错误: {str(e)}")
        return 0

# 辅助函数：运行命令并记录日志
def run_command(cmd, log_file=None):
    """运行命令并捕获输出日志"""
    print(f"执行命令: {cmd}")

    if log_file:
        with open(log_file, 'a') as f:
            process = subprocess.Popen(cmd, shell=True,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,
                                     universal_newlines=True)

            for line in process.stdout:
                print(line.strip())  # 打印到控制台
                f.write(line)        # 写入日志文件

            process.wait()
            return process.returncode
    else:
        result = subprocess.run(cmd, shell=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              universal_newlines=True)
        print(result.stdout)
        if result.stderr:
            print("错误输出:")
            print(result.stderr)
        return result.returncode

# 创建结果记录
results = []
processed_files = []
unprocessed_files = []

# 创建日志目录
os.makedirs("processing_logs", exist_ok=True)

for idx, (mask_file, placed_file, zh_caption) in enumerate(zip(mask_files, placed_files, dataset['train']['zh'])):
    # 获取视频路径和时长
    #video_path = os.path.join(placed_output_folder, placed_file)
    duration = get_video_duration(placed_file)

    log_file = f"processing_logs/process_{idx}_{os.path.splitext(placed_file)[0]}.log"

    if duration <= 10:
        # 构建命令
        cmd = f'python vace/vace_wan_inference.py --model_name vace-1.3B ' \
              f'--src_video "{placed_file}" ' \
              f'--src_mask "{mask_file}" ' \
              f'--prompt "{zh_caption}"'

        # 执行命令并记录日志
        return_code = run_command(cmd, log_file = None)

        results.append({
            'index': idx,
            'mask_file': mask_file,
            'placed_file': placed_file,
            'zh_caption': zh_caption,
            'processed': True,
            'duration': duration,
            'return_code': return_code,
            'log_file': log_file
        })
        processed_files.append(placed_file)
    else:
        results.append({
            'index': idx,
            'mask_file': mask_file,
            'placed_file': placed_file,
            'zh_caption': zh_caption,
            'processed': False,
            'duration': duration,
            'return_code': None,
            'log_file': None
        })
        unprocessed_files.append(placed_file)

# 保存结果到CSV
df = pd.DataFrame(results)
df.to_csv('video_processing_report.csv', index=False)

print(f"\n处理完成。共处理了 {len(processed_files)} 个文件，跳过了 {len(unprocessed_files)} 个文件。")
print(f"处理报告已保存到 video_processing_report.csv")
print(f"详细日志保存在 processing_logs 目录")



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

#!/bin/bash

# 源目录
src_dir="results/vace-1.3B"

# 目标目录
# 目标目录（前缀已修改）
dest_dir1="Xiang_Float_After_Tomorrow_Head_SPLITED_VACE_OutPainting_videos"  # 存放out_video.mp4
dest_dir2="Xiang_Float_After_Tomorrow_Head_SPLITED_VACE_OutPainting_masks"   # 存放src_mask.mp4
dest_dir3="Xiang_Float_After_Tomorrow_Head_SPLITED_VACE_OutPainting_srcs"    # 存放src_video.mp4

# 创建目标目录
mkdir -p {$dest_dir1,$dest_dir2,$dest_dir3}

# 获取按时间排序的文件夹列表（从旧到新）
sorted_folders=$(ls -1tr $src_dir | tail -40)

counter=0

# 遍历最晚的40个文件夹
for folder in $sorted_folders; do
    # 格式化序号为5位数
    seq_num=$(printf "%05d" $counter)

    # 拷贝并重命名文件
    cp "$src_dir/$folder/out_video.mp4" "$dest_dir1/${seq_num}.mp4"
    cp "$src_dir/$folder/src_mask.mp4" "$dest_dir2/${seq_num}.mp4"
    cp "$src_dir/$folder/src_video.mp4" "$dest_dir3/${seq_num}.mp4"

    echo "已处理: $folder -> $seq_num.mp4"
    ((counter++))
done

echo "完成！共处理了$counter个文件夹"

python vace/vace_wan_inference.py --size "832*480" --prompt "两只手互相牵着，远处狂风巨浪的大海，镜头缓缓推进，一艘渺小的帆船在汹涌的波涛中挣扎漂荡。海面上白沫翻滚，帆船时隐时现，仿佛随时可能被巨浪吞噬。天空乌云密布，雷声轰鸣，海鸥在空中盘旋尖叫。帆船上的人们紧紧抓住缆绳，努力保持平衡。画面风格写实，充满紧张和动感。近景特写，强调风浪的冲击力和帆船的摇晃"

python vace/vace_wan_inference.py --size "832*480" --prompt "夕阳西下的金色海滩，镜头缓缓拉远，温暖的阳光洒在波光粼粼的海面上，细浪轻柔地拍打着岸边。远处海天相接处泛起粉紫色的晚霞，几只海鸥优雅地滑翔。沙滩上留下潮水退去的波纹痕迹，整个画面充满浪漫温馨的氛围。画面风格写实，色调温暖柔和。"
python vace/vace_wan_inference.py --size "832*480" --prompt "春日清晨的樱花林，镜头缓缓推进，粉白的花瓣随风飘舞，如同温柔的雪。晨雾轻柔地笼罩着树林，阳光透过花枝形成斑驳的光影。一条铺满落花的小径蜿蜒伸向远方，空气中仿佛弥漫着甜蜜的气息。画面风格唯美，色彩柔和。"
python vace/vace_wan_inference.py --size "832*480" --prompt "秋日黄昏的枫叶谷，镜头缓缓摇过，漫山遍野的红叶在夕阳下如火般燃烧。一条清澈的小溪穿过山谷，水面上漂浮着红叶，倒映着绚丽的天空。微风拂过，树叶沙沙作响，整个场景充满诗意的浪漫。画面风格写实，色彩浓郁。"
python vace/vace_wan_inference.py --size "832*480" --prompt "夏夜星空下的薰衣草田，镜头缓缓上升，紫色的花海在微风中轻轻摇曳，散发出淡淡的香气。银河横贯夜空，繁星点点，远处有萤火虫飞舞。月光温柔地洒在花田上，营造出梦幻般的氛围。画面风格浪漫，色调神秘。"
python vace/vace_wan_inference.py --size "832*480" --prompt "冬日雪后的松树林，镜头缓缓推进，白雪覆盖的树枝在阳光下闪闪发光。远处山峰被晨雾笼罩，阳光透过云层形成神圣的光束。雪地上有动物留下的足迹，整个场景纯净而宁静。画面风格清新，色调冷冽中带着温暖。"
python vace/vace_wan_inference.py --size "832*480" --prompt "雨季的江南水乡，镜头缓缓平移，细雨如丝般落下，在水面激起无数涟漪。古老的石桥倒映在水中，两岸白墙黛瓦的民居被雨水洗得发亮。柳枝轻拂水面，远处有乌篷船缓缓驶过。画面风格水墨意境，充满东方韵味。"
python vace/vace_wan_inference.py --size "832*480" --prompt "清晨的向日葵田，镜头缓缓下降，金黄色的花朵齐刷刷地面向初升的太阳。露珠在花瓣上闪烁，微风吹过掀起层层花浪。远处地平线上朝阳刚刚升起，将天空染成橘红色。整个画面充满生机与希望。画面风格明亮欢快。"

After_Tomorrow_SPLITED_Fix/0018_明天过后 - 张杰.mp3
After_Tomorrow_SPLITED_Fix/0043_明天过后 - 张杰.mp3

对下面的代码 当 音频为上面两个mp3 时 video 部分 使用的时 video
最中间帧的静态图片而非视频

from moviepy.editor import *
import os

# 定义文件夹路径
mp3_folder = "After_Tomorrow_SPLITED_Fix"
mp4_folder = "Xiang_Float_After_Tomorrow_Mix_SPLITED_VACE_OutPainting_videos"

# 获取并排序mp3和mp4文件
def get_sorted_files(folder, extension):
    files = [f for f in os.listdir(folder) if f.endswith(extension)]
    files.sort()  # 按字典序排序
    return [os.path.join(folder, f) for f in files]

mp3_files = get_sorted_files(mp3_folder, ".mp3")
mp4_files = get_sorted_files(mp4_folder, ".mp4")

# 确保文件数量匹配
if len(mp3_files) != len(mp4_files):
    raise ValueError(f"文件数量不匹配: {len(mp3_files)}个mp3 vs {len(mp4_files)}个mp4")

# 处理每对(mp3, mp4)文件
processed_clips = []
for mp3_path, mp4_path in zip(mp3_files, mp4_files):
    # 加载音频和视频
    audio = AudioFileClip(mp3_path)
    video = VideoFileClip(mp4_path).without_audio()

    # 计算速度调整因子
    speed_factor = video.duration / audio.duration

    # 调整视频速度以匹配音频长度
    if speed_factor != 1.0:
        video = video.fx(vfx.speedx, speed_factor)

    # 设置音频
    video = video.set_audio(audio)

    # 添加淡入淡出效果(1秒)
    #video = video.crossfadein(0.1).crossfadeout(0.1)

    processed_clips.append(video)

# 合并所有视频片段
final_video = concatenate_videoclips(processed_clips, method="compose")

# 输出最终视频
output_path = "final_output.mp4"
final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

print(f"视频处理完成，已保存到: {output_path}")

https://github.com/svjack/Practical-RIFE

python3 inference_video.py --multi=4 --video=final_output.mp4

python3 inference_video.py --multi=8 --scale=2 --video=final_output.mp4

python3 inference_video.py --multi=32 --scale=4 --video=final_output.mp4
python vace/vace_wan_inference.py --size "832*480" --prompt "冬日雪后的松树林，镜头缓缓推进，白雪覆盖的树枝在阳光下闪闪发光。远处山峰被晨雾笼罩，阳光透过云层形成神圣的光束。雪地上有动物留下的足迹，整个场景纯净而宁静。画面风格清新，色调冷冽中带着温暖。"
python vace/vace_wan_inference.py --size "832*480" --prompt "雨季的江南水乡，镜头缓缓平移，细雨如丝般落下，在水面激起无数涟漪。古老的石桥倒映在水中，两岸白墙黛瓦的民居被雨水洗得发亮。柳枝轻拂水面，远处有乌篷船缓缓驶过。画面风格水墨意境，充满东方韵味。"
python vace/vace_wan_inference.py --size "832*480" --prompt "清晨的向日葵田，镜头缓缓下降，金黄色的花朵齐刷刷地面向初升的太阳。露珠在花瓣上闪烁，微风吹过掀起层层花浪。远处地平线上朝阳刚刚升起，将天空染成橘红色。整个画面充满生机与希望。画面风格明亮欢快。"

git clone https://huggingface.co/spaces/svjack/ReSize-Image-Outpainting

import os
from pathlib import Path
from tqdm import tqdm
from gradio_client import Client, handle_file
from PIL import Image

def process_images():
    # 设置路径
    input_dir = Path("Xiang_Float_After_Tomorrow_Head_SPLITED_First")
    premask_dir = Path("Xiang_Float_After_Tomorrow_Head_SPLITED_First_PreMASK")
    fullbody_dir = Path("Xiang_Float_After_Tomorrow_Head_SPLITED_First_FullBody")

    # 创建输出目录
    premask_dir.mkdir(parents=True, exist_ok=True)
    fullbody_dir.mkdir(parents=True, exist_ok=True)

    # 初始化Gradio客户端
    client = Client("http://localhost:7860/")

    # 获取所有PNG文件并按自然排序
    png_files = sorted(input_dir.glob("*.png"), key=lambda x: int(x.stem.split('_')[0]))

    # 处理每个文件并显示进度条
    for png_file in tqdm(png_files, desc="Processing images"):
        try:
            # 调用API处理文件
            result = client.predict(
                image=handle_file(str(png_file)),
                width=1280,
                height=720,
                overlap_percentage=10,
                num_inference_steps=30,
                resize_option="25%",
                prompt_input="A handsome slim man",
                alignment="Top",
                overlap_left=False,
                overlap_right=False,
                overlap_top=False,
                overlap_bottom=False,
                api_name="/infer"
            )

            # 获取输入文件名（不含扩展名）
            base_name = png_file.stem

            # 处理PreMASK输出（result[0]）
            if result[0]:
                webp_path = Path(result[0])
                output_path = premask_dir / f"{base_name}.png"

                # 转换WEBP为PNG并保存
                with Image.open(webp_path) as img:
                    img.save(output_path, "PNG")

            # 处理FullBody输出（result[1]）
            if result[1]:
                webp_path = Path(result[1])
                output_path = fullbody_dir / f"{base_name}.png"

                # 转换WEBP为PNG并保存
                with Image.open(webp_path) as img:
                    img.save(output_path, "PNG")

        except Exception as e:
            print(f"\n处理文件 {png_file.name} 时出错: {e}")
            continue

if __name__ == "__main__":
    process_images()
    print("\n所有文件处理完成！")

from PIL import Image, ImageDraw

def create_image_with_resized_square(output_path, bg_width=1280, bg_height=720,
                                   square_size=720, resize_percent=25):
    """
    创建白色背景图片并在顶部中间放置缩小后的黑色方块

    参数:
        output_path: 输出图片路径
        bg_width: 背景宽度(默认1280)
        bg_height: 背景高度(默认720)
        square_size: 原始方块大小(默认512)
        resize_percent: 缩小百分比(默认25)
    """
    # 创建白色背景图片
    bg_color = (255, 255, 255)  # 白色
    image = Image.new('RGB', (bg_width, bg_height), bg_color)

    # 计算缩小后的方块尺寸
    new_size = int(square_size * resize_percent / 100)

    # 创建黑色方块
    square_color = (0, 0, 0)  # 黑色
    square = Image.new('RGB', (new_size, new_size), square_color)

    # 计算放置位置（顶部中间）
    position_x = (bg_width - new_size) // 2
    position_y = 0  # 顶部

    # 将方块粘贴到背景上
    image.paste(square, (position_x, position_y))

    # 保存图片
    image.save(output_path)
    print(f"图片已保存到: {output_path}")

# 使用示例
if __name__ == "__main__":
    create_image_with_resized_square("output_image.png")

from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
from moviepy.video.fx.all import resize

def place_video_on_scenery(video_path, scenery_image_path, output_path):
    """
    将视频调整为正方形大小后，放置在风景图片的指定 BBOX 位置（550, 0, 730, 180）

    参数:
        video_path: 输入视频路径
        scenery_image_path: 背景风景图片路径
        output_path: 输出视频路径
    """
    # 加载视频和背景图片
    video = VideoFileClip(video_path)
    scenery = ImageClip(scenery_image_path).set_duration(video.duration)

    # 计算视频的目标尺寸（正方形，边长取视频的较短边）
    min_dim = min(video.size)  # 取宽高中的较小值
    square_video = video.fx(resize, width=min_dim, height=min_dim)

    # 定义 BBOX 位置 (550, 0, 730, 180) -> 宽=180, 高=180
    bbox_x, bbox_y, bbox_x_max, bbox_y_max = 550, 0, 730, 180
    bbox_width = bbox_x_max - bbox_x
    bbox_height = bbox_y_max - bbox_y

    # 调整视频大小以匹配 BBOX 尺寸
    square_video_resized = square_video.fx(resize, width=bbox_width, height=bbox_height)

    # 设置视频在背景上的位置（左上角坐标）
    square_video_resized = square_video_resized.set_position((bbox_x, bbox_y))

    # 合成视频和背景
    final_video = CompositeVideoClip([scenery, square_video_resized])

    # 输出视频
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
    print(f"视频已保存到: {output_path}")

# 使用示例
place_video_on_scenery("Xiang_Float_After_Tomorrow_Head_SPLITED/0005_明天过后 - 张杰.mp4",
                             "Xiang_Float_After_Tomorrow_Head_SPLITED_First_FullBody/0005_明天过后 - 张杰.png",
                             "output.mp4")

