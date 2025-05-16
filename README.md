<p align="center">

<h1 align="center">VACE: All-in-One Video Creation and Editing</h1>
<p align="center">
    <strong>Zeyinzi Jiang<sup>*</sup></strong>
    ·
    <strong>Zhen Han<sup>*</sup></strong>
    ·
    <strong>Chaojie Mao<sup>*&dagger;</sup></strong>
    ·
    <strong>Jingfeng Zhang</strong>
    ·
    <strong>Yulin Pan</strong>
    ·
    <strong>Yu Liu</strong>
    <br>
    <b>Tongyi Lab - <a href="https://github.com/Wan-Video/Wan2.1"><img src='https://ali-vilab.github.io/VACE-Page/assets/logos/wan_logo.png' alt='wan_logo' style='margin-bottom: -4px; height: 20px;'></a> </b>
    <br>
    <br>
        <a href="https://arxiv.org/abs/2503.07598"><img src='https://img.shields.io/badge/VACE-arXiv-red' alt='Paper PDF'></a>
        <a href="https://ali-vilab.github.io/VACE-Page/"><img src='https://img.shields.io/badge/VACE-Project_Page-green' alt='Project Page'></a>
        <a href="https://huggingface.co/collections/ali-vilab/vace-67eca186ff3e3564726aff38"><img src='https://img.shields.io/badge/VACE-HuggingFace_Model-yellow'></a>
        <a href="https://modelscope.cn/collections/VACE-8fa5fcfd386e43"><img src='https://img.shields.io/badge/VACE-ModelScope_Model-purple'></a>
    <br>
</p>

```bash
git clone https://github.com/ali-vilab/VACE.git && cd VACE
pip install torch==2.5.1 torchvision==0.20.1
pip install -r requirements.txt
pip install wan@git+https://github.com/Wan-Video/Wan2.1  # If you want to use Wan2.1-based VACE.
pip install moviepy==1.0.3
#pip install ltx-video@git+https://github.com/Lightricks/LTX-Video@ltx-video-0.9.1 sentencepiece --no-deps # If you want to use LTX-Video-0.9-based VACE. It may conflict with Wan.

python vace/vace_wan_inference.py --model_name vace-1.3B --src_video <path-to-src-video> --src_mask <path-to-src-mask> --src_ref_images <paths-to-src-ref-images> --prompt <prompt>  # wan

huggingface-cli download  Wan-AI/Wan2.1-VACE-1.3B --local-dir models/Wan2.1-VACE-1.3B --repo-type model

huggingface-cli download ali-vilab/VACE-Benchmark --local-dir benchmarks/VACE-Benchmark --repo-type dataset

python vace/vace_wan_inference.py --model_name vace-1.3B \
 --src_video "benchmarks/VACE-Benchmark/assets/examples/outpainting/src_video.mp4" \
 --src_mask "benchmarks/VACE-Benchmark/assets/examples/outpainting/src_mask.mp4" \
 --prompt "赛博朋克风格，无人机俯瞰视角下的现代西安城墙，镜头穿过永宁门时泛起金色涟漪，城墙砖块化作数据流重组为唐代长安城。周围的街道上流动的人群和飞驰的机械交通工具交织在一起，现代与古代的交融，城墙上的灯光闪烁，形成时空隧道的效果。全息投影技术展现历史变迁，粒子重组特效细腻逼真。大远景逐渐过渡到特写，聚焦于城门特效。"

```

### One Outpainting Demo 
```python
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

```

```bash
python vace/vace_wan_inference.py --model_name vace-1.3B \
 --src_video "AnimateDiff_00011-audio_placed.mp4" \
 --src_mask "mask.mp4" \
 --prompt "一个戴眼镜男青年在自然风光美丽的无人室外唱歌。"
```

#### In Loop Outpainting Demo 
```python
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

#vim run_outpainting.py

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
```



## Introduction
<strong>VACE</strong> is an all-in-one model designed for video creation and editing. It encompasses various tasks, including reference-to-video generation (<strong>R2V</strong>), video-to-video editing (<strong>V2V</strong>), and masked video-to-video editing (<strong>MV2V</strong>), allowing users to compose these tasks freely. This functionality enables users to explore diverse possibilities and streamlines their workflows effectively, offering a range of capabilities, such as Move-Anything, Swap-Anything, Reference-Anything, Expand-Anything, Animate-Anything, and more.

<img src='./assets/materials/teaser.jpg'>


## 🎉 News
- [x] May 14, 2025: 🔥Wan2.1-VACE-1.3B and Wan2.1-VACE-14B models are now available at [HuggingFace](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B) and [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-VACE-14B)!
- [x] Mar 31, 2025: 🔥VACE-Wan2.1-1.3B-Preview and VACE-LTX-Video-0.9 models are now available at [HuggingFace](https://huggingface.co/collections/ali-vilab/vace-67eca186ff3e3564726aff38) and [ModelScope](https://modelscope.cn/collections/VACE-8fa5fcfd386e43)!
- [x] Mar 31, 2025: 🔥Release code of model inference, preprocessing, and gradio demos. 
- [x] Mar 11, 2025: We propose [VACE](https://ali-vilab.github.io/VACE-Page/), an all-in-one model for video creation and editing.


## 🪄 Models
| Models                   | Download Link                                                                                                                                           | Video Size        | License                                                                                       |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-----------------------------------------------------------------------------------------------|
| VACE-Wan2.1-1.3B-Preview | [Huggingface](https://huggingface.co/ali-vilab/VACE-Wan2.1-1.3B-Preview) 🤗  [ModelScope](https://modelscope.cn/models/iic/VACE-Wan2.1-1.3B-Preview) 🤖 | ~ 81 x 480 x 832  | [Apache-2.0](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/blob/main/LICENSE.txt)             |
| VACE-LTX-Video-0.9       | [Huggingface](https://huggingface.co/ali-vilab/VACE-LTX-Video-0.9) 🤗     [ModelScope](https://modelscope.cn/models/iic/VACE-LTX-Video-0.9) 🤖          | ~ 97 x 512 x 768  | [RAIL-M](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.license.txt) |
| Wan2.1-VACE-1.3B         | [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B) 🤗     [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-VACE-1.3B) 🤖          | ~ 81 x 480 x 832  | [Apache-2.0](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/blob/main/LICENSE.txt)             |
| Wan2.1-VACE-14B          | [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B) 🤗     [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-VACE-14B) 🤖            | ~ 81 x 720 x 1280 | [Apache-2.0](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/blob/main/LICENSE.txt)             |

- The input supports any resolution, but to achieve optimal results, the video size should fall within a specific range.
- All models inherit the license of the original model.


## ⚙️ Installation
The codebase was tested with Python 3.10.13, CUDA version 12.4, and PyTorch >= 2.5.1.

### Setup for Model Inference
You can setup for VACE model inference by running:
```bash
git clone https://github.com/ali-vilab/VACE.git && cd VACE
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124  # If PyTorch is not installed.
pip install -r requirements.txt
pip install wan@git+https://github.com/Wan-Video/Wan2.1  # If you want to use Wan2.1-based VACE.
pip install ltx-video@git+https://github.com/Lightricks/LTX-Video@ltx-video-0.9.1 sentencepiece --no-deps # If you want to use LTX-Video-0.9-based VACE. It may conflict with Wan.
```
Please download your preferred base model to `<repo-root>/models/`. 

### Setup for Preprocess Tools
If you need preprocessing tools, please install:
```bash
pip install -r requirements/annotator.txt
```
Please download [VACE-Annotators](https://huggingface.co/ali-vilab/VACE-Annotators) to `<repo-root>/models/`.

### Local Directories Setup
It is recommended to download [VACE-Benchmark](https://huggingface.co/datasets/ali-vilab/VACE-Benchmark) to `<repo-root>/benchmarks/` as examples in `run_vace_xxx.sh`.

We recommend to organize local directories as:
```angular2html
VACE
├── ...
├── benchmarks
│   └── VACE-Benchmark
│       └── assets
│           └── examples
│               ├── animate_anything
│               │   └── ...
│               └── ...
├── models
│   ├── VACE-Annotators
│   │   └── ...
│   ├── VACE-LTX-Video-0.9
│   │   └── ...
│   └── VACE-Wan2.1-1.3B-Preview
│       └── ...
└── ...
```

## 🚀 Usage
In VACE, users can input **text prompt** and optional **video**, **mask**, and **image** for video generation or editing.
Detailed instructions for using VACE can be found in the [User Guide](./UserGuide.md).

### Inference CIL
#### 1) End-to-End Running
To simply run VACE without diving into any implementation details, we suggest an end-to-end pipeline. For example:
```bash
# run V2V depth
python vace/vace_pipeline.py --base wan --task depth --video assets/videos/test.mp4 --prompt 'xxx'

# run MV2V inpainting by providing bbox
python vace/vace_pipeline.py --base wan --task inpainting --mode bbox --bbox 50,50,550,700 --video assets/videos/test.mp4 --prompt 'xxx'
```
This script will run video preprocessing and model inference sequentially, 
and you need to specify all the required args of preprocessing (`--task`, `--mode`, `--bbox`, `--video`, etc.) and inference (`--prompt`, etc.). 
The output video together with intermediate video, mask and images will be saved into `./results/` by default.

> 💡**Note**:
> Please refer to [run_vace_pipeline.sh](./run_vace_pipeline.sh) for usage examples of different task pipelines.


#### 2) Preprocessing
To have more flexible control over the input, before VACE model inference, user inputs need to be preprocessed into `src_video`, `src_mask`, and `src_ref_images` first.
We assign each [preprocessor](./vace/configs/__init__.py) a task name, so simply call [`vace_preprocess.py`](./vace/vace_preproccess.py) and specify the task name and task params. For example:
```angular2html
# process video depth
python vace/vace_preproccess.py --task depth --video assets/videos/test.mp4

# process video inpainting by providing bbox
python vace/vace_preproccess.py --task inpainting --mode bbox --bbox 50,50,550,700 --video assets/videos/test.mp4
```
The outputs will be saved to `./processed/` by default.

> 💡**Note**:
> Please refer to [run_vace_pipeline.sh](./run_vace_pipeline.sh) preprocessing methods for different tasks.
Moreover, refer to [vace/configs/](./vace/configs/) for all the pre-defined tasks and required params.
You can also customize preprocessors by implementing at [`annotators`](./vace/annotators/__init__.py) and register them at [`configs`](./vace/configs).


#### 3) Model inference
Using the input data obtained from **Preprocessing**, the model inference process can be performed as follows:
```bash
# For Wan2.1 single GPU inference (1.3B-480P)
python vace/vace_wan_inference.py --ckpt_dir <path-to-model> --src_video <path-to-src-video> --src_mask <path-to-src-mask> --src_ref_images <paths-to-src-ref-images> --prompt "xxx"

# For Wan2.1 Multi GPU Acceleration inference (1.3B-480P)
pip install "xfuser>=0.4.1"
torchrun --nproc_per_node=8 vace/vace_wan_inference.py --dit_fsdp --t5_fsdp --ulysses_size 1 --ring_size 8 --ckpt_dir <path-to-model> --src_video <path-to-src-video> --src_mask <path-to-src-mask> --src_ref_images <paths-to-src-ref-images> --prompt "xxx"

# For Wan2.1 Multi GPU Acceleration inference (14B-720P)
torchrun --nproc_per_node=8 vace/vace_wan_inference.py --dit_fsdp --t5_fsdp --ulysses_size 8 --ring_size 1 --size 720p --model_name 'vace-14B' --ckpt_dir <path-to-model> --src_video <path-to-src-video> --src_mask <path-to-src-mask> --src_ref_images <paths-to-src-ref-images> --prompt "xxx"

# For LTX inference, run
python vace/vace_ltx_inference.py --ckpt_path <path-to-model> --text_encoder_path <path-to-model> --src_video <path-to-src-video> --src_mask <path-to-src-mask> --src_ref_images <paths-to-src-ref-images> --prompt "xxx"
```
The output video together with intermediate video, mask and images will be saved into `./results/` by default.

> 💡**Note**: 
> (1) Please refer to [vace/vace_wan_inference.py](./vace/vace_wan_inference.py) and [vace/vace_ltx_inference.py](./vace/vace_ltx_inference.py) for the inference args.
> (2) For LTX-Video and English language Wan2.1 users, you need prompt extension to unlock the full model performance. 
Please follow the [instruction of Wan2.1](https://github.com/Wan-Video/Wan2.1?tab=readme-ov-file#2-using-prompt-extension) and set `--use_prompt_extend` while running inference.
> (3) When performing prompt extension in editing tasks, it's important to pay attention to the results of expanding plain text. Since the visual information being input is unknown, this may lead to the extended output not matching the video being edited, which can affect the final outcome.

### Inference Gradio
For preprocessors, run 
```bash
python vace/gradios/vace_preprocess_demo.py
```
For model inference, run
```bash
# For Wan2.1 gradio inference
python vace/gradios/vace_wan_demo.py

# For LTX gradio inference
python vace/gradios/vace_ltx_demo.py
```

## Acknowledgement

We are grateful for the following awesome projects, including [Scepter](https://github.com/modelscope/scepter), [Wan](https://github.com/Wan-Video/Wan2.1), and [LTX-Video](https://github.com/Lightricks/LTX-Video).


## BibTeX

```bibtex
@article{vace,
    title = {VACE: All-in-One Video Creation and Editing},
    author = {Jiang, Zeyinzi and Han, Zhen and Mao, Chaojie and Zhang, Jingfeng and Pan, Yulin and Liu, Yu},
    journal = {arXiv preprint arXiv:2503.07598},
    year = {2025}
}
