# Copyright      2025                          (authors: Feiteng Li)
import gradio as gr
import os
import sys
import warnings
import random
import tempfile
import subprocess
from datetime import datetime
from pathlib import Path
import torch
from PIL import Image
from torchvision.utils import save_image

warnings.filterwarnings('ignore')

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import save_video

class WanVideoGenerator:
    def __init__(self):
        self.current_pipeline = None
        self.current_task = None
        self.current_ckpt_dir = None

    def _init_pipeline(self, task, ckpt_dir, device_id=0, rank=0, t5_cpu=True, convert_model_dtype=True, progress_callback=None):
        """初始化或重新初始化pipeline"""
        # 如果task或ckpt_dir发生变化，清除当前pipeline
        if (self.current_task != task or 
            self.current_ckpt_dir != ckpt_dir or 
            self.current_pipeline is None):

            # 清理之前的pipeline
            if self.current_pipeline is not None:
                if progress_callback:
                    progress_callback(0.17, desc="清理旧模型...")
                del self.current_pipeline
                torch.cuda.empty_cache()

            if progress_callback:
                progress_callback(0.18, desc="加载模型配置...")
            
            # 获取配置
            cfg = WAN_CONFIGS[task]

            if progress_callback:
                progress_callback(0.20, desc="创建模型实例...")

            # 根据任务类型创建相应的pipeline
            if "t2v" in task:
                self.current_pipeline = wan.WanT2V(
                    config=cfg,
                    checkpoint_dir=ckpt_dir,
                    device_id=device_id,
                    rank=rank,
                    t5_fsdp=False,
                    dit_fsdp=False,
                    use_sp=False,
                    t5_cpu=t5_cpu,
                    convert_model_dtype=convert_model_dtype,
                )
            elif "ti2v" in task:
                self.current_pipeline = wan.WanTI2V(
                    config=cfg,
                    checkpoint_dir=ckpt_dir,
                    device_id=device_id,
                    rank=rank,
                    t5_fsdp=False,
                    dit_fsdp=False,
                    use_sp=False,
                    t5_cpu=t5_cpu,
                    convert_model_dtype=convert_model_dtype,
                )
            else:  # i2v-A14B
                self.current_pipeline = wan.WanI2V(
                    config=cfg,
                    checkpoint_dir=ckpt_dir,
                    device_id=device_id,
                    rank=rank,
                    t5_fsdp=False,
                    dit_fsdp=False,
                    use_sp=False,
                    t5_cpu=t5_cpu,
                    convert_model_dtype=convert_model_dtype,
                )

            # 更新当前状态
            self.current_task = task
            self.current_ckpt_dir = ckpt_dir
            
            if progress_callback:
                progress_callback(0.24, desc="模型加载完成")

    def generate_video(self, task, size, prompt, image, ckpt_dir, 
                      sample_steps, sample_shift, sample_guide_scale, 
                      frame_num, base_seed, offload_model, t5_cpu, 
                      convert_model_dtype, progress=gr.Progress()):
        """
        生成视频的主要函数
        """
        try:
            progress(0.05, desc="验证参数...")

            # 验证参数
            if not ckpt_dir or not os.path.exists(ckpt_dir):
                return None, "错误：请提供有效的模型检查点目录"

            if not prompt.strip():
                return None, "错误：请输入提示词"

            if task == "i2v-A14B" and image is None:
                return None, "错误：I2V任务需要输入图像"

            progress(0.1, desc="准备配置...")

            # 获取配置
            cfg = WAN_CONFIGS[task]

            # 设置默认值
            if sample_steps == 0:
                sample_steps = cfg.sample_steps
            if sample_shift == 0:
                sample_shift = cfg.sample_shift
            if sample_guide_scale == 0:
                sample_guide_scale = cfg.sample_guide_scale
            if frame_num == 0:
                frame_num = cfg.frame_num
            if base_seed == -1:
                base_seed = random.randint(0, sys.maxsize)

            progress(0.15, desc="初始化模型...")

            # 初始化或重用pipeline
            self._init_pipeline(task, ckpt_dir, device_id=0, rank=0, 
                              t5_cpu=t5_cpu, convert_model_dtype=convert_model_dtype,
                              progress_callback=progress)

            progress(0.25, desc="准备生成参数...")

            # 使用当前pipeline生成视频
            if "t2v" in task:
                progress(0.3, desc=f"T2V推理中 (共{sample_steps}步)...")
                video = self.current_pipeline.generate(
                    prompt,
                    size=SIZE_CONFIGS[size],
                    frame_num=frame_num,
                    shift=sample_shift,
                    sample_solver='unipc',
                    sampling_steps=sample_steps,
                    guide_scale=sample_guide_scale,
                    seed=base_seed,
                    offload_model=offload_model
                )

            elif "ti2v" in task:
                # 处理输入图像（可选）
                img = None
                if image is not None:
                    progress(0.3, desc="处理输入图像...")
                    img = Image.open(image).convert("RGB") if isinstance(image, str) else image
                    progress(0.32, desc="图像预处理完成")

                progress(0.35, desc=f"TI2V推理中 (共{sample_steps}步)...")
                video = self.current_pipeline.generate(
                    prompt,
                    img=img,
                    size=SIZE_CONFIGS[size],
                    max_area=MAX_AREA_CONFIGS[size],
                    frame_num=frame_num,
                    shift=sample_shift,
                    sample_solver='unipc',
                    sampling_steps=sample_steps,
                    guide_scale=sample_guide_scale,
                    seed=base_seed,
                    offload_model=offload_model
                )

            else:  # i2v-A14B
                # 处理输入图像
                progress(0.3, desc="处理输入图像...")
                img = Image.open(image).convert("RGB") if isinstance(image, str) else image
                progress(0.32, desc="图像预处理完成")

                progress(0.35, desc=f"I2V推理中 (共{sample_steps}步)...")
                video = self.current_pipeline.generate(
                    prompt,
                    img,
                    max_area=MAX_AREA_CONFIGS[size],
                    frame_num=frame_num,
                    shift=sample_shift,
                    sample_solver='unipc',
                    sampling_steps=sample_steps,
                    guide_scale=sample_guide_scale,
                    seed=base_seed,
                    offload_model=offload_model
                )

            progress(0.8, desc="推理完成，准备保存...")

            progress(0.85, desc="保存视频文件...")

            # 生成输出文件名
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = prompt.replace(" ", "_").replace("/", "_")[:50]
            
            # 根据帧数决定保存格式
            if frame_num == 1:
                output_path = f"{task}_{size.replace('*','x')}_{formatted_prompt}_{formatted_time}.png"
                progress(0.9, desc="保存为PNG图像...")
            else:
                output_path = f"{task}_{size.replace('*','x')}_{formatted_prompt}_{formatted_time}.mp4"
                progress(0.9, desc="编码视频...")

            # 根据帧数保存为不同格式
            if frame_num == 1:
                # 保存为PNG图像
                save_image(
                    video[0],
                    output_path,
                    normalize=True,
                    value_range=(-1, 1)
                )
            else:
                # 保存为MP4视频
                save_video(
                    tensor=video[None],
                    save_file=output_path,
                    fps=cfg.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1)
                )

            progress(0.95, desc="清理内存...")

            progress(1.0, desc="生成完成!")

            # 清理视频内存（但保留pipeline）
            del video
            torch.cuda.empty_cache()

            # 根据文件格式返回不同的成功信息
            file_type = "图像" if frame_num == 1 else "视频"
            return output_path, f"{file_type}生成成功！种子值: {base_seed}, 保存至: {output_path}"

        except Exception as e:
            return None, f"生成失败：{str(e)}"

def create_gradio_interface():
    generator = WanVideoGenerator()

    # 示例提示词
    example_prompts = {
        "t2v-A14B": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
        "i2v-A14B": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression.",
        "ti2v-5B": "5 秒、16∶9 画幅、24 fps 的航拍镜头。机位在距地面约 2 米处快速前移，俯视并跟拍一名 23 岁、体格健美的女性。"
    }

    def update_size_choices(task):
        return gr.Dropdown(choices=SUPPORTED_SIZES[task], value=SUPPORTED_SIZES[task][0])

    def update_prompt_example(task):
        return example_prompts.get(task, "")

    def update_image_visibility(task):
        return gr.Image(visible=task in ["i2v-A14B", "ti2v-5B"])
    
    def generate_video_wrapper(task, size, prompt, image, ckpt_dir, 
                              sample_steps, sample_shift, sample_guide_scale, 
                              frame_num, base_seed, offload_model, t5_cpu, 
                              convert_model_dtype, progress=gr.Progress()):
        """包装函数，处理输出显示"""
        result_path, info = generator.generate_video(
            task, size, prompt, image, ckpt_dir,
            sample_steps, sample_shift, sample_guide_scale,
            frame_num, base_seed, offload_model, t5_cpu,
            convert_model_dtype, progress
        )
        
        if result_path is None:
            # 生成失败
            return None, None, gr.Video(visible=True), gr.Image(visible=False), info
        
        if frame_num == 1:
            # 单帧，显示图像
            return None, result_path, gr.Video(visible=False), gr.Image(visible=True), info
        else:
            # 多帧，显示视频
            return result_path, None, gr.Video(visible=True), gr.Image(visible=False), info

    with gr.Blocks(title="Wan Video Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎬 Wan Video Generator

        基于 Wan 模型的视频生成工具，支持文本到视频(T2V)、图像到视频(I2V)和文本+图像到视频(TI2V)生成。
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 基础设置")

                task = gr.Dropdown(
                    choices=list(WAN_CONFIGS.keys()),
                    value="ti2v-5B",
                    label="任务类型",
                    info="选择生成任务类型"
                )

                size = gr.Dropdown(
                    choices=SUPPORTED_SIZES["ti2v-5B"],
                    value="1280*704",
                    label="视频尺寸",
                    info="生成视频的分辨率"
                )

                ckpt_dir = gr.Textbox(
                    value="./Wan2.2-TI2V-5B",
                    label="模型检查点目录",
                    info="模型权重文件的路径"
                )

                prompt = gr.Textbox(
                    lines=4,
                    value=example_prompts["ti2v-5B"],
                    label="提示词",
                    info="描述要生成的视频内容"
                )

                image = gr.Image(
                    type="pil",
                    label="输入图像 (I2V任务必需，TI2V任务可选)",
                    visible=True
                )

            with gr.Column(scale=1):
                gr.Markdown("### 高级参数")

                with gr.Row():
                    sample_steps = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=50,
                        step=1,
                        label="采样步数",
                        info="默认值=50，采样步数越多，生成的视频质量越高，但时间也越长"
                    )

                    frame_num = gr.Slider(
                        minimum=1,
                        maximum=200,
                        value=121,
                        step=4,
                        label="帧数",
                        info="生成视频的帧数，默认值=121"
                    )

                with gr.Row():
                    sample_shift = gr.Slider(
                        minimum=0.0,
                        maximum=10.0,
                        value=0,
                        step=0.1,
                        label="采样偏移",
                        info="流匹配调度器的采样偏移因子，0表示使用默认值"
                    )

                    sample_guide_scale = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=0,
                        step=0.1,
                        label="引导尺度",
                        info="分类器无关引导尺度，0表示使用默认值"
                    )

                base_seed = gr.Number(
                    value=-1,
                    label="随机种子",
                    info="生成的随机种子，-1表示随机生成"
                )

                with gr.Row():
                    offload_model = gr.Checkbox(
                        value=True,
                        label="模型卸载",
                        info="每次前向后将模型卸载到CPU以减少GPU内存使用"
                    )

                    t5_cpu = gr.Checkbox(
                        value=True,
                        label="T5使用CPU",
                        info="将T5模型放在CPU上运行"
                    )

                    convert_model_dtype = gr.Checkbox(
                        value=True,
                        label="转换模型精度",
                        info="转换模型参数的数据类型"
                    )

                generate_btn = gr.Button(
                    "🎬 生成视频",
                    variant="primary",
                    size="lg"
                )

        with gr.Row():
            with gr.Column():
                output_video = gr.Video(
                    label="生成的视频",
                    height=400,
                    visible=True
                )
                
                output_image = gr.Image(
                    label="生成的图像",
                    height=400,
                    visible=False
                )

                output_info = gr.Textbox(
                    label="生成信息",
                    interactive=False
                )

        # 事件处理
        task.change(
            fn=update_size_choices,
            inputs=[task],
            outputs=[size]
        )

        task.change(
            fn=update_prompt_example,
            inputs=[task],
            outputs=[prompt]
        )

        task.change(
            fn=update_image_visibility,
            inputs=[task],
            outputs=[image]
        )

        generate_btn.click(
            fn=generate_video_wrapper,
            inputs=[
                task, size, prompt, image, ckpt_dir,
                sample_steps, sample_shift, sample_guide_scale,
                frame_num, base_seed, offload_model, t5_cpu,
                convert_model_dtype
            ],
            outputs=[output_video, output_image, output_video, output_image, output_info],
            show_progress=True
        )

        # 示例
        gr.Markdown("### 📝 使用说明")
        gr.Markdown("""
        1. **选择任务类型**：
           - `t2v-A14B`: 文本到视频生成
           - `i2v-A14B`: 图像到视频生成（需要上传图像）
           - `ti2v-5B`: 文本+图像到视频生成（图像可选）

        2. **设置参数**：
           - 根据需要调整视频尺寸、采样参数等
           - 高级参数中的0值表示使用模型默认设置

        3. **输入内容**：
           - 输入描述性的提示词
           - 对于I2V任务，必须上传图像
           - 对于TI2V任务，可以选择上传图像或仅使用文本

        4. **点击生成**：等待模型处理完成

        **注意**：首次运行需要加载模型，可能需要较长时间。确保有足够的GPU内存。
        """)

    return demo

if __name__ == "__main__":
    # 创建Gradio界面
    demo = create_gradio_interface()

    # 启动服务
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,       # 端口
        share=False,            # 是否创建公共链接
        debug=True,             # 调试模式
        show_error=True         # 显示错误信息
    )
