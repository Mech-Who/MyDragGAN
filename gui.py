from backend import UI_Backend
import numpy as np
import dearpygui.dearpygui as dpg
from array import array
import threading
import random
import os.path as osp
import os
import sys
sys.path.append('stylegan2_ada')


model = UI_Backend(device='cpu')  # 定义后端逻辑


def change_device(sender, app_data):
    """
    切换设备的回调函数,直接调用DragModel里面实现的函数
    """
    model.change_device(app_data)


def weight_selected(sender):
    """
    选择模型checkpoint文件的回调函数
    当点击按钮时，弹出文件选择窗口(文件选择窗口id=“weight selector”)
    """
    dpg.show_item("weight selector")


def seed_checkbox_pressed(sender):
    """
    随机选取seed的回调函数
    """
    # 是否随机选取seed
    checked = dpg.get_value('seed_random')
    if checked:
        dpg.disable_item('seed')  # 关闭seed输入框
    else:
        dpg.enable_item('seed')


def generate_image(sender, app_data, user_data):
    """
    点击生成图像按钮的回调函数
    """
    # 检查是否是随机选取seed
    checked = dpg.get_value('seed_random')
    if checked:
        # 系统随机选择seed
        seed = int(random.randint(0, 65536))
        dpg.set_value('seed', value=seed)
    else:
        # 用户指定的seed
        seed = dpg.get_value('seed')
    # 调用生成函数
    image = model.gen_img(seed)


# 定义窗口尺寸
width, height = 260, 200
posx, posy = 0, 0
with dpg.window(
    label='StyleGAN setting', width=width, height=height, pos=(posx, posy),
    no_move=True, no_close=True, no_collapse=True, no_resize=True,
):
    # 设备选择列表
    dpg.add_text('device:', pos=(5, 20))
    dpg.add_combo(
        ('cpu', 'cuda'), default_value='cpu', width=60, pos=(70, 20),
        callback=change_device,
    )
    dpg.add_text('weight:', pos=(5, 40))

    # 模型文件选择窗口
    def select_cb(sender, app_data):
        selections = app_data['selections']
        if selections:
            for fn in selections:
                # 这里加载模型参数
                model.load_ckpt(selections[fn])  # 模型加载参数
                dpg.set_value('weight_name', osp.basename(
                    model.model_path))  # ui显示加载的模型文件名
                break

    def cancel_cb(sender, app_data):
        ...
    with dpg.file_dialog(
        directory_selector=False, show=False, callback=select_cb, id='weight selector',
        cancel_callback=cancel_cb, width=700, height=400
    ):
        dpg.add_file_extension('.*')
        dpg.add_button(
            label="browse", callback=weight_selected,
            pos=(70, 40),
        )
        dpg.add_text('', tag='weight_name', pos=(125, 40))

        # 随机种子配置，可以手动输入，也可以系统随机
        dpg.add_text('seed:', pos=(5, 60))
        dpg.add_input_int(
            label='', width=100, pos=(70, 60), tag='seed', default_value=0,
        )
        dpg.add_checkbox(label='random seed', tag='seed_random',
                         callback=seed_checkbox_pressed, pos=(70, 80))

        # 生成图像按钮
        dpg.add_button(label="generate", pos=(
            70, 100), callback=generate_image)
        
# 定义显示图像的窗口
texture_format = dpg.mvFormat_Float_rgba
# 这里是显示图像的分辨率
image_width, image_height, rgb_channel, rgba_channel = 256, 256, 3, 4
image_pixels = image_height * image_width
raw_data_size = image_width * image_height * rgba_channel
# ui的图像部件会与raw_data绑定，我们只需要更新raw_data的值即可使前端显示不同的图像
raw_data = array('f', [1] * raw_data_size)
with dpg.texture_registry(show=False):
    dpg.add_raw_texture(
        width=image_width, height=image_height, default_value=raw_data,
        format=texture_format, tag="image"
    )

image_posx, image_posy = 2 + width, 0
with dpg.window(
    label='Image', pos=(image_posx, image_posy), tag='Image Win',
    no_move=True, no_close=True, no_collapse=True, no_resize=True,
):
    dpg.add_image("image", show=True, tag='image_data', pos=(10, 30))

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()


