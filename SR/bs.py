from pydub import AudioSegment
from main import durata
def change_audio(input_path, output_path):
    # 读取音频文件
    sound = AudioSegment.from_file(input_path)

    # 用户选择要应用的效果
    print("请选择要应用的效果:")
    print("1. 变调")
    print("2. 变速")
    print("3. 调节音量")
    print("4. 添加回声")

    selected_effects = input("请输入效果编号（多个编号用逗号分隔）：").split(",")

    # 根据选择的效果提示用户输入参数，并应用效果
    for effect in selected_effects:
        if effect == "1":
            pitch_factor = float(input("请输入变调因子（大于1为升调，小于1为降调）："))
            sound = sound._spawn(sound.raw_data, overrides={
                "frame_rate": int(sound.frame_rate * pitch_factor)
            })
        elif effect == "2":
            speed_factor = float(input("请输入速度因子（大于1加速，小于1减速）："))
            sound = sound.speedup(playback_speed=speed_factor)
        elif effect == "3":
            volume_factor = float(input("请输入音量调节因子（大于1增加音量，小于1减小音量）："))
            sound = sound.apply_gain(volume_factor)
        elif effect == "4":
            delay = int(input("请输入回声延迟时间（毫秒）："))
            decay = float(input("请输入回声衰减系数（0到1之间）："))
    
          # 创建回声
            echo_sound = sound[:delay].fade_out(100).overlay(sound)
            echo_sound = echo_sound.fade_in(100)  # 回声开始时的渐入效果
            echo_sound = echo_sound.fade_out(100 * (1 - decay))  # 回声的渐出效果
    
          # 叠加回声
            sound = sound.overlay(echo_sound, position=delay)


    # 确保输出长度为3秒
    target_length = durata*1000  # 毫秒
    if len(sound) > target_length:
        sound = sound[:target_length]  # 截断音频
    elif len(sound) < target_length:
        target_length=target_length+1000
        # 填充静音
        padding = AudioSegment.silent(duration=target_length - len(sound))
        sound = sound + padding

    # 生成输出文件路径
    #output_filename = os.path.basename(input_path)
    #output_path = os.path.join(output_dir, output_filename)

    # 写入变声后的音频文件
    sound.export(output_path, format="wav")

    print("音频处理完成，已保存到：", output_path)

# 指定输入文件路径、输出文件夹路径
input_path = r"E:\Laptop_XiaoXin16Pro\D\Documents\PycharmProjects\SR\train\train_1.wav"
output_path = r"E:\Laptop_XiaoXin16Pro\D\Documents\PycharmProjects\SR\test\test_voice.wav"

# 执行变声功能
change_audio(input_path, output_path)



