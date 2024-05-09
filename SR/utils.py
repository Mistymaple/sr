from sklearn.cluster import KMeans
from pydub import AudioSegment
from main import durata
import tkinter as tk
from tkinter import filedialog,messagebox




# ---------------------------------------------------------------------------------

def vqlbg(d, k):
	# Reshape data if necessary
	d = d.T if d.shape[0] != len(d) else d
	
	# Create KMeans model
	kmeans = KMeans(n_clusters=k, random_state=0)
	
	# Fit the model to the data
	kmeans.fit(d)
	
	# Get cluster centers
	r = kmeans.cluster_centers_.T
	
	return r

# ---------------------------------------------------------------------------------


def apply_effect(effect, param=None, durata=durata, input_p=None, output_d=None):
	if not input_p:
		messagebox.showerror("错误", "未选择输入路径")
		return
	
	if not output_d:
		messagebox.showerror("错误", "未选择输出路径")
		return
	
	sound = AudioSegment.from_file(input_p)
	
	if effect == "变调":
		pitch_factor = param
		if pitch_factor:
			sound = sound._spawn(sound.raw_data, overrides={
				"frame_rate": int(sound.frame_rate * pitch_factor)
			})
	elif effect == "变速":
		speed_factor = param
		if speed_factor:
			sound = sound.speedup(playback_speed=speed_factor)
	elif effect == "调节音量":
		volume_factor = param
		if volume_factor:
			sound = sound.apply_gain(volume_factor)
	elif effect == "添加回声":
		delay, decay = param
		if delay is not None and decay is not None:
			# 创建回声
			echo_sound = sound[:delay].fade_out(100).overlay(sound)
			echo_sound = echo_sound.fade_in(100)  # 回声开始时的渐入效果
			echo_sound = echo_sound.fade_out(100 * (1 - decay))  # 回声的渐出效果
			
			# 叠加回声
			sound = sound.overlay(echo_sound, position=delay)
	
	# 确保输出长度为
	target_length = durata * 1000  # 毫秒
	if len(sound) > target_length:
		sound = sound[:target_length]  # 截断音频
	elif len(sound) < target_length:
		target_length = target_length + 1000
		# 填充静音
		padding = AudioSegment.silent(duration=target_length - len(sound))
		sound = sound + padding
	
	# 写入变声后的音频文件
	sound.export(output_d, format="wav")
	
	messagebox.showinfo("保存完成", f"音频处理完成，已保存到：{output_d}")


def open_effect_window(root):
	# 创建一个新的Toplevel窗口
	effect_window = tk.Toplevel(root)
	effect_window.title("选择应用的效果")
	
	# 选择效果并设置参数
	effects = []
	params = []
	options = ["变调", "变速", "调节音量", "添加回声"]
	for option in options:
		var = tk.IntVar()
		option_frame = tk.Frame(effect_window)
		option_frame.pack()
		
		check_button = tk.Checkbutton(option_frame, text=option, variable=var)
		check_button.grid(row=0, column=0)
		
		if option:
			param_label = tk.Label(option_frame, text="参数:")
			param_label.grid(row=0, column=1)
			param_entry = tk.Entry(option_frame)
			param_entry.grid(row=0, column=2)
			params.append(param_entry)
		else:
			param_label = tk.Label(option_frame, text="参数: ")
			param_label.grid(row=0, column=1)
			params.append(None)
		
		effects.append((option, var))
	
	# 确认选择的效果并获取输入输出路径
	confirm_button = tk.Button(effect_window, text="确定",
	                           command=lambda: process_effects(effect_window, effects, params))
	confirm_button.pack()


def process_effects(window, effects, params):
	# 获取输入输出路径
	input_p = filedialog.askopenfilename(filetypes=[("Wave files", "*.wav")])
	if not input_p:
		messagebox.showerror("错误", "未选择输入路径")
		return
	
	output_d = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("Wave files", "*.wav")])
	if not output_d:
		messagebox.showerror("错误", "未选择输出路径")
		return
	
	# 应用选择的效果并参数
	for effect, var in effects:
		if var.get():
			index = effects.index((effect, var))
			param = params[index].get() if params[index] else None
			apply_effect(effect, param, durata, input_p=input_p, output_d=output_d)
	
	# 关闭效果选择窗口
	window.destroy()
			