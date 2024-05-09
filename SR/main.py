import os
import glob
import pickle
import numpy as np
import soundfile as sf
import sounddevice as sd
from utils import *
from scipy.io import wavfile
from python_speech_features import mfcc
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Global variables
sound_number = 0    # 初始化音频数量
k = 64              # 全局VQ质心数量
code = []           # 初始化VQ码本数组
durata = 5          # 全局录音时长，便于使矩阵大小相等以进行计算
samplingfrequency, samplingbits = 22050, 8
min_idx = 0
mfcc_fea = np.zeros((0))

data = {}
if os.path.isfile('sound_database.pkl'):
    with open('sound_database.pkl', 'rb') as tf:
        data = pickle.load(tf)
    sound_number = data['sound_number']
else:
    data = {'data': [], 'sound_number': 0, 'samplingfrequency': 22050, 'samplingbits': 8}
    with open('sound_database.pkl', 'wb') as tf:
        pickle.dump(data, tf)
    sound_number = data['sound_number']

min_len = 20
user_num = 8
samplingfrequency = data['samplingfrequency']
samplingbits = data['samplingbits']

# 调用麦克风采集音频
def add_voice():
    def get_input():
        classe = entry1.get()
        if not classe:
            classe = sound_number + 1
            print(classe)
        sound_name = entry2.get()
        if not sound_name:
            sound_name = 'name_' + str(classe)
            print(sound_name)
        root.destroy()  # 销毁窗口
        record_audio(classe, sound_name)
        
    def record_audio(classe, sound_name):
        global sound_number, data, durata, samplingfrequency, samplingbits
        durata = int(durata)
        if messagebox.askyesno('录音确认', f"您是否已准备好录音？\n注：录音时间为{durata}s"):
            micrecorder = sd.rec(int(durata * samplingfrequency), samplerate=samplingfrequency, channels=1, dtype='float64')
            sd.wait()  # 等待录制完成
            y1 = np.squeeze(micrecorder)
            y_uint8 = np.uint8((micrecorder + 1) * 127.5)
            
            if y_uint8.ndim > 1 and y_uint8.shape[1] == 2:
                y_uint8 = y_uint8[:, 0]
            
            y = y_uint8.astype(np.float64)
            
            sound_number += 1
            print(sound_number, '\tsound_number')
            # 将音频数据、类别和其他信息存储到data数组中
            # 用于保证data键内有足够空间
            while len(data['data']) < sound_number:
                data['data'].append([])
            
            data['sound_number'] = sound_number
            data['data'][sound_number - 1] = [y, classe, 'Mircophone', sound_name]
            
            st = f'train/train_{sound_number}.wav'  # 文件路径
            sf.write(st, y1, samplingfrequency, subtype='PCM_U8')  # 假设y1是录制的音频数据，8位无符号整数PCM格式
            
            # 保存data和sound_number到sound_database.pkl文件
            with open('sound_database.pkl', 'wb') as tf:
                pickle.dump(data, tf)
            
            # 弹出消息框
            msg = '声音已添加到数据库中'
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo('添加结果反馈', msg)
        
        else:
            return


    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口

    # 创建新窗口
    input_window = tk.Toplevel(root)
    input_window.title("输入框窗口")

    # 标签和输入框
    label1 = tk.Label(input_window, text="请输入用于识别的类别编号（声音ID）：")
    label1.grid(row=0, column=0, padx=5, pady=5)
    entry1 = tk.Entry(input_window)
    entry1.grid(row=0, column=1, padx=5, pady=5)

    label2 = tk.Label(input_window, text="请输入用于识别的类别名称（声音名称）：")
    label2.grid(row=1, column=0, padx=5, pady=5)
    entry2 = tk.Entry(input_window)
    entry2.grid(row=1, column=1, padx=5, pady=5)

    # 确认按钮
    button_confirm = tk.Button(input_window, text="确认", command=get_input)
    button_confirm.grid(row=2, columnspan=2, padx=5, pady=5)


# 声纹识别函数
def speaker_recognition():
    if os.path.isfile('sound_database.pkl'):
        with open('sound_database.pkl', 'rb') as tf:
            data = pickle.load(tf)
        Fs = data['samplingfrequency']
        sound_number = data['sound_number']
        global min_len, durata, mfcc_fea, min_idx

        if messagebox.askyesno('录音确认', f"您是否已准备好录音？\n注：录音时间为{durata}s"):
            y = sd.rec(int(durata * Fs), samplerate=int(Fs), channels=1, dtype='float64')
            sd.wait()
        else:
            return
        
        y1 = np.squeeze(y)
        st = f'test/test_voice.wav'
        sf.write(st, y1, samplingfrequency, subtype='PCM_U8')
        
        print('正在进行MFCC系数计算和VQ码本训练...')
        print()

        global k, code
        
        file_list = glob.glob('train/train_*.wav')
        for file_path in file_list:
            sr, s = wavfile.read(file_path)
            v = mfcc(s, samplerate=sr, nfft=1024)
            code.append(vqlbg(v, k))
        
        
        subdir = 'test'
        filename = 'test_voice.wav'
        file_path = os.path.join(subdir, filename)
        sr, s = wavfile.read(file_path)
        v = mfcc(s, samplerate=sr, nfft=1024)
        mfcc_fea = np.array(v)
        distmin = float('inf')
        k1 = 0
        dist_info = ""
        
        for ii in range(sound_number):
            d = cdist(v, code[ii].T, metric='euclidean')
            
            if np.any(d):
                dist = np.sum(np.min(d, axis=1)) / d.shape[0]
                # 将距离信息添加到字符串变量中
                dist_info += f'For User #{ii + 1} Dist : {dist}\n'
                if dist < distmin:
                    distmin = dist
                    k1 = ii                # 循环找到距离最小值，并确定最小值索引k1
            else:
                messagebox.showinfo("匹配结果", f"For User #{ii + 1} Dist : No match found")
        
        messagebox.showinfo("欧氏距离", dist_info)



        if distmin < min_len:
            min_index = k1
            speech_id = data['data'][min_index][1]
            messagebox.showinfo("匹配结果", f"匹配的声音:\n匹配的用户的ID为: {speech_id}，用户名为: {data['data'][min_index][3]}")
            min_idx = min_index

        else:
            messagebox.showinfo("匹配结果", "声音匹配不成功")

    else:
        messagebox.showinfo('识别有误', '数据库是空的，请先添加声音进行训练。')


# 删除数据库
def delete_database():
    global sound_number, data
    if sound_number == 0:
        messagebox.showwarning("系统提示", "数据库是空的，无需删除。")
        return
    # 待添加删除某一项数据
    
    confirmation = messagebox.askyesno("系统提示", "你真的想删除数据库吗?")
    if confirmation:
        for i in range(sound_number):
            filename = f"train/train_{i + 1}.wav"
            os.remove(filename)
                
        if os.path.exists("test/v.wav"):
            os.remove("test/v.wav")
        sound_number = 0
        os.remove("sound_database.pkl")
        messagebox.showinfo("删除结果反馈", "数据库已成功删除")


def visualize_clusters(data, centroids):
    data = np.array(data)  # 将数据转换为 NumPy 数组
    centroids = np.array(centroids)  # 将聚类中心转换为 NumPy 数组
    
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], c='b', alpha=0.5, label='Data Points')
    ax.scatter(centroids[:, 0], centroids[:, 1], c='r', marker='*', label='Cluster Centers')
    
    for centroid in centroids:
        # 计算每个聚类中心周围数据点的范围
        points = data[np.linalg.norm(data - centroid, axis=1) < 0.5]
        if len(points) > 0:
            # 计算范围的边界
            min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
            min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
            # 用线段划分范围
            polygon = Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)], closed=True, fill=None, edgecolor='r', linewidth=1)
            ax.add_patch(polygon)
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('KMeans Clustering Visualization')
    ax.legend()
    
    return fig

# 创建一个新的 Tkinter 窗口来显示 Matplotlib 图形
def show_visualization(data, centroids):
    root = tk.Tk()
    root.title("KMeans Clustering Visualization")
    
    # 创建一个 Figure 对象，并在其上绘制聚类图
    fig = visualize_clusters(data, centroids)
    
    # 将 Figure 对象转换为 Tkinter 可以显示的格式
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
    # 添加关闭按钮
    button = tk.Button(root, text="关闭", command=root.destroy)
    button.pack(side=tk.BOTTOM)
    
    # 运行 Tkinter 事件循环
    root.mainloop()

# Main
def main():
    root = tk.Tk()
    root.title("说话人识别系统")
    
    global mfcc_fea, min_idx
    
    def on_add_sound():
        add_voice()

    def on_recognize_voice():
        speaker_recognition()

    def on_delete_database():
        delete_database()

    def on_view_audio_info():
        file_path1 = os.path.join(os.getcwd(), 'test', 'test_voice.wav')
        file_path2 = filedialog.askopenfilename(filetypes=[("Wave files", "*.wav")])
        if file_path1 and file_path2:
            show_audio_info(file_path1, file_path2, root)
            
        show_visualization(mfcc_fea, code[min_idx].T)
    
    def on_change_audio():
        open_effect_window(root)

    def exit_system():
        if messagebox.askyesno("退出系统", "确定要退出系统吗？"):
            root.destroy()

    # UI 设置
    tk.Label(root, text="Speaker Recognition System", font=("Helvetica", 16)).grid(row=0, column=0, columnspan=3)

    tk.Button(root, text="添加声音", command=on_add_sound).grid(row=1, column=0, padx=10, pady=5, sticky="ew")
    tk.Button(root, text="识别声音", command=on_recognize_voice).grid(row=2, column=0, padx=10, pady=5, sticky="ew")
    tk.Button(root, text="删除数据库", command=on_delete_database).grid(row=3, column=0, padx=10, pady=5, sticky="ew")
    tk.Button(root, text="查看音频信息", command=on_view_audio_info).grid(row=4, column=0, padx=10, pady=5, sticky="ew")
    tk.Button(root, text="退出系统", command=exit_system).grid(row=5, column=0, padx=10, pady=5, sticky="ew")
    tk.Button(root, text="变声", command=on_change_audio).grid(row=5, column=1, padx=10, pady=5, sticky="ew")

    # 创建绘图区域
    canvas1 = tk.Canvas(root, width=400, height=200, bg="white")
    canvas1.grid(row=1, column=1, padx=20, pady=20, rowspan=2)
    canvas2 = tk.Canvas(root, width=400, height=200, bg="white")
    canvas2.grid(row=3, column=1, padx=20, pady=20, rowspan=2)
    canvas3 = tk.Canvas(root, width=400, height=200, bg="white")
    canvas3.grid(row=1, column=2, padx=20, pady=20, rowspan=2)
    canvas4 = tk.Canvas(root, width=400, height=200, bg="white")
    canvas4.grid(row=3, column=2, padx=20, pady=20, rowspan=2)

    root.mainloop()

def show_audio_info(file_path1, file_path2, root):
    fig1, ax1 = plt.subplots(figsize=(6, 3))  # 调整图形大小
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    fig4, ax4 = plt.subplots(figsize=(6, 3))

    # Load audio files and display waveform
    data1, sr1 = sf.read(file_path1)
    data2, sr2 = sf.read(file_path2)
    ax1.plot(data1)
    ax1.set_title('Test_Voice_Waveform')
    ax1.set_xlim(0, len(data1))  # 调整横轴范围
    ax1.set_ylim(-1, 1)  # 调整纵轴范围
    ax2.plot(data2)
    ax2.set_title('Train_Waveform')
    ax2.set_xlim(0, len(data2))  # 调整横轴范围
    ax2.set_ylim(-1, 1)  # 调整纵轴范围

    ax3.specgram(data1, Fs=sr1)
    ax3.set_title('Test_Vocie_Spectrogram')
    ax4.specgram(data2, Fs=sr2)
    ax4.set_title('Train_Spectrogram')

    canvas1 = FigureCanvasTkAgg(fig1, master=root)
    canvas1.draw()
    canvas1.get_tk_widget().grid(row=1, column=1, padx=20, pady=20)

    canvas2 = FigureCanvasTkAgg(fig2, master=root)
    canvas2.draw()
    canvas2.get_tk_widget().grid(row=3, column=1, padx=20, pady=20)

    canvas3 = FigureCanvasTkAgg(fig3, master=root)
    canvas3.draw()
    canvas3.get_tk_widget().grid(row=1, column=2, padx=20, pady=20)

    canvas4 = FigureCanvasTkAgg(fig4, master=root)
    canvas4.draw()
    canvas4.get_tk_widget().grid(row=3, column=2, padx=20, pady=20)

if __name__ == "__main__":
    main()
