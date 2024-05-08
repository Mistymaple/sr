# root = tk.Tk()
# root.withdraw()
#
# input_window = tk.Toplevel(root)
# input_window.title("输入框窗口")
#
# label1 = tk.Label(input_window, text="请输入要创建的声音的ID")
# label1.grid(row=0, column=0, padx=5, pady=5)
# entry1 = tk.Entry(input_window)
# entry1.grid(row=0, column=0, padx=5, pady=5)
#
# label2 = tk.Label(input_window, text="请输入要创建的声音的名称")
# label2.grid(row=1, column=0, padx=5, pady=5)
# entry2 = tk.Entry(input_window)
# entry2.grid(row=1, column=1, padx=5, pady=5)
#
# label_error = tk.Label(input_window, text="输入信息有误，应为数字！", fg="red")
# label_error.grid(row=2, columnspan=2, padx=5, pady=5)
#
# button_confirm = tk.Button(input_window, text="确认", command=)
#
# classe = entry1.get()
# sound_name = entry2.get()
# try:
# 	classe = int(classe)
# 	root.destory()
# except ValueError:
# 	label_error.config(text="请输入一个整数", fg="red")
# 	entry1.delete(0, tk.END)