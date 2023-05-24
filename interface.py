import customtkinter as ctk
import tkinter.ttk as ttk
from tkinter import filedialog as fd
from PIL import ImageTk, Image
import os
import threading
import sys

from driver import Driver

ctk.set_appearance_mode("Dark")


class App(ctk.CTk):
    def __init__(self):
        ctk.CTk.__init__(self)
        self.title("เบลอใบหน้า")
        self.geometry("600x400")

        self.container = ctk.CTkFrame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        # driver = Driver()

        self.frames = {}
        for F in (FileFrame, ExportFrame):
            page_name = F.__name__
            frame = F(parent=self.container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("FileFrame")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

    def add_new_frame(self):
        frame = SelectFrame(parent=self.container, controller=self)
        self.frames["SelectFrame"] = frame
        frame.grid(row=0, column=0, sticky="nsew")


class FileFrame(ctk.CTkFrame):
    def __init__(self, parent, controller):
        ctk.CTkFrame.__init__(self, parent)
        self.controller = controller
        self.driver = Driver()

        # first file btn
        self.input_path = ""
        self.file_open_btn = ctk.CTkButton(
            self, text="เปิดหาไฟล์", command=lambda: self.openFile()
        )
        self.file_open_btn.place(relx=0.5, rely=0.5, anchor=ctk.CENTER)

        # after choose first file component
        self.fileNameFrame = ctk.CTkFrame(self)
        self.modeNameFrame = ctk.CTkFrame(self)
        self.modeOptionFrame = ctk.CTkFrame(self)
        self.processFrame = ctk.CTkFrame(self)

        self.file_change_btn = ctk.CTkButton(
            self.fileNameFrame, text="เปลี่ยนไฟล์", command=lambda: self.changeFile()
        )
        self.file_change_btn.pack(side="left")
        self.file_path_label = ctk.CTkLabel(self.fileNameFrame, text=self.input_path)
        self.file_path_label.pack(side="right", padx=20)

        self.mode_label = ctk.CTkLabel(self.modeNameFrame, text="เลือกรูปแบบการเบลอ")
        self.mode_label.pack()

        self.mode = ctk.IntVar()
        self.option1 = ctk.CTkRadioButton(
            self.modeOptionFrame, text="เลือกไม่เบลอบางคน", variable=self.mode, value=0
        )
        self.option2 = ctk.CTkRadioButton(
            self.modeOptionFrame, text="เบลอใบหน้าทุกคน", variable=self.mode, value=1
        )
        self.option1.pack(side="left", padx=20)
        self.option2.pack(side="right", padx=20)

        self.process_btn = ctk.CTkButton(
            self.processFrame, text="ต่อไป", command=lambda: self.processVideo()
        )
        self.process_btn.pack(side="right")

    def openFile(self):
        filetypes = (("Video files", "*.mp4"), ("All files", "*.*"))
        input_path = fd.askopenfilename(
            title="Open a file", initialdir="/", filetypes=filetypes
        )

        if not input_path or input_path == "":
            return

        self.input_path = input_path
        self.file_path_label.configure(text=input_path)
        self.driver.input_path = input_path

        self.file_open_btn.place_forget()

        self.fileNameFrame.place(relx=0.5, rely=0.2, anchor=ctk.N)
        self.modeNameFrame.place(relx=0.5, rely=0.4, anchor=ctk.N)
        self.modeOptionFrame.place(relx=0.5, rely=0.5, anchor=ctk.N)
        self.processFrame.place(relx=0.5, rely=0.7, anchor=ctk.N)

    def changeFile(self):
        filetypes = (("Video files", "*.mp4"), ("All files", "*.*"))
        input_path = fd.askopenfilename(
            title="Open a file", initialdir="/", filetypes=filetypes
        )

        if not input_path or input_path == "":
            return

        self.input_path = input_path
        self.file_path_label.configure(text=input_path)
        self.driver.input_path = input_path

    def processVideo(self):
        if self.mode.get() == 0:
            self.driver.mode = 0
            self.driver.blur_except_process()
            self.controller.add_new_frame()
            self.controller.show_frame("SelectFrame")
        else:
            self.driver.mode = 1
            self.controller.show_frame("ExportFrame")


class SelectFrame(ctk.CTkFrame):
    def __init__(self, parent, controller):
        ctk.CTkFrame.__init__(self, parent)
        self.controller = controller
        self.driver = Driver()
        self.face_option = self.driver.blur_except_get_face()

        self.select_label = ctk.CTkLabel(
            self, text="เลือกใบหน้าของคนที่ไม่ต้องการจะเบลอ"
        )
        self.select_label.pack()

        self.face_frame = ctk.CTkFrame(self)
        self.images = [
            ImageTk.PhotoImage(Image.open("./Clusters/{}".format(f)).resize((100, 100)))
            for f in self.face_option
        ]
        self.data = [ctk.BooleanVar() for _ in self.face_option]

        for i, face in enumerate(self.face_option):
            q, mod = divmod(i, 5)
            ttk.Checkbutton(
                self.face_frame,
                image=self.images[i],
                variable=self.data[i],
            ).grid(row=q + 1, column=mod)

            self.face_frame.pack()

        self.confirm_face = ctk.CTkButton(
            self, text="ต่อไป", command=lambda: self.chooseFace()
        )
        self.confirm_face.pack()

    def chooseFace(self):
        choose_face = [index for index, value in enumerate(self.data) if value.get()]
        self.driver.face_except_list = choose_face
        self.controller.show_frame("ExportFrame")


class ExportFrame(ctk.CTkFrame):
    def __init__(self, parent, controller):
        ctk.CTkFrame.__init__(self, parent)
        self.controller = controller
        self.driver = Driver()
        self.thread = None

        self.export_label = ctk.CTkLabel(self, text="เลือกปลายทางในการนำออกวิดีโอ")
        self.export_label.pack()

        self.export_path = ""
        self.export_path_btn = ctk.CTkButton(
            self,
            text="เลือกตำแหน่งปลายทาง",
            command=lambda: self.openFile(),
        )
        self.export_path_btn.pack()
        self.export_path_label = ctk.CTkLabel(self, text=self.export_path)
        self.export_path_label.pack()

        self.export_btn = ctk.CTkButton(
            self, text="เริ่มต้นการเบลอ", command=lambda: self.exportVideo()
        )

        self.home_btn = ctk.CTkButton(
            self,
            text="กลับหน้าแรก",
            command=lambda: self.controller.show_frame("FileFrame"),
        )
        if self.thread != None:
            if not self.thread.isAlive:
                self.home_btn.pack(pady=20)
                self.export_btn["state"] = ctk.NORMAL
            else:
                self.export_btn["state"] = ctk.DISABLED

    def openFile(self):
        export_path = fd.askdirectory()
        if export_path == "":
            return

        self.export_path_label.configure(text=export_path)
        path = os.path.split(self.driver.input_path)
        export_path = os.path.join(export_path, "_" + path[1])
        self.driver.output_path = export_path
        self.export_path = os.path.join(export_path, "blur_" + path[1])
        if not self.export_btn.winfo_ismapped():
            self.export_btn.pack(pady=20)

    def exportVideo(self):
        if self.driver.output_path == "":
            return

        if self.driver.mode == 0:
            self.thread = threading.Thread(
                target=self.driver.blur_except_output, name="blur"
            )
            self.thread.start()
        else:
            self.thread = threading.Thread(
                target=self.driver.blur_all_output(), name="blur"
            )
            self.thread.start()
