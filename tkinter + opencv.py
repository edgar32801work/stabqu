import cv2
import tkinter as tk
from PIL import Image, ImageTk


class VideoStreamApp:
    def __init__(self, window, video_source=0):
        self.window = window
        self.window.title("Video Stream")

        self.video_source = video_source
        self.vid = VideoCapture(self.video_source)

        self.canvas = tk.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()

        self.btn_start = tk.Button(window, text="Start", command=self.start_video_stream)
        self.btn_start.pack(side=tk.LEFT)

        self.btn_stop = tk.Button(window, text="Stop", command=self.stop_video_stream)
        self.btn_stop.pack(side=tk.LEFT)

        self.window.mainloop()

    def start_video_stream(self):
        self.update()

    def stop_video_stream(self):
        self.window.quit()

    def update(self):
        ret, frame = self.vid.get_frame()

        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(15, self.update)


class VideoCapture:
    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None
        else:
            return ret, None


if __name__ == '__main__':
    # Создание окна Tkinter
    root = tk.Tk()

    # Установка размера окна
    root.geometry("700x500")

    # Создание приложения видеопотока
    app = VideoStreamApp(root)

