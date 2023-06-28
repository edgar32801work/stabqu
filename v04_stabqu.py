"""
- Добавить фичу статическая маска (чтобы траектории не исчезали)
- Сокращение области поиска точек можно реализовать через маску goodfeaturestotrack
"""

import cv2 as cv
import numpy as np

import tkinter as tk
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from collections import deque

BIAS_LINE_COLOR = (255, 0, 0)


class VideoStreamApp:
    def __init__(self, window, video_source=0):
        self.window = window
        self.window.title("Video Stream")

        # создание объекта видеозахвата
        self.video_source = video_source
        self.video = VideoCapture(self.video_source, BIAS_LINE_COLOR)

        # первый блок - управление
        self.button_frame = tk.Frame(window)
        self.button_frame.pack(side=tk.LEFT, fill='both')

        self.btn_start = tk.Button(self.button_frame, text="Start", command=self.start_video_stream)
        self.btn_start.pack(side=tk.TOP)

        self.btn_stop = tk.Button(self.button_frame, text="Stop", command=self.stop_video_stream)
        self.btn_stop.pack(side=tk.TOP)

        # -1 ограничитель пустого фрейма
        self.tscale = tk.Scale(self.button_frame, from_=0, to=self.video.height/2-40, orient=tk.VERTICAL)
        self.tscale.pack(side=tk.TOP)

        self.lscale = tk.Scale(self.button_frame, from_=0, to=self.video.width/2-40, orient=tk.HORIZONTAL)
        self.lscale.pack(side=tk.LEFT)

        position = tk.IntVar()
        position.set(1080)
        self.rscale = tk.Scale(self.button_frame, from_=self.video.width/2+40, to=self.video.width, orient=tk.HORIZONTAL, variable=position)
        self.rscale.pack(side=tk.RIGHT)

        position = tk.IntVar()
        position.set(1920)
        self.bscale = tk.Scale(self.button_frame, from_=self.video.height/2+40, to=self.video.height, orient=tk.VERTICAL, variable=position)
        self.bscale.pack(side=tk.BOTTOM)

        # блок вывода видео
        self.canvas = tk.Canvas(window, width=self.video.width, height=self.video.height)
        self.canvas.pack(side=tk.LEFT)

        # блок отрисовки графиков
        self.plot = Plot()
        self.canvas_graph = FigureCanvasTkAgg(self.plot.figure, master=window)
        self.canvas_graph.draw()
        self.canvas_graph.get_tk_widget().pack(side=tk.LEFT)

        self.window.mainloop()

    def start_video_stream(self):
        _, prev_frame = self.video.capture.read()
        prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
        prev_points = cv.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.2, minDistance=7)
        prev_data = (prev_gray, prev_points)
        self.update(prev_data)

    def stop_video_stream(self):
        self.video.capture.release()
        cv.destroyAllWindows()
        self.window.quit()

    def update(self, prev_data):

        rbias = self.rscale.get()
        lbias = self.lscale.get()
        tbias = self.tscale.get()
        bbias = self.bscale.get()
        ret, frame, prev_data, shift = self.video.get_frame(prev_data, bias=(rbias, lbias, tbias, bbias))

        self.plot.renovate(x1_shift=shift[0][0], x2_shift=shift[0][1])
        self.canvas_graph.draw()

        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(15, self.update, prev_data)


class VideoCapture:
    def __init__(self, source, BIAS_LINE_COLOR):
        self.capture = cv.VideoCapture(source)
        assert self.capture.isOpened(), 'Capture isnt opened'

        self.width = int(self.capture.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.bias_line_color = BIAS_LINE_COLOR

    def get_frame(self, prev_data, bias):
        prev_gray, prev_point = prev_data
        rbias, lbias, tbias, bbias = bias
        if self.capture.isOpened():
            ret, frame = self.capture.read()

            feature_mask = np.zeros_like(prev_gray)
            feature_mask[tbias: bbias, lbias: rbias] = 1

            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            prev_points = cv.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.2, minDistance=7, mask = feature_mask)
            next_points, _, _ = cv.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_points, None)

            # отрисовка точек на кадр
            mask = np.zeros_like(frame)
            for i, (prev_pt, next_pt) in enumerate(zip(prev_points, next_points)):
                x_prev, y_prev = prev_pt.ravel().astype(int)
                x_next, y_next = next_pt.ravel().astype(int)
                mask = cv.line(mask, pt1=(x_next, y_next), pt2=(x_prev, y_prev), color=(0, 255, 0), thickness=2)
                frame = cv.circle(frame, (x_next, y_next), 5, (0, 0, 255), -1)

            # затемнение вокруг области поиска
            feature_mask = np.where(feature_mask == 0, 0.4, feature_mask)
            feature_mask = feature_mask[:, :, np.newaxis]
            result = cv.add(frame, mask)
            result = np.multiply(result, feature_mask)
            result = cv.convertScaleAbs(result)

            shift = np.mean((next_points - prev_points), axis=0)

            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            result = cv.cvtColor(result, cv.COLOR_BGR2RGB)
            prev_data = (frame, next_points)
            return ret, result, prev_data, shift

    # def resize(self, frame, prev_gray, bias):
    #
    #
    #     if frame.shape == frame[tbias: bbias, lbias: rbias].shape and prev_gray.shape == prev_gray[tbias: bbias, lbias: rbias].shape:
    #         print(True)




class Plot:
    def __init__(self):
        self.figure, axes = plt.subplots(3, figsize=(10, 6))

        self.x1 = deque(np.zeros(40), maxlen=40)
        self.x2 = deque(np.zeros(40), maxlen=40)
        self.x3 = deque(np.zeros(40), maxlen=40)
        self.x1std = np.round(np.std(self.x1), 3)
        self.x2std = np.round(np.std(self.x2), 3)
        self.x3std = np.round(np.std(self.x3), 3)

        for i in range(3):
            axes[i].set_ylim(-80, 80)
        self.x1text = axes[0].text(35, 50, f'СКО: {self.x1std}')
        self.x1line, = axes[0].plot(self.x1)
        self.x2text = axes[1].text(35, 50, f'СКО: {self.x2std}')
        self.x2line, = axes[1].plot(self.x2)
        self.x3text = axes[2].text(35, 50, f'СКО: {self.x3std}')
        self.x3line, = axes[2].plot(self.x3)

    def renovate(self, x1_shift, x2_shift):
        self.x1.append(x1_shift)
        self.x2.append(x2_shift)
        self.x1std = np.round(np.std(self.x1), 3)
        self.x2std = np.round(np.std(self.x2), 3)
        self.x3std = np.round(np.std(self.x3), 3)

        self.x1text.set_text(f'СКО: {self.x1std}')
        self.x2text.set_text(f'СКО: {self.x2std}')
        self.x1line.set_ydata(self.x1)
        self.x2line.set_ydata(self.x2)


if __name__ == '__main__':
    # Создание окна Tkinter
    root = tk.Tk()

    # Установка размера окна
    root.geometry("1000x1100")

    # Создание приложения видеопотока
    app = VideoStreamApp(root)

# input('- press "Enter" to close the window ...')
