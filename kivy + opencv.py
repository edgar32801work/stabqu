import cv2
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock


class VideoStreamWidget(Image):
    def __init__(self, capture, **kwargs):
        super(VideoStreamWidget, self).__init__(**kwargs)
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # Обновление с частотой 30 кадров в секунду

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            # Конвертирование BGR-изображения в RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Обновление изображения в виджете
            self.texture = self.create_texture(frame)

    def create_texture(self, frame):
        # Создание текстуры для отображения изображения
        texture = self.texture
        if not texture:
            texture = self.texture = Texture.create(size=(frame.shape[1], frame.shape[0]))
            texture.flip_vertical()  # Переворачиваем изображение вертикально

        texture.blit_buffer(frame.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        return texture


class VideoStreamApp(App):
    def build(self):
        # Создание объекта захвата видео
        capture = cv2.VideoCapture(0)

        # Создание виджета видеопотока
        video_stream_widget = VideoStreamWidget(capture=capture)

        return video_stream_widget

    def on_stop(self):
        # Освобождение ресурсов захвата видео при закрытии приложения
        self.root.capture.release()


if __name__ == '__main__':
    VideoStreamApp().run()
