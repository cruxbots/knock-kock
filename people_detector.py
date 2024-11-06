from ultralytics import YOLO
from pathlib import Path
from playsound import playsound
import cv2
import numpy as np
import random
import multiprocessing

class FrameRunner:

    def __init__(self, process, end_function = lambda: None) -> None:

        self.process = process
        self.cam = cv2.VideoCapture(4)
        self.end_function = end_function

    def runner(self):

        while True:

            ret, frame = self.cam.read()
            self.process(frame)
            if cv2.waitKey(1) == ord('q'):
                break
        cv2.destroyAllWindows()
        self.end_function()

class Actions:

    def __init__(self) -> None:
        
        self.model = YOLO("yolo11n.pt")
        self.voice_box_path: Path = Path("voice_box")
        self.person_flag = multiprocessing.Value('i')
        self.action_flag = multiprocessing.Value('i')
        self.action_flag.value = 1
    
    def people_predictor(self, frame: np.ndarray) -> None:

        result = self.model.predict(frame,
                                classes = [0],
                                conf = 0.5,
                                verbose = False
                                )
        cv2.imshow("live stream",
                result[0].plot())
        
        if len(result[0].boxes.cls) >= 1: self.person_flag.value = 1
        else: self.person_flag.value = 0
    
    def run_actions(self) -> None:

        while True:
            self.shout()

            if self.action_flag.value == 0:
                break
    
    def end(self):

        self.action_flag.value = 0

    
    def shout(self):

        voice_path = self.voice_box_path
        voice_paths = list(self.voice_box_path.iterdir())
        voice_path = voice_paths[random.randint(0, len(voice_paths)) - 1]
        
        if self.person_flag.value == 1:
            playsound(voice_path)


if __name__ == "__main__":

    act = Actions()

    runner = FrameRunner(process=act.people_predictor, 
                        end_function= act.end)

    frame_process = multiprocessing.Process(
        target=runner.runner
    )
    sound_process = multiprocessing.Process(
        target=act.run_actions
    )

    sound_process.start()
    frame_process.start()
    
    sound_process.join()
    frame_process.join()
    