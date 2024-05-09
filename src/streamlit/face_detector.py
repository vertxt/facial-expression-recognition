from mtcnn import MTCNN
# import YOLO, RetinaFace and DLib here

class FaceDetector:
    def __init__(self, model_type):
        self.model_type = model_type
        if model_type == 'mtcnn':
            self.model = MTCNN()
        elif model_type == 'retinaface':
            pass
        elif model_type == 'yolo':
            pass
        elif model_type == 'dlib':
            pass
        else:
            raise ValueError(f'Unsupported model type: {model_type}')
        
    def detect_faces(self, img):
        if self.model_type == 'mtcnn':
            return self.model.detect_faces(img)
        elif self.model_type == 'retinaface':
            pass
        elif self.model_type == 'yolo':
            pass
        elif self.model_type == 'dlib':
            pass
    
    def get_bounding_box(self, detected_faces, threshold=0.5):
        bounding_boxes = []
        num_detected_faces = len(detected_faces)
        if num_detected_faces:
            for face in detected_faces:
                if face['confidence'] >= threshold:
                    if self.model_type == 'mtcnn':
                        bounding_boxes.append(face['box'][0:4])
                    elif self.model_type == 'retinaface':
                        pass
                    elif self.model_type == 'yolo':
                        pass
                    elif self.model_type == 'dlib':
                        pass
        return bounding_boxes
            
        
            