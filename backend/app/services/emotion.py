import sys
import os

# Add project root to path so AIModels can be resolved
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from AIModels.emotion_detection import EmotionCascadeSystem

cascade_system = None

def get_emotion_system():
    global cascade_system
    if cascade_system is None:
        cascade_system = EmotionCascadeSystem()
        models_path = os.path.join(project_root, "AIModels", "emotion_models")
        cascade_system.load_models(base_path=models_path)
    return cascade_system

def detect_emotion(text: str) -> str:
    system = get_emotion_system()
    try:
        prediction = system.predict(text)
        return prediction.lower()
    except Exception as e:
        print(f"Emotion prediction error: {e}")
        return "neutral"
