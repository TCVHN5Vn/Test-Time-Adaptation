# Activity label mapping  {raw_value -> class_index}
ACTIVITY_MAP     = {15.0: 0, 60.0: 1, 65.0: 2, 70.0: 3, 85.0: 4}
INV_ACTIVITY_MAP = {v: k for k, v in ACTIVITY_MAP.items()}
NUM_CLASSES      = 5
CLASS_NAMES      = ['Other(15)', 'Cook(60)', 'Sleep(65)', 'Bathe(70)', 'Toilet(85)']
