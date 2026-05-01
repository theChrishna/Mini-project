import mediapipe as mp
print(dir(mp))
try:
    print(mp.solutions)
except AttributeError:
    print("No solutions attribute")
