try:
    import mediapipe.solutions.hands as mp_hands
    print("Success: mediapipe.solutions.hands imported")
    print(mp_hands)
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Other error: {e}")
