import config
import requests
import json

class LLMInterpreter:
    def __init__(self):
        self.available = False
        self.api_key = config.GEMINI_API_KEY
        if self.api_key:
            self.available = True
            print("[LLM] Gemini REST API initialized successfully.")
        else:
            print("[LLM] Gemini API key not found in config.py. Using fallback.")

    def translate(self, signs, emotion):
        """
        Sends raw ASL sign sequence and detected emotion to Gemini via REST API.
        Falls back to a simple passthrough if Gemini is not available or fails.
        """
        if not config.ENABLE_LLM_TRANSLATION or not self.available:
            return self._fallback_translation(signs, emotion)

        if not signs:
            return ""

        prompt = (
            f"You are an expert ASL-to-English translator. I will provide you with a list of raw ASL glosses "
            f"(signs) and the current facial emotion of the signer.\n\n"
            f"Your job is to treat these signs as a 'bag of words' and use them to construct a single, "
            f"natural, grammatically correct English sentence. The original order of the signs does not matter.\n\n"
            f"Rules:\n"
            f"1. You may rearrange the words freely to make the most logical sentence.\n"
            f"2. Break apart compound signs if needed (e.g., 'I LOVE YOU' can be treated as individual words 'I', 'love', 'you').\n"
            f"3. Add necessary articles, prepositions, and verb conjugations to make it sound natural.\n"
            f"4. Use the emotion to determine the tone and punctuation (e.g., Happy -> '!`, Questioning -> '?').\n"
            f"5. Do not hallucinate entirely new subjects or meanings not present in the signs.\n"
            f"6. ONLY return the final English sentence. Do not include any explanations, greetings, or formatting.\n\n"
            f"Input:\n"
            f"Signs: {signs}\n"
            f"Emotion: {emotion}\n\n"
            f"Translation:"
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.api_key}"
        headers = {'Content-Type': 'application/json'}
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if "candidates" in data and len(data["candidates"]) > 0:
                    candidate = data["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        text = candidate["content"]["parts"][0]["text"]
                        return text.strip()
                    else:
                        print(f"[LLM] Gemini API returned no text. Reason: {candidate.get('finishReason', 'Unknown')}")
                        print(f"[LLM] Full response: {json.dumps(data)}")
                else:
                    print("[LLM] Gemini API returned no candidates.")
            else:
                print(f"[LLM] Gemini API error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"[LLM] Gemini request failed: {e}")

        return self._fallback_translation(signs, emotion)

    def _fallback_translation(self, signs, emotion):
        """
        Passthrough fallback: joins sign tokens into a readable sentence.
        """
        if not signs:
            return ""
        return f"[RAW] {' '.join(signs.split())} [{emotion}]"
