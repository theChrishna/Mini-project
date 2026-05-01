# ASL Voice Translator — Gesture & Sign Reference
> **Version:** Post Phase 1 + Phase 2  
> **Last updated:** 2026-04-29  
> **Recognition system:** 3-tier (MediaPipe Gesture → Word Signs → Letter Fallback)

---

## How Recognition Works

```
Your hand gesture
      │
      ▼
Tier 1 — MediaPipe GestureRecognizer  ──► matched?  → WORD (high accuracy)
      │ no match
      ▼
Tier 2 — Position + Motion Rules      ──► matched?  → WORD (medium accuracy)
      │ no match
      ▼
Tier 3 — Letter Classifier (A–Z)      ──────────►   LETTER (fallback)
```

Hold any sign steady for **~7 frames (~0.23s)** to register.  
After **2.5s of silence** with signs in the buffer → auto-speaks.

---

## Tier 1 — MediaPipe Built-in Gestures

> High accuracy, model-based. Fires before any other classifier.  
> Minimum confidence threshold: **70%** — shown as `Sign: YES  (94%)` in UI.

| Gesture | How to Perform | Output |
|---------|---------------|--------|
| 👍 **Thumbs Up** | Fist with thumb pointing straight up | `YES` |
| 👎 **Thumbs Down** | Fist with thumb pointing straight down | `NO` |
| ✋ **Open Palm** | All 5 fingers extended, palm facing outward | `STOP` |
| ☝️ **Pointing Up** | Index finger straight up, all others curled | `ATTENTION` |
| ✌️ **Victory** | Index + middle up, spread apart (V shape) | `PEACE` |
| 🤟 **ILY Sign** | Thumb + index + pinky extended | `I LOVE YOU` |
| ✊ **Closed Fist** | All fingers curled, thumb over fingers | `HOLD ON` |

---

## Tier 2 — Position + Motion Word Signs

> Uses where your hand is on your body **+** how it moves.  
> Requires face detection to be ON (`E` key) for body zones to work.

### Zone Map
```
Your face on screen (y = 0 top, 1 bottom):

  ┌──────────────────────────┐
  │  FOREHEAD  (near brow)   │  ← used for HELLO, SICK
  │  FACE      (eye/nose)    │  ← GOODBYE, EAT, CALL
  │  CHIN      (mouth/jaw)   │  ← THANK YOU, GOOD
  │  CHEST     (below chin)  │  ← PLEASE, SORRY, FINE, WAIT
  └──────────────────────────┘
```

### Motion Types
| Motion | What it means |
|--------|--------------|
| `STATIC` | Hand barely moves — held in place |
| `LATERAL` | Hand sweeps left/right |
| `VERTICAL` | Hand moves up/down |
| `CIRCULAR` | Hand moves in a circle |

### Word Sign Table

| Output | Handshape | Zone | Motion | How to Perform |
|--------|-----------|------|--------|---------------|
| **HELLO** | Open (B) | FOREHEAD | LATERAL | Raise flat hand to forehead, sweep sideways like a salute |
| **SICK** | Open (B) | FOREHEAD | STATIC | Touch flat hand to forehead and hold still |
| **GOODBYE** | Open (B) | FACE | LATERAL | Wave open hand side to side at face level |
| **THANK YOU** | Open (B) | CHIN | VERTICAL | Touch fingertips to chin, move hand downward |
| **GOOD** | Open (B) | CHIN | STATIC | Flat hand held at chin level, still |
| **PLEASE** | Open (B) | CHEST | CIRCULAR | Flat hand on chest, rub in a clockwise circle |
| **FINE** | Open (B) | CHEST | STATIC | Flat hand at chest, fingers spread, held still |
| **WAIT** | Open (B) | CHEST | LATERAL | Open hand at chest, wiggle side to side |
| **SORRY** | Fist (A/S) | CHEST | CIRCULAR | Closed fist on chest, rub in a circle |
| **MORE** | Fist | NEUTRAL/BELLY | VERTICAL | Fist below chest, tap up and down |
| **EAT** | Fist | FACE | STATIC | Bunched hand near mouth, hold still |
| **CALL** | Y-shape (thumb+pinky) | FACE | STATIC | Thumb+pinky extended (phone shape) near ear |
| **DANGER** | Thumb+index (L) | any | LATERAL | L-shaped hand, thrust sideways quickly |

> **Note:** If emotion detection (`E`) is OFF, face zones won't work — Tier 2 word signs won't fire.
> Tier 1 and Tier 3 still work regardless.

---

## Tier 3 — Letter Classifier (A–Z Fallback)

> **[DISABLED]** This feature was removed per user request. The system now only recognizes full word signs.

| Letter | How to Form It | Notes |
|--------|---------------|-------|
| **A** | Fist, thumb beside index pointing up | Clean, works well |
| **B** | All 4 fingers straight up, thumb folded | Also triggers if open palm not confident enough for Tier 1 |
| **C** | Fingers curved in C shape, thumb parallel | Moderate curl required |
| **D** | Index up, thumb + others form O base | Thumb near middle fingertip |
| **E** | All fingers bent down, thumb tucked | Tight curl required |
| **F** | Index+thumb pinch, other 3 fingers up | Thumb near index tip |
| **G** | Index pointing sideways, thumb same direction | |
| **I** | Pinky only up | Clean, works well |
| **K** | Index+middle up, thumb between them | Thumb y between index and middle |
| **L** | Thumb out + index up (L-shape) | Clean, works well |
| **O** | All fingers curved to form O | Thumb near index tip, all curled |
| **P** | Middle finger only up | |
| **R** | Index+middle crossed | Middle x overlaps index |
| **S** | Tight fist, thumb over fingers | Thumb not extended |
| **T** | Thumb tucked between index+middle pips | |
| **U** | Index+middle up, close together | Small gap between fingers |
| **V** | Index+middle up, spread apart | Wide gap between fingers |
| **W** | Index+middle+ring up | |
| **X** | Index bent/hooked (tip lower than pip) | |
| **Y** | Thumb + pinky extended | |
| **4** | 4 fingers up, no thumb | |

### Not Supported
| Letter | Reason |
|--------|--------|
| H | Sideways orientation needs wrist rotation data |
| J | Motion-based (drawing J) |
| M | Overlaps with S/E in boolean detection |
| N | Same issue as M |
| Q | Downward index conflicts with other signs |
| Z | Motion-based (drawing Z) |

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| `H` | Toggle hand tracking ON/OFF |
| `E` | Toggle emotion + face zone detection ON/OFF |
| `L` | Toggle LLM translation ON/OFF |
| `V` | Toggle voice (TTS) ON/OFF |
| `S` | **Manually speak** current buffer now |
| `C` | Clear the buffer |
| `Q` | Quit |

---

## Quick Reference Card

```
╔══════════════════════════════════════════════════════════╗
║         ASL VOICE TRANSLATOR — GESTURE CHEATSHEET        ║
╠══════════════════════════════════════════════════════════╣
║  TIER 1 — HOLD GESTURE AND WAIT                          ║
║  👍 Thumb Up    → YES                                    ║
║  👎 Thumb Down  → NO                                     ║
║  ✋ Open Palm   → STOP                                   ║
║  ☝️  Point Up   → ATTENTION                              ║
║  ✌️  Victory    → PEACE                                  ║
║  🤟 ILY Sign   → I LOVE YOU                             ║
║  ✊ Closed Fist → HOLD ON                               ║
╠══════════════════════════════════════════════════════════╣
║  TIER 2 — POSITION + MOTION (face detection must be ON)  ║
║  HELLO    → Flat hand at forehead, sweep sideways        ║
║  SICK     → Flat hand at forehead, hold still            ║
║  GOODBYE  → Open hand wave at face level                 ║
║  THANK YOU→ Fingers at chin, move down                   ║
║  GOOD     → Flat hand at chin, still                     ║
║  PLEASE   → Flat hand on chest, rub in circle            ║
║  SORRY    → Fist on chest, rub in circle                 ║
║  MORE     → Fist below chest, tap up/down                ║
║  EAT      → Bunched hand near mouth, still               ║
║  CALL     → Thumb+pinky (phone) near ear, still          ║
║  WAIT     → Open hand at chest, wiggle sideways          ║
║  FINE     → Open hand at chest, still                    ║
║  DANGER   → L-shape hand, thrust sideways                ║
╠══════════════════════════════════════════════════════════╣
║  TIER 3 — LETTERS (fallback when no word matched)        ║
║  A B C D E F G I K L O P R S T U V W X Y 4              ║
╠══════════════════════════════════════════════════════════╣
║  KEYS: [S] Speak  [C] Clear  [Q] Quit                    ║
║        [H] Hand   [E] Emotion  [V] Voice  [L] LLM        ║
╚══════════════════════════════════════════════════════════╝
```
